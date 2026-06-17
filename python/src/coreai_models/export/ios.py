# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
iOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram for iOS.
The iOS export produces 4 entrypoints:
- load_embeddings: returns the embedding table
- gather_embeddings: token IDs -> embedded representations
- extend: single forward pass (decode mode)
- prompt_opt: forward pass in prefill mode
"""

import logging

import torch
from coreai.authoring import AIProgram
from coreai.authoring.types import AllocationType, HardwareConstraints
from coreai_torch import TorchConverter

from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# iOS graph I/O names (must match what the Swift runner expects)
# ---------------------------------------------------------------------------

KV_CACHE_INTERLEAVE_FACTOR = 8

LOAD_EMBEDDINGS_FUNCTION_NAME = "load_embeddings"
GATHER_EMBEDDINGS_FUNCTION_NAME = "gather_embeddings"
EXTEND_FUNCTION_NAME = "extend"
PROMPT_OPT_FUNCTION_NAME = "prompt_opt"

EMBEDDING_TABLE_INPUT_NAME = "embedding_table"
LOAD_EMBEDDINGS_OUTPUT_NAME = "embedding_table"
TOKEN_IDS_INPUT_NAME = "in_new_token_ids"
GATHERED_EMBEDDINGS_OUTPUT_NAME = "gathered_embeddings"

TRANSFORMER_INPUT_NAME = "transformer_input"
POSITION_IDS_INPUT_NAME = "position_ids"
IN_STEP_INPUT_NAME = "in_step"
CAUSAL_MASK_INPUT_NAME = "causal_mask"
KEY_CACHE_INPUT_NAME = "key_cache"
VALUE_CACHE_INPUT_NAME = "value_cache"
KEY_CACHE_OUTPUT_NAME = "new_k_cache"
VALUE_CACHE_OUTPUT_NAME = "new_v_cache"
OUTPUT_LOGITS_NAME = "out_logits"

# Sliding-window attention (Gemma4): a second compacted ring cache plus a
# runner-built windowed mask. Present only for models whose decoder declares a
# ``sliding_cache`` (see ``models/ios/gemma4_text.py``).
SLIDING_CAUSAL_MASK_INPUT_NAME = "sliding_causal_mask"
SLIDING_IN_STEP_INPUT_NAME = "sliding_in_step"
SLIDING_KEY_CACHE_INPUT_NAME = "sliding_key_cache"
SLIDING_VALUE_CACHE_INPUT_NAME = "sliding_value_cache"
SLIDING_KEY_CACHE_OUTPUT_NAME = "new_sliding_k_cache"
SLIDING_VALUE_CACHE_OUTPUT_NAME = "new_sliding_v_cache"

# Blocked global KV cache (Gemma4 large-context): the global cache is stored
# block-outermost (a fixed block_size sequence axis) and attention is computed
# block-by-block with flash / online softmax, so each tensor stays within the
# Neural Engine's supported sizes. The runner supplies the destination block index
# (in_step // block_size). The global ``causal_mask`` becomes blocked too:
# (1, n_blocks, block_size, 1, q_len).
GLOBAL_BLOCK_IDX_INPUT_NAME = "global_block_idx"
DEFAULT_KV_BLOCK_SIZE = 8192

# Per-function query-length ladder for iOS static-shape specialization.
QUERY_LENGTHS = [8, 16, 64]


def sliding_ring_size(sliding_window: int, max_query_len: int) -> int:
    """Minimal sliding-cache ring depth S that holds a full prefill chunk's
    window union and never wraps a contiguous q_len write.

    S >= sliding_window + max_query_len - 1, rounded up to a multiple of
    max_query_len (so an in_step aligned to q_len keeps every write within
    ``[in_step % S, in_step % S + q_len)``).
    """
    raw = sliding_window + max_query_len - 1
    return ((raw + max_query_len - 1) // max_query_len) * max_query_len



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ios_reference_inputs(
    model: torch.nn.Module,
    config,
    max_context_length: int,
    vocab_size: int,
) -> dict:
    """Build reference input tensors for iOS model export.

    Returns a dict with all inputs needed by the extend function, plus
    the embed_tokens inputs and dynamic shapes for each entrypoint.
    """
    batch_size = 1
    query_len = 8

    input_ids = torch.randint(1, vocab_size, (batch_size, query_len), dtype=torch.int32)
    position_ids = (
        torch.arange(query_len).to(torch.uint16).unsqueeze(0).expand(batch_size, query_len)
    )
    in_step = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros(1, max_context_length, 1, query_len, dtype=torch.float16)

    if hasattr(config, "head_dim") and isinstance(config.head_dim, int):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    global_head_dim = getattr(config, "global_head_dim", None)

    # Two compacted caches (Gemma4 sliding-window design) vs the single cache
    # used by every other iOS model.
    has_sliding = hasattr(model, "extend") and hasattr(model.extend, "sliding_cache")

    if has_sliding:
        n_kv = config.num_key_value_heads
        n_global_storing = model.extend.model.n_global_storing
        n_sliding_storing = model.extend.model.n_sliding_storing
        global_channels = n_kv * global_head_dim
        sliding_channels = n_kv * head_dim
        sliding_ring = sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS))

        # Blocked global cache: block-outermost 5D state with a fixed block_size
        # sequence axis, so each tensor stays within the Neural Engine's supported
        # sizes. 5D (not 6D) matches the flat cache's rank, which keeps it resident.
        block_size = getattr(model.extend.model, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
        n_blocks = (max_context_length + block_size - 1) // block_size

        key_cache = torch.zeros(
            n_blocks, n_global_storing, global_channels, 1, block_size, dtype=torch.float16
        )
        value_cache = key_cache.clone()
        sliding_key_cache = torch.zeros(
            n_sliding_storing, 1, sliding_channels, 1, sliding_ring, dtype=torch.float16
        )
        sliding_value_cache = sliding_key_cache.clone()
        sliding_causal_mask = torch.zeros(1, sliding_ring, 1, query_len, dtype=torch.float16)
        # Global mask is blocked too: one (block_size, q_len) tile per block.
        causal_mask = torch.zeros(1, n_blocks, block_size, 1, query_len, dtype=torch.float16)
    else:
        # Merged dual cache for models with global_head_dim, single cache otherwise.
        if global_head_dim is not None:
            kv_channels = config.num_key_value_heads * (head_dim + global_head_dim)
        else:
            kv_channels = config.num_key_value_heads * head_dim

        key_cache = torch.zeros(
            config.num_hidden_layers, 1, kv_channels, 1, max_context_length, dtype=torch.float16
        )
        value_cache = key_cache.clone()

    # Generate embeddings from the model
    embedding_table = model.load_embeddings.embedding_table
    transformer_input = model.gather_embeddings(input_ids, embedding_table)

    forward_inputs = {
        TRANSFORMER_INPUT_NAME: transformer_input,
        POSITION_IDS_INPUT_NAME: position_ids,
        IN_STEP_INPUT_NAME: in_step,
    }
    if has_sliding:
        # Runner-provided sliding ring write offset (in_step % S); the modulo is
        # computed in the runner and passed in rather than done in the graph.
        forward_inputs[SLIDING_IN_STEP_INPUT_NAME] = torch.zeros((1,), dtype=torch.int32)
        # Runner-provided destination block index for the global cache write
        # (in_step // block_size); the in-block offset is derived in-graph as
        # in_step - block_idx*block_size (mul+sub, never an in-graph modulo).
        forward_inputs[GLOBAL_BLOCK_IDX_INPUT_NAME] = torch.zeros((1,), dtype=torch.int32)
    forward_inputs[CAUSAL_MASK_INPUT_NAME] = causal_mask
    if has_sliding:
        forward_inputs[SLIDING_CAUSAL_MASK_INPUT_NAME] = sliding_causal_mask
    forward_inputs[KEY_CACHE_INPUT_NAME] = key_cache
    forward_inputs[VALUE_CACHE_INPUT_NAME] = value_cache
    if has_sliding:
        forward_inputs[SLIDING_KEY_CACHE_INPUT_NAME] = sliding_key_cache
        forward_inputs[SLIDING_VALUE_CACHE_INPUT_NAME] = sliding_value_cache
    forward_inputs[EMBEDDING_TABLE_INPUT_NAME] = embedding_table

    # PLE input for models with externalized per-layer embeddings (e.g. Gemma4)
    ple_dim_per_layer = getattr(config, "hidden_size_per_layer_input", 0)
    if ple_dim_per_layer and hasattr(model, "_ple_weight"):
        ple_total_dim = config.num_hidden_layers * ple_dim_per_layer
        ple_embeddings = torch.randint(
            -128, 127, (batch_size, query_len, 1, ple_total_dim), dtype=torch.int8
        )
        forward_inputs["ple_embeddings"] = ple_embeddings

    embed_tokens_inputs = (input_ids, embedding_table)

    seq_len_dim = torch.export.Dim("seq_len", max=max_context_length)
    cache_len_dim = torch.export.Dim("cache_len", max=max_context_length)

    forward_dynamic_shapes = {
        TRANSFORMER_INPUT_NAME: {1: seq_len_dim},
        POSITION_IDS_INPUT_NAME: {1: seq_len_dim},
        IN_STEP_INPUT_NAME: None,
        CAUSAL_MASK_INPUT_NAME: {1: cache_len_dim, 3: seq_len_dim},
        KEY_CACHE_INPUT_NAME: {4: cache_len_dim},
        VALUE_CACHE_INPUT_NAME: {4: cache_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }
    if has_sliding:
        # Global cache + mask are blocked. block_size is static; the block count
        # scales with context. The block dim is kept STATIC per export (not a
        # torch.export Dim) — a fixed block count is what keeps the blocked global
        # cache resident on the Neural Engine; the query/seq dim stays dynamic.
        # A given export therefore pins n_blocks = ceil(max_ctx / block_size).
        # TODO: emit a per-context block ladder via separate statically-shaped
        # programs so smaller contexts use fewer blocks.
        forward_dynamic_shapes[KEY_CACHE_INPUT_NAME] = None
        forward_dynamic_shapes[VALUE_CACHE_INPUT_NAME] = None
        forward_dynamic_shapes[CAUSAL_MASK_INPUT_NAME] = {4: seq_len_dim}
        # Sliding mask: only the query dim is dynamic; the ring depth S is static.
        # Sliding caches are fully static (fixed ring depth), so no dynamic dims.
        forward_dynamic_shapes[SLIDING_IN_STEP_INPUT_NAME] = None
        forward_dynamic_shapes[GLOBAL_BLOCK_IDX_INPUT_NAME] = None
        forward_dynamic_shapes[SLIDING_CAUSAL_MASK_INPUT_NAME] = {3: seq_len_dim}
        forward_dynamic_shapes[SLIDING_KEY_CACHE_INPUT_NAME] = None
        forward_dynamic_shapes[SLIDING_VALUE_CACHE_INPUT_NAME] = None

    if ple_dim_per_layer and hasattr(model, "_ple_weight"):
        forward_dynamic_shapes["ple_embeddings"] = {1: seq_len_dim}

    embed_tokens_dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }

    return {
        "forward_inputs": forward_inputs,
        "embed_tokens_inputs": embed_tokens_inputs,
        "forward_dynamic_shapes": forward_dynamic_shapes,
        "embed_tokens_dynamic_shapes": embed_tokens_dynamic_shapes,
    }


def _export_ios_programs(
    model: torch.nn.Module,
    inputs: dict,
) -> tuple:
    """Export the 4 ExportedPrograms for the iOS entrypoints.

    Returns:
        Tuple of (extend_program, prompt_program, gather_program, load_program).
    """
    forward_inputs = inputs["forward_inputs"]
    embed_tokens_inputs = inputs["embed_tokens_inputs"]
    forward_dynamic_shapes = inputs["forward_dynamic_shapes"]
    embed_tokens_dynamic_shapes = inputs["embed_tokens_dynamic_shapes"]

    with torch.no_grad():
        # iOS decomp table: keep silu as-is
        decomp_table = torch.export.default_decompositions()
        decomp_table.pop(torch.ops.aten.silu.default)
        decomp_table.pop(torch.ops.aten.silu.out)

        logger.info("Exporting extend module...")
        extend_exported_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(extend_exported_program)

        model.set_prefill_mode(True)
        logger.info("Exporting extend module (prefill mode)...")
        prompt_exported_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(prompt_exported_program)

        logger.info("Exporting gather_embeddings module...")
        gather_embeddings_exported_program = torch.export.export(
            model.gather_embeddings,
            args=embed_tokens_inputs,
            dynamic_shapes=embed_tokens_dynamic_shapes,
        )

        logger.info("Exporting load_embeddings module...")
        load_embeddings_exported_program = torch.export.export(model.load_embeddings, args=tuple())

    return (
        extend_exported_program,
        prompt_exported_program,
        gather_embeddings_exported_program,
        load_embeddings_exported_program,
    )


async def _convert_to_coreai(
    extend_program: torch.export.ExportedProgram,
    prompt_program: torch.export.ExportedProgram,
    gather_embeddings_program: torch.export.ExportedProgram,
    load_embeddings_program: torch.export.ExportedProgram,
    max_context_length: int,
    kv_cached_embed_size: int,
    hidden_size: int,
    num_layers: int,
    has_ple: bool = False,
    ple_total_dim: int = 0,
    has_sliding: bool = False,
    sliding_ring: int = 0,
    n_sliding_storing: int = 0,
    sliding_channels: int = 0,
    block_size: int = 0,
    n_blocks: int = 0,
    n_global_storing: int = 0,
) -> AIProgram:
    """Convert exported programs to a single AIProgram with iOS constraints.

    For models with sliding-window attention (Gemma4), ``num_layers`` /
    ``kv_cached_embed_size`` describe the full-context *global* cache, and the
    ``sliding_*`` args describe the second fixed-depth ring cache.

    This function:
    1. Adds all 4 exported programs to a TorchConverter
    2. Sets static shape configs for iOS shape specialization
    3. Sets hardware constraints (IOSurface allocations, interleave factors)
    4. Runs optimization and resolve-llo-mapped-composites pass
    """
    converter = TorchConverter()
    register_custom_torch_lowering(converter)

    converter.add_exported_program(
        load_embeddings_program,
        input_names=[],
        output_names=[LOAD_EMBEDDINGS_OUTPUT_NAME],
        entrypoint_name=LOAD_EMBEDDINGS_FUNCTION_NAME,
    )

    converter.add_exported_program(
        gather_embeddings_program,
        input_names=[TOKEN_IDS_INPUT_NAME, EMBEDDING_TABLE_INPUT_NAME],
        output_names=[GATHERED_EMBEDDINGS_OUTPUT_NAME],
        entrypoint_name=GATHER_EMBEDDINGS_FUNCTION_NAME,
    )

    input_names = [
        TRANSFORMER_INPUT_NAME,
        POSITION_IDS_INPUT_NAME,
        IN_STEP_INPUT_NAME,
    ]
    if has_sliding:
        input_names.append(SLIDING_IN_STEP_INPUT_NAME)
        input_names.append(GLOBAL_BLOCK_IDX_INPUT_NAME)
    input_names.append(CAUSAL_MASK_INPUT_NAME)
    if has_sliding:
        input_names.append(SLIDING_CAUSAL_MASK_INPUT_NAME)
    input_names.append(EMBEDDING_TABLE_INPUT_NAME)
    if has_ple:
        input_names.append("ple_embeddings")
    state_names = [
        KEY_CACHE_INPUT_NAME,
        VALUE_CACHE_INPUT_NAME,
    ]
    if has_sliding:
        state_names += [SLIDING_KEY_CACHE_INPUT_NAME, SLIDING_VALUE_CACHE_INPUT_NAME]
    output_names = [
        OUTPUT_LOGITS_NAME,
    ]

    converter.add_exported_program(
        extend_program,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
        entrypoint_name=EXTEND_FUNCTION_NAME,
    )
    converter.add_exported_program(
        prompt_program,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
        entrypoint_name=PROMPT_OPT_FUNCTION_NAME,
    )

    coreai_program: AIProgram = converter.to_coreai()

    # ----- Static shape configs for iOS specialization -----
    query_lengths = QUERY_LENGTHS

    gather_static_cfg: dict[str, dict[str, tuple[int, ...]]] = {}
    for q_len in query_lengths:
        gather_static_cfg[f'"{q_len}"'] = {TOKEN_IDS_INPUT_NAME: (1, q_len)}

    forward_static_cfg: dict[str, dict[str, tuple[int, ...]]] = {}
    if has_sliding:
        # Blocked global cache: block_size is fixed and the block count is pinned by
        # the export (the block dim is static in the program — see the dynamic-shape
        # note in _build_ios_reference_inputs). So we emit one block count
        # (n_blocks = ceil(max_ctx/block_size)); only the query length is laddered.
        # The config key stays the total context (n_blocks*block_size).
        ctx = n_blocks * block_size
        for q_len in query_lengths:
            cfg = {
                TRANSFORMER_INPUT_NAME: (1, q_len, 1, hidden_size),
                POSITION_IDS_INPUT_NAME: (1, q_len),
                CAUSAL_MASK_INPUT_NAME: (1, n_blocks, block_size, 1, q_len),
                KEY_CACHE_INPUT_NAME: (
                    n_blocks, n_global_storing, kv_cached_embed_size, 1, block_size
                ),
                VALUE_CACHE_INPUT_NAME: (
                    n_blocks, n_global_storing, kv_cached_embed_size, 1, block_size
                ),
                SLIDING_CAUSAL_MASK_INPUT_NAME: (1, sliding_ring, 1, q_len),
                SLIDING_KEY_CACHE_INPUT_NAME: (
                    n_sliding_storing, 1, sliding_channels, 1, sliding_ring
                ),
                SLIDING_VALUE_CACHE_INPUT_NAME: (
                    n_sliding_storing, 1, sliding_channels, 1, sliding_ring
                ),
            }
            if has_ple:
                cfg["ple_embeddings"] = (1, q_len, 1, ple_total_dim)
            forward_static_cfg[f'"{ctx}_{q_len}"'] = cfg
    else:
        # Context-length ladder (256, 512, ... up to max_context_length). The runner
        # picks the smallest cache window covering the current position, so decode
        # doesn't always pay for the full max_context_length.
        cache_len = 256
        while cache_len <= max_context_length:
            for q_len in query_lengths:
                cfg = {
                    TRANSFORMER_INPUT_NAME: (1, q_len, 1, hidden_size),
                    POSITION_IDS_INPUT_NAME: (1, q_len),
                    CAUSAL_MASK_INPUT_NAME: (1, cache_len, 1, q_len),
                    KEY_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
                    VALUE_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
                }
                if has_ple:
                    # PLE seq dim must be specialized alongside transformer_input, else it
                    # stays dynamic in the otherwise-static specialized function.
                    cfg["ple_embeddings"] = (1, q_len, 1, ple_total_dim)
                forward_static_cfg[f'"{cache_len}_{q_len}"'] = cfg
            cache_len *= 2

    coreai_program.set_static_shape_config(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_static_cfg)
    coreai_program.set_static_shape_config(EXTEND_FUNCTION_NAME, forward_static_cfg)
    coreai_program.set_static_shape_config(PROMPT_OPT_FUNCTION_NAME, forward_static_cfg)

    # ----- Hardware constraints + optimization -----
    emb_table_constraints = HardwareConstraints(
        AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
    )
    if has_sliding:
        # Blocked global cache is 5D [n_blocks, n_slots, C, 1, block_size]: channel C
        # at dim 2 and seq = block_size at dim 4 — the SAME positions as the flat 5D
        # cache, so it reuses the proven channel interleave + big seq-stride alignment
        # (interleave * block_size) that keeps the cache ops ANE-resident.
        cache_constraints = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * block_size, 1],
        )
    else:
        cache_constraints = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * max_context_length, 1],
        )

    gather_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}
    forward_constraints = {
        EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints,
        KEY_CACHE_INPUT_NAME: cache_constraints,
        KEY_CACHE_OUTPUT_NAME: cache_constraints,
        VALUE_CACHE_INPUT_NAME: cache_constraints,
        VALUE_CACHE_OUTPUT_NAME: cache_constraints,
    }
    if has_sliding:
        # Same channel interleave; seq-dim alignment is relative to the fixed
        # ring depth S, not max_context_length.
        sliding_cache_constraints = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * sliding_ring, 1],
        )
        forward_constraints[SLIDING_KEY_CACHE_INPUT_NAME] = sliding_cache_constraints
        forward_constraints[SLIDING_KEY_CACHE_OUTPUT_NAME] = sliding_cache_constraints
        forward_constraints[SLIDING_VALUE_CACHE_INPUT_NAME] = sliding_cache_constraints
        forward_constraints[SLIDING_VALUE_CACHE_OUTPUT_NAME] = sliding_cache_constraints
    load_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}

    logger.info("Applying optimization passes...")
    coreai_program.set_hardware_constraints(LOAD_EMBEDDINGS_FUNCTION_NAME, load_constraints)
    coreai_program.set_hardware_constraints(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_constraints)
    coreai_program.set_hardware_constraints(EXTEND_FUNCTION_NAME, forward_constraints)
    coreai_program.set_hardware_constraints(PROMPT_OPT_FUNCTION_NAME, forward_constraints)
    coreai_program.optimize()

    return coreai_program


async def export_ios_model(
    model: torch.nn.Module,
    config,
    export_config,
) -> AIProgram:
    """Export an iOS model to a AIProgram.

    This is the main entry point for iOS model export. It:
    1. Builds reference inputs for all 4 entrypoints
    2. Exports each entrypoint through torch.export
    3. Converts to a single multi-function AIProgram with iOS constraints

    Args:
        model: A loaded PyTorch iOS model (already in the correct dtype).
            Must have ``extend``, ``gather_embeddings``, ``load_embeddings``
            submodules and a ``set_prefill_mode`` method.
        config: HuggingFace model config.
        export_config: An ExportConfig instance.

    Returns:
        An optimized AIProgram with 4 entrypoints, static shape
        configs, and hardware constraints set for iOS.
    """
    max_context_length = getattr(export_config, "max_context_length", None)
    if max_context_length is None:
        max_context_length = getattr(config, "max_position_embeddings", 2048)

    vocab_size = config.vocab_size

    logger.info(
        f"Exporting iOS model (max_context_length={max_context_length}, vocab_size={vocab_size})"
    )

    # 1. Build reference inputs
    inputs = _build_ios_reference_inputs(model, config, max_context_length, vocab_size)

    # 2. Export 4 programs
    (
        extend_program,
        prompt_program,
        gather_program,
        load_program,
    ) = _export_ios_programs(model, inputs)

    # 3. Convert to Core AI with iOS constraints
    if hasattr(config, "head_dim") and isinstance(config.head_dim, int):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    global_head_dim = getattr(config, "global_head_dim", None)

    has_ple = hasattr(model, "_ple_weight")
    ple_dim_per_layer = getattr(config, "hidden_size_per_layer_input", 0)
    ple_total_dim = config.num_hidden_layers * ple_dim_per_layer if has_ple else 0

    has_sliding = hasattr(model, "extend") and hasattr(model.extend, "sliding_cache")
    if has_sliding:
        # Global cache holds only the storing global layers, global channels, and is
        # blocked (6D, block-outermost) for large-context ANE residency.
        n_kv = config.num_key_value_heads
        num_layers = model.extend.model.n_global_storing
        kv_cached_embed_size = n_kv * global_head_dim
        block_size = getattr(model.extend.model, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
        n_blocks = (max_context_length + block_size - 1) // block_size
        sliding_kwargs = {
            "has_sliding": True,
            "sliding_ring": sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS)),
            "n_sliding_storing": model.extend.model.n_sliding_storing,
            "sliding_channels": n_kv * head_dim,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "n_global_storing": model.extend.model.n_global_storing,
        }
    else:
        if global_head_dim is not None:
            kv_cached_embed_size = config.num_key_value_heads * (head_dim + global_head_dim)
        else:
            kv_cached_embed_size = config.num_key_value_heads * head_dim
        num_layers = config.num_hidden_layers
        sliding_kwargs = {}

    coreai_program = await _convert_to_coreai(
        extend_program=extend_program,
        prompt_program=prompt_program,
        gather_embeddings_program=gather_program,
        load_embeddings_program=load_program,
        max_context_length=max_context_length,
        kv_cached_embed_size=kv_cached_embed_size,
        hidden_size=config.hidden_size,
        num_layers=num_layers,
        has_ple=has_ple,
        ple_total_dim=ple_total_dim,
        **sliding_kwargs,
    )

    return coreai_program

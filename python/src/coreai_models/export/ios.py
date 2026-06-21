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
# Gemma4 large-context (blocked-ladder) path: RoPE cos/sin are precomputed in the
# runner and passed as float16 graph inputs (the combined sliding+global table rows
# for the chunk's positions), replacing ``position_ids`` + the in-graph gather — a
# 131k position index overflows a 16-bit input and a 32-bit position input is not
# supported by the streaming compile path. All other iOS models keep ``position_ids``.
ROPE_COS_INPUT_NAME = "rope_cos"
ROPE_SIN_INPUT_NAME = "rope_sin"
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

# Large-context global attention (Gemma4): the global cache stays FLAT
# ``(n_global_storing, 1, C_g, 1, ctx)`` (one dynamic write offset), and the chunked
# flash ``BlockedSDPA`` walks the flat slot in ``block_size`` chunks so each score op's
# key axis stays within the accelerator's per-dimension size limit up to ctx ~32768.
# ``block_size`` is the flash chunk width, not a cache dimension.
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


# Smallest context bucket. Short prompts pay only this context (a single
# ctx-wide flash chunk), exactly like every other iOS model's ladder.
MIN_CONTEXT_LENGTH = 256


def context_ladder(max_context_length: int, block_size: int) -> list[int]:
    """Context-length buckets for the flat-cache flash ladder: powers of two from
    ``MIN_CONTEXT_LENGTH`` up to the smallest power of two that covers
    ``max_context_length``.

    Each context size is its own statically-shaped program (the flash block loop is
    unrolled, so its block count — ``ceil(ctx / block_size)`` — is baked into the
    graph and can't be a runtime specialization). The runner picks the smallest
    bucket whose context covers the current position, so a short prompt runs a small
    ``ctx``-wide graph instead of always paying the full max context.

    For ``ctx <= block_size`` the flash is a single chunk of width ``ctx`` (the
    small/fast buckets short prompts use); for ``ctx > block_size`` it unrolls
    ``ctx / block_size`` chunks of width ``block_size``. ``block_size`` only bounds
    each chunk's score key-axis (keeping it within the accelerator's per-dimension size
    limit); it is not a cache dimension or a floor on the smallest bucket.

    Examples (``block_size`` 8192): max 131072 -> [256, 512, ..., 65536, 131072];
    max 32768 -> [256, 512, ..., 32768]; max 512 -> [256, 512].
    """
    ctx_max = 1
    while ctx_max < max_context_length:
        ctx_max *= 2
    buckets: list[int] = []
    ctx = MIN_CONTEXT_LENGTH
    while ctx < ctx_max:
        buckets.append(ctx)
        ctx *= 2
    buckets.append(ctx_max)
    # Dev override: GEMMA4_LADDER_ONLY="131072" (comma-separated) emits only those buckets,
    # to isolate a single context's graphs (e.g. measure the 131072-only realize floor).
    import os as _os

    _only = _os.environ.get("GEMMA4_LADDER_ONLY")
    if _only:
        want = {int(x) for x in _only.split(",")}
        buckets = [b for b in buckets if b in want] or [ctx_max]
    return buckets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ios_reference_inputs(
    model: torch.nn.Module,
    config,
    max_context_length: int,
    vocab_size: int,
    ctx: int | None = None,
) -> dict:
    """Build reference input tensors for iOS model export.

    Returns a dict with all inputs needed by the extend function, plus
    the embed_tokens inputs and dynamic shapes for each entrypoint.

    For the flat-global-cache (sliding) path, ``ctx`` pins the global cache /
    mask context length for this export (one ladder bucket). When ``None`` it
    defaults to ``max_context_length``.
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

        # FLAT global cache `[n_global_storing, 1, C_g, 1, ctx]` (same layout as the
        # generic flat KVCacheHandler): a single dynamic write offset (in_step), so each
        # attention region has ≤1 dynamic-offset slice. ``ctx`` is pinned per export
        # bucket; the chunked flash SDPA walks it in block_size chunks (a single
        # ctx-wide chunk when ctx <= block_size). (Above ctx ~32768 the flat slot read +
        # mask exceed the accelerator's per-dimension size limit.)
        if ctx is None:
            ctx = max_context_length

        key_cache = torch.zeros(
            n_global_storing, 1, global_channels, 1, ctx, dtype=torch.float16
        )
        value_cache = key_cache.clone()
        sliding_key_cache = torch.zeros(
            n_sliding_storing, 1, sliding_channels, 1, sliding_ring, dtype=torch.float16
        )
        sliding_value_cache = sliding_key_cache.clone()
        sliding_causal_mask = torch.zeros(1, sliding_ring, 1, query_len, dtype=torch.float16)
        # Flat global mask `(1, ctx, 1, q_len)` (same layout as the sliding mask); the
        # flash SDPA walks it in block_size chunks with static bounds.
        causal_mask = torch.zeros(1, ctx, 1, query_len, dtype=torch.float16)
        # RoPE cos/sin are precomputed in the runner (the combined sliding+global
        # table rows), so the graph takes them as float16 inputs instead of
        # ``position_ids``. Width = sliding head_dim + global_head_dim.
        rope_width = head_dim + global_head_dim
        rope_cos = torch.zeros(1, query_len, rope_width, dtype=torch.float16)
        rope_sin = torch.zeros(1, query_len, rope_width, dtype=torch.float16)
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
    }
    if has_sliding:
        # Gemma4 blocked-ladder path: runner-precomputed RoPE cos/sin (no position_ids).
        forward_inputs[ROPE_COS_INPUT_NAME] = rope_cos
        forward_inputs[ROPE_SIN_INPUT_NAME] = rope_sin
    else:
        forward_inputs[POSITION_IDS_INPUT_NAME] = position_ids
    forward_inputs[IN_STEP_INPUT_NAME] = in_step
    if has_sliding:
        # Runner-provided sliding ring write offset (in_step % S); the modulo is
        # computed in the runner and passed in rather than done in the graph. The
        # global cache is FLAT, written at the single absolute offset `in_step` — no
        # separate block index input.
        forward_inputs[SLIDING_IN_STEP_INPUT_NAME] = torch.zeros((1,), dtype=torch.int32)
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
        IN_STEP_INPUT_NAME: None,
        CAUSAL_MASK_INPUT_NAME: {1: cache_len_dim, 3: seq_len_dim},
        KEY_CACHE_INPUT_NAME: {4: cache_len_dim},
        VALUE_CACHE_INPUT_NAME: {4: cache_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }
    if has_sliding:
        forward_dynamic_shapes[ROPE_COS_INPUT_NAME] = {1: seq_len_dim}
        forward_dynamic_shapes[ROPE_SIN_INPUT_NAME] = {1: seq_len_dim}
    else:
        forward_dynamic_shapes[POSITION_IDS_INPUT_NAME] = {1: seq_len_dim}
    if has_sliding:
        # Flat global cache `[n_global_storing, 1, C_g, 1, ctx]` + flat global mask
        # `(1, ctx, 1, q_len)`. ctx is STATIC per export bucket (the flash SDPA slices
        # it with static bounds), so the cache seq dim is NOT a torch.export Dim; only
        # the query/seq dim is dynamic. One dynamic write offset (in_step), no block idx.
        forward_dynamic_shapes[KEY_CACHE_INPUT_NAME] = None
        forward_dynamic_shapes[VALUE_CACHE_INPUT_NAME] = None
        # Flat global mask `(1, ctx, 1, q_len)`: only the query dim (3) is dynamic; ctx
        # is static per export bucket (the flash walks it with static bounds).
        forward_dynamic_shapes[CAUSAL_MASK_INPUT_NAME] = {3: seq_len_dim}
        # Sliding mask: only the query dim is dynamic; the ring depth S is static.
        # Sliding caches are fully static (fixed ring depth), so no dynamic dims.
        forward_dynamic_shapes[SLIDING_IN_STEP_INPUT_NAME] = None
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


def _ios_decomp_table():
    """iOS decomposition table: keep ``silu`` as-is (the accelerator has a fused op)."""
    decomp_table = torch.export.default_decompositions()
    decomp_table.pop(torch.ops.aten.silu.default)
    decomp_table.pop(torch.ops.aten.silu.out)
    return decomp_table


def _export_forward_pair(
    model: torch.nn.Module,
    forward_inputs: dict,
    forward_dynamic_shapes: dict,
    decomp_table,
) -> tuple:
    """Export the (extend decode, prompt prefill) pair for one set of inputs.

    Resets to decode mode before the extend export so this is safe to call in a
    loop (the blocked-cache ladder exports one pair per block-count bucket).
    """
    with torch.no_grad():
        model.set_prefill_mode(False)
        logger.info("Exporting extend module...")
        extend_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(extend_program)

        model.set_prefill_mode(True)
        logger.info("Exporting extend module (prefill mode)...")
        prompt_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(prompt_program)
    return extend_program, prompt_program


def _export_aux_programs(
    model: torch.nn.Module,
    embed_tokens_inputs: tuple,
    embed_tokens_dynamic_shapes: dict,
) -> tuple:
    """Export the (gather_embeddings, load_embeddings) programs (n_blocks-independent)."""
    with torch.no_grad():
        logger.info("Exporting gather_embeddings module...")
        gather_embeddings_exported_program = torch.export.export(
            model.gather_embeddings,
            args=embed_tokens_inputs,
            dynamic_shapes=embed_tokens_dynamic_shapes,
        )
        logger.info("Exporting load_embeddings module...")
        load_embeddings_exported_program = torch.export.export(model.load_embeddings, args=tuple())
    return gather_embeddings_exported_program, load_embeddings_exported_program


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
) -> AIProgram:
    """Convert exported programs to a single AIProgram with iOS constraints.

    This is the single-program path used by every iOS model except the
    blocked-global-cache (Gemma4 sliding-window) models, which route through
    ``_convert_blocked_ladder_to_coreai``.

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
        CAUSAL_MASK_INPUT_NAME,
        EMBEDDING_TABLE_INPUT_NAME,
    ]
    if has_ple:
        input_names.append("ple_embeddings")
    state_names = [
        KEY_CACHE_INPUT_NAME,
        VALUE_CACHE_INPUT_NAME,
    ]
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
    load_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}

    logger.info("Applying optimization passes...")
    coreai_program.set_hardware_constraints(LOAD_EMBEDDINGS_FUNCTION_NAME, load_constraints)
    coreai_program.set_hardware_constraints(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_constraints)
    coreai_program.set_hardware_constraints(EXTEND_FUNCTION_NAME, forward_constraints)
    coreai_program.set_hardware_constraints(PROMPT_OPT_FUNCTION_NAME, forward_constraints)
    coreai_program.optimize()

    return coreai_program


async def _convert_blocked_ladder_to_coreai(
    ladder: list,
    gather_embeddings_program: torch.export.ExportedProgram,
    load_embeddings_program: torch.export.ExportedProgram,
    kv_cached_embed_size: int,
    hidden_size: int,
    has_ple: bool,
    ple_total_dim: int,
    sliding_ring: int,
    n_sliding_storing: int,
    sliding_channels: int,
    n_global_storing: int,
    rope_width: int,
) -> AIProgram:
    """Convert a flat-global-cache context ladder to a single multi-function AIProgram.

    ``ladder`` is a list of ``(ctx, extend_program, prompt_program)``, one per
    context bucket. Each context is its own exported program (the flash block loop
    is unrolled, so ``ceil(ctx / block_size)`` is baked into the graph). Each is
    registered as its own entrypoint ``extend_{ctx}`` / ``prompt_opt_{ctx}`` with
    ``q_len`` as the only static-shape specialization, so the emitted functions are
    ``extend_{ctx}_{q}`` / ``prompt_opt_{ctx}_{q}`` — exactly what the runner picks
    from by context. The shared model weights are referenced (not copied) across
    the programs, and the runner right-sizes the global cache to the running bucket's
    ctx (growing + re-laying-out the written prefix on a bucket switch).
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
        ROPE_COS_INPUT_NAME,
        ROPE_SIN_INPUT_NAME,
        IN_STEP_INPUT_NAME,
        SLIDING_IN_STEP_INPUT_NAME,
        CAUSAL_MASK_INPUT_NAME,
        SLIDING_CAUSAL_MASK_INPUT_NAME,
        EMBEDDING_TABLE_INPUT_NAME,
    ]
    if has_ple:
        input_names.append("ple_embeddings")
    state_names = [
        KEY_CACHE_INPUT_NAME,
        VALUE_CACHE_INPUT_NAME,
        SLIDING_KEY_CACHE_INPUT_NAME,
        SLIDING_VALUE_CACHE_INPUT_NAME,
    ]
    output_names = [OUTPUT_LOGITS_NAME]

    # One entrypoint per context bucket. The base entrypoint name carries the
    # context; the static-shape key carries the query length -> extend_{ctx}_{q}.
    entrypoints: list[tuple[str, int]] = []  # (entrypoint_name, ctx)
    for ctx, extend_program, prompt_program in ladder:
        extend_name = f"{EXTEND_FUNCTION_NAME}_{ctx}"
        prompt_name = f"{PROMPT_OPT_FUNCTION_NAME}_{ctx}"
        converter.add_exported_program(
            extend_program,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
            entrypoint_name=extend_name,
        )
        converter.add_exported_program(
            prompt_program,
            input_names=input_names,
            state_names=state_names,
            output_names=output_names,
            entrypoint_name=prompt_name,
        )
        entrypoints.append((extend_name, ctx))
        entrypoints.append((prompt_name, ctx))

    coreai_program: AIProgram = converter.to_coreai()

    # ----- Static shape configs (query-length specialization within each bucket) -----
    gather_static_cfg = {
        f'"{q_len}"': {TOKEN_IDS_INPUT_NAME: (1, q_len)} for q_len in QUERY_LENGTHS
    }
    coreai_program.set_static_shape_config(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_static_cfg)

    def _forward_static_cfg(ctx: int) -> dict:
        cfg_by_q: dict[str, dict[str, tuple[int, ...]]] = {}
        for q_len in QUERY_LENGTHS:
            cfg = {
                TRANSFORMER_INPUT_NAME: (1, q_len, 1, hidden_size),
                ROPE_COS_INPUT_NAME: (1, q_len, rope_width),
                ROPE_SIN_INPUT_NAME: (1, q_len, rope_width),
                CAUSAL_MASK_INPUT_NAME: (1, ctx, 1, q_len),
                KEY_CACHE_INPUT_NAME: (
                    n_global_storing, 1, kv_cached_embed_size, 1, ctx
                ),
                VALUE_CACHE_INPUT_NAME: (
                    n_global_storing, 1, kv_cached_embed_size, 1, ctx
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
            cfg_by_q[f'"{q_len}"'] = cfg
        return cfg_by_q

    # ----- Hardware constraints (per-bucket ctx; channel interleave + seq-stride
    # alignment match the proven flat-cache constraints). -----
    emb_table_constraints = HardwareConstraints(
        AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
    )
    # Flat global cache is 5D [n_global_storing, 1, C, 1, ctx]: channel C at dim 2,
    # seq = ctx at dim 4 — the proven flat-cache layout (channel interleave + big
    # seq-stride alignment). Alignment uses the bucket's ctx (set per entrypoint).
    sliding_cache_constraints = HardwareConstraints(
        AllocationType.IOSurface,
        interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
        alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * sliding_ring, 1],
    )

    def _forward_constraints(ctx: int) -> dict:
        # Per-bucket seq-stride alignment = THIS bucket's ctx (channel stride = ctx·interleave).
        # The runner right-sizes the global KV cache per session and re-lays-out (grow + copy the
        # written prefix) on a bucket switch, so each bucket's buffer is laid out for its own ctx
        # — smaller allocation + better decode locality than one shared max-ctx buffer. The runner
        # MUST allocate/grow the global cache to match the running bucket's ctx exactly (it does;
        # the sliding ring is fixed-size and never grows).
        cache_constraints = HardwareConstraints(
            AllocationType.IOSurface,
            interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
            alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * ctx, 1],
        )
        return {
            EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints,
            KEY_CACHE_INPUT_NAME: cache_constraints,
            KEY_CACHE_OUTPUT_NAME: cache_constraints,
            VALUE_CACHE_INPUT_NAME: cache_constraints,
            VALUE_CACHE_OUTPUT_NAME: cache_constraints,
            SLIDING_KEY_CACHE_INPUT_NAME: sliding_cache_constraints,
            SLIDING_KEY_CACHE_OUTPUT_NAME: sliding_cache_constraints,
            SLIDING_VALUE_CACHE_INPUT_NAME: sliding_cache_constraints,
            SLIDING_VALUE_CACHE_OUTPUT_NAME: sliding_cache_constraints,
        }
    gather_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}
    load_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}

    logger.info("Applying optimization passes...")
    coreai_program.set_hardware_constraints(LOAD_EMBEDDINGS_FUNCTION_NAME, load_constraints)
    coreai_program.set_hardware_constraints(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_constraints)
    for entrypoint_name, ctx in entrypoints:
        coreai_program.set_static_shape_config(entrypoint_name, _forward_static_cfg(ctx))
        coreai_program.set_hardware_constraints(entrypoint_name, _forward_constraints(ctx))
    coreai_program.optimize()

    return coreai_program


async def _export_ios_blocked_ladder(
    model: torch.nn.Module,
    config,
    max_context_length: int,
    vocab_size: int,
    head_dim: int,
    global_head_dim: int,
    has_ple: bool,
    ple_total_dim: int,
) -> AIProgram:
    """Export the flat-global-cache (Gemma4) model as a per-context ladder.

    For each context in ``context_ladder(max_context_length, block_size)`` we
    export one (extend, prompt) pair (the unrolled flash block loop bakes
    ``ceil(ctx / block_size)`` into the graph). gather/load are exported once.
    Everything is converted into a single AIProgram by
    ``_convert_blocked_ladder_to_coreai``.
    """
    n_kv = config.num_key_value_heads
    block_size = getattr(model.extend.model, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
    buckets = context_ladder(max_context_length, block_size)
    logger.info(
        f"iOS flat-cache context ladder: block_size={block_size} contexts={buckets}"
    )

    decomp_table = _ios_decomp_table()
    ladder: list = []
    gather_program = None
    load_program = None
    for ctx in buckets:
        logger.info(f"Exporting context bucket: ctx={ctx} "
                    f"(flash chunks={(ctx + block_size - 1) // block_size})...")
        inputs = _build_ios_reference_inputs(
            model, config, max_context_length, vocab_size, ctx=ctx
        )
        extend_program, prompt_program = _export_forward_pair(
            model,
            inputs["forward_inputs"],
            inputs["forward_dynamic_shapes"],
            decomp_table,
        )
        ladder.append((ctx, extend_program, prompt_program))
        if gather_program is None:
            gather_program, load_program = _export_aux_programs(
                model,
                inputs["embed_tokens_inputs"],
                inputs["embed_tokens_dynamic_shapes"],
            )

    return await _convert_blocked_ladder_to_coreai(
        ladder=ladder,
        gather_embeddings_program=gather_program,
        load_embeddings_program=load_program,
        kv_cached_embed_size=n_kv * global_head_dim,
        hidden_size=config.hidden_size,
        has_ple=has_ple,
        ple_total_dim=ple_total_dim,
        sliding_ring=sliding_ring_size(config.sliding_window, max(QUERY_LENGTHS)),
        n_sliding_storing=model.extend.model.n_sliding_storing,
        sliding_channels=n_kv * head_dim,
        n_global_storing=model.extend.model.n_global_storing,
        rope_width=head_dim + global_head_dim,
    )


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

    # Feature detection (drives the single-program vs blocked-ladder split).
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
        # Flat-global-cache models (Gemma4): emit a per-context ladder — one
        # statically-shaped program per context size (powers of two from 256) — so
        # short/mid prompts run a small ctx-wide graph instead of always the full
        # max context.
        return await _export_ios_blocked_ladder(
            model=model,
            config=config,
            max_context_length=max_context_length,
            vocab_size=vocab_size,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            has_ple=has_ple,
            ple_total_dim=ple_total_dim,
        )

    # ----- Single-program path (all other iOS models): context-length ladder via
    # static-shape specialization of one dynamic program. -----
    inputs = _build_ios_reference_inputs(model, config, max_context_length, vocab_size)
    (
        extend_program,
        prompt_program,
        gather_program,
        load_program,
    ) = _export_ios_programs(model, inputs)

    if global_head_dim is not None:
        kv_cached_embed_size = config.num_key_value_heads * (head_dim + global_head_dim)
    else:
        kv_cached_embed_size = config.num_key_value_heads * head_dim
    num_layers = config.num_hidden_layers

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
    )

    return coreai_program

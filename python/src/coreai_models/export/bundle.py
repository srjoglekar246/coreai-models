# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Create model bundles from exported .aimodel files."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

METADATA_VERSION = "0.2"


def bundle_llm_asset(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    per_layer_embeddings: str | None = None,
) -> None:
    """Add tokenizer and metadata.json (0.2 schema) to an LLM bundle.

    Expects ``{name}.aimodel`` to already exist inside bundle_path.

    Args:
        per_layer_embeddings: Optional filename (relative to the bundle) of the
            externalized INT8 Per-Layer Embeddings artifact (Gemma4). When set,
            the runner mmaps it and gathers per-token rows for the
            ``ple_embeddings`` graph input.
    """
    _write_tokenizer(bundle_path / "tokenizer", hf_model_id)
    _write_metadata(
        bundle_path, hf_model_id, hf_config, compression, name, per_layer_embeddings
    )


def _write_tokenizer(dest: Path, hf_model_id: str) -> None:
    logger.info(f"Saving tokenizer from {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _resolve_eos_token_ids(hf_model_id: str, text_config: Any) -> list[int]:
    """Collect end-of-generation token ids from the generation config.

    Falls back to the model config's ``eos_token_id`` when no generation config
    is available. Always returns a de-duplicated list of ints.
    """
    ids: list[int] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                _add(v)
        elif isinstance(value, int) and value not in ids:
            ids.append(value)

    try:
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(hf_model_id)
        _add(gen_config.eos_token_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not load generation config for eos tokens: {exc}")

    _add(getattr(text_config, "eos_token_id", None))
    return ids


def _write_metadata(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    per_layer_embeddings: str | None = None,
) -> None:
    # Some models (e.g. Gemma4 multimodal) nest the text decoder config; the
    # vocab/context live on ``text_config`` in that case.
    text_config = getattr(hf_config, "text_config", hf_config)

    language: dict[str, Any] = {
        "tokenizer": hf_model_id,
        "vocab_size": getattr(text_config, "vocab_size", None),
        "max_context_length": getattr(text_config, "max_position_embeddings", None),
        "embedded_tokenizer": True,
        "function_map": {"main": ["main"]},
    }
    if per_layer_embeddings is not None:
        language["per_layer_embeddings"] = per_layer_embeddings

    # Sliding-window size, needed by the runner to build the windowed mask for
    # models with a sliding KV cache (Gemma4). Harmless for others — the runner
    # only consults it when the graph declares a ``sliding_causal_mask`` input.
    sliding_window = getattr(text_config, "sliding_window", None)
    if isinstance(sliding_window, int) and sliding_window > 0:
        language["sliding_window"] = sliding_window

    # Dual-RoPE parameters for models that precompute cos/sin in the runner
    # (Gemma4 large-context): the graph takes ``rope_cos``/``rope_sin`` instead of
    # ``position_ids`` (a 131k position overflows a 16-bit input and a 32-bit input
    # crashes the MPSGraph streaming compiler). The runner needs the two head dims,
    # the two RoPE bases, and the global partial-rotary factor to build the combined
    # table rows. Emitted only when the model declares the Gemma4 dual-RoPE config.
    rope_parameters = getattr(text_config, "rope_parameters", None)
    global_head_dim = getattr(text_config, "global_head_dim", None)
    if (
        isinstance(rope_parameters, dict)
        and "sliding_attention" in rope_parameters
        and "full_attention" in rope_parameters
        and isinstance(global_head_dim, int)
    ):
        full = rope_parameters["full_attention"]
        language["rope"] = {
            "sliding_head_dim": text_config.head_dim,
            "global_head_dim": global_head_dim,
            "sliding_rope_theta": rope_parameters["sliding_attention"]["rope_theta"],
            "global_rope_theta": full["rope_theta"],
            "partial_rotary_factor": full.get("partial_rotary_factor", 0.25),
        }

    # End-of-generation token ids. The tokenizer exposes only a single
    # ``eos_token`` (e.g. ``<eos>``), but Gemma chat models stop on additional
    # tokens (``<end_of_turn>``) that the tokenizer's eos doesn't cover. Carry
    # the full ``generation_config`` eos list so the runner halts cleanly.
    #
    # Scoped to the Gemma family on purpose: other models (Qwen/Llama/Mistral)
    # already stop correctly on the tokenizer eos, so emitting extra ids here
    # would change their runtime stopping behavior. The runner unions these with
    # the tokenizer eos, so adding them is only safe where it's actually needed.
    model_type = getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", "")
    if "gemma" in model_type:
        eos_token_ids = _resolve_eos_token_ids(hf_model_id, text_config)
        if eos_token_ids:
            language["eos_token_ids"] = eos_token_ids

    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": language,
        "source": {
            "model_definition": "torch",
            "hf_model_id": hf_model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }
    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")

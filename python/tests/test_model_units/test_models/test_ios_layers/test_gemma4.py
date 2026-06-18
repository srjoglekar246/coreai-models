# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS Gemma4 two-cache sliding-window attention parity with HuggingFace.

The iOS Gemma4 decoder uses two compacted KV caches — a full-context global
cache and a small sliding-window ring (depth S) — and applies windowed attention
via a runner-built ``sliding_causal_mask``. These tests mimic the Swift runner's
chunked prefill against persistent caches and compare per-position logits to a
single HF forward (which applies sliding-window attention internally). The key
case is a prompt longer than both W and S so the ring wraps — exactly where the
previous single-plain-causal-mask implementation silently diverged.
"""

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Gemma4TextConfig  # noqa: E402

try:
    from transformers.models.gemma4 import Gemma4ForCausalLM
except Exception:  # pragma: no cover - requires transformers>=5.5
    Gemma4ForCausalLM = None

from coreai_models.models.ios.gemma4_text import (  # noqa: E402
    Gemma4ForCausalLMForiOS,
    _compute_kv_layout,
)

DTYPE = torch.float32
NEG = float("-inf")

pytestmark = pytest.mark.skipif(
    Gemma4ForCausalLM is None, reason="gemma4 requires transformers>=5.5"
)


def _ring_size(window: int, max_q: int) -> int:
    raw = window + max_q - 1
    return ((raw + max_q - 1) // max_q) * max_q


def _make_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=16,
        num_kv_shared_layers=2,
        sliding_window=8,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rms_norm_eps=1e-6,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
    )


def _build_ios_model(cfg, hf_sd):
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(cfg, model_device="cpu", disable_embedding_quantization=True)
    model.to(DTYPE).eval()
    model._mutate_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)
    if hasattr(model, "_ple_scale_pending"):
        model.extend.ple_scale = model._ple_scale_pending
        model.extend.ple_zp = model._ple_zp_pending
    return model


def _ple_input_fp(hf_model, cfg, token_ids):
    # Build the fp ``ple_embeddings`` graph input the way export quantizes it
    # (ple_weight[token] * sqrt(ple_dim)), but kept fp to avoid INT8 noise.
    w = hf_model.state_dict()["model.embed_tokens_per_layer.weight"]
    ple_dim = cfg.hidden_size_per_layer_input
    total = cfg.num_hidden_layers * ple_dim
    rows = w[:, :total][token_ids].to(DTYPE) * (float(ple_dim) ** 0.5)
    return rows.reshape(1, len(token_ids), 1, total)


def _global_mask(ctx, q_len, aligned_step):
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        m[0, : aligned_step + i + 1, 0, i] = 0.0
    return m


def _sliding_mask(S, q_len, aligned_step, window):
    m = torch.full((1, S, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = aligned_step + i
        for pos in range(max(0, p - window + 1), p + 1):
            m[0, pos % S, 0, i] = 0.0
    return m


def _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx):
    n_kv = cfg.num_key_value_heads
    sliding_storing, global_storing, _ = _compute_kv_layout(cfg)
    seq = len(token_ids)

    key_cache = torch.zeros(
        len(global_storing), 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE
    )
    value_cache = key_cache.clone()
    skey_cache = torch.zeros(len(sliding_storing), 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
    svalue_cache = skey_cache.clone()

    out_logits = torch.zeros(seq, cfg.vocab_size, dtype=DTYPE)
    for start in range(0, seq, q_len):
        chunk = token_ids[start : start + q_len]
        ids = chunk.reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        with torch.no_grad():
            out = ios(
                ids,
                pos,
                in_step,
                sliding_in_step,
                _global_mask(ctx, q_len, start),
                _sliding_mask(S, q_len, start, cfg.sliding_window),
                key_cache,
                value_cache,
                skey_cache,
                svalue_cache,
                _ple_input_fp(hf, cfg, chunk),
            )
        out_logits[start : start + q_len] = out.reshape(q_len, cfg.vocab_size)
    return out_logits


def test_kv_layout_dead_slot_compaction():
    """Layout keeps only storing layers; shared layers reuse the source slot."""
    cfg = _make_config()
    sliding_storing, global_storing, layout = _compute_kv_layout(cfg)
    assert sliding_storing == [0, 1, 3]
    assert global_storing == [2]
    # Shared sliding layer 4 -> source 3 -> sliding slot 2; shared global 5 -> global slot 0.
    assert layout[4] == (True, True, 2)
    assert layout[5] == (False, True, 0)


def test_sliding_parity_with_ring_wrap():
    """Chunked prefill over a prompt longer than both W and S matches HF."""
    torch.manual_seed(0)
    cfg = _make_config()
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = _ring_size(cfg.sliding_window, q_len)  # 12
    ctx = 32
    seq = 24  # > S (ring wraps) and > W (windowing active)
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)

    assert (ios_logits.argmax(-1) == hf_logits.argmax(-1)).all()
    torch.testing.assert_close(ios_logits, hf_logits, atol=1e-3, rtol=1e-3)


def test_short_prompt_regression():
    """A prompt shorter than the window still matches HF (no windowing applied)."""
    torch.manual_seed(1)
    cfg = _make_config()
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    ios = _build_ios_model(cfg, dict(hf.state_dict()))

    q_len = 4
    S = _ring_size(cfg.sliding_window, q_len)
    ctx = 32
    seq = 4  # < W
    token_ids = torch.randint(0, cfg.vocab_size, (seq,))

    with torch.no_grad():
        hf_logits = hf(
            input_ids=token_ids.reshape(1, seq),
            position_ids=torch.arange(seq).reshape(1, seq),
        ).logits[0]

    ios_logits = _chunked_prefill_logits(ios, hf, cfg, token_ids, q_len, S, ctx)
    torch.testing.assert_close(ios_logits, hf_logits, atol=1e-3, rtol=1e-3)

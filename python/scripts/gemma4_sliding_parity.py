# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch-level parity for the iOS Gemma4 two-cache sliding-window design.

Mimics the Swift runner's chunked prefill: feeds a prompt longer than both the
sliding window W and the ring depth S (so the sliding ring wraps), one q_len
chunk at a time, against persistent global + sliding KV caches, and compares the
per-position next-token logits to a single HuggingFace forward (which applies
sliding-window attention internally). Run with the transformers>=5.5 overlay:

    uv run --with "transformers>=5.5.0" python python/scripts/gemma4_sliding_parity.py
"""

import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import Gemma4ForCausalLM

from coreai_models.models.ios.gemma4_text import (
    Gemma4CombinedRoPE,
    Gemma4ForCausalLMForiOS,
    _compute_kv_layout,
)

torch.manual_seed(0)
DTYPE = torch.float32
NEG = float("-inf")


def ring_size(window: int, max_q: int) -> int:
    raw = window + max_q - 1
    return ((raw + max_q - 1) // max_q) * max_q


def make_config() -> Gemma4TextConfig:
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


def build_ios_model(cfg, hf_sd):
    sd = dict(hf_sd)
    model = Gemma4ForCausalLMForiOS(
        cfg, model_device="cpu", disable_embedding_quantization=True
    )
    model.to(DTYPE).eval()
    model._mutate_state_dict(sd)
    model.load_state_dict(sd, assign=True, strict=True)
    if hasattr(model, "_ple_scale_pending"):
        model.extend.ple_scale = model._ple_scale_pending
        model.extend.ple_zp = model._ple_zp_pending
    return model


def ple_input_fp(hf_model, cfg, token_ids):
    """Build the fp ``ple_embeddings`` graph input the way export quantizes it,
    but kept in fp to avoid INT8 noise: ple_weight[token] * sqrt(ple_dim)."""
    w = hf_model.state_dict()["model.embed_tokens_per_layer.weight"]  # (vocab, L*ple)
    ple_dim = cfg.hidden_size_per_layer_input
    total = cfg.num_hidden_layers * ple_dim
    w = w[:, :total]
    scale = float(ple_dim) ** 0.5
    rows = w[token_ids].to(DTYPE) * scale  # (seq, total)
    return rows.reshape(1, len(token_ids), 1, total)


def global_mask(ctx, q_len, aligned_step):
    """Flat global causal mask ``(1, ctx, 1, q_len)``. Query col ``i`` (abs pos
    ``aligned_step+i``) attends key ``j`` iff ``j <= pos`` (full causal)."""
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = aligned_step + i
        m[0, : p + 1, 0, i] = 0.0
    return m


def sliding_mask(S, q_len, aligned_step, window):
    m = torch.full((1, S, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = aligned_step + i
        for pos in range(max(0, p - window + 1), p + 1):
            m[0, pos % S, 0, i] = 0.0
    return m


def run(max_context: int = 32, block_size: int = 8):
    cfg = make_config()
    cfg.kv_block_size = block_size
    W = cfg.sliding_window
    hf = Gemma4ForCausalLM(cfg).to(DTYPE).eval()
    hf_sd = dict(hf.state_dict())

    sliding_storing, global_storing, layout = _compute_kv_layout(cfg)
    print(f"layout: sliding_storing={sliding_storing} global_storing={global_storing}")
    print(f"        per-layer (is_sliding,is_shared,slot)={layout}")

    ios = build_ios_model(cfg, hf_sd)

    # Combined RoPE table (sliding + global), the runner's reference: the graph no
    # longer gathers cos/sin internally — it takes precomputed ``rope_cos`` /
    # ``rope_sin`` rows (1, q_len, sliding_hd + global_hd). We reproduce that here
    # with the same ``Gemma4CombinedRoPE`` the runner reimplements.
    sliding_params = cfg.rope_parameters["sliding_attention"]
    global_params = cfg.rope_parameters["full_attention"]
    rope = Gemma4CombinedRoPE(
        sliding_head_dim=cfg.head_dim,
        global_head_dim=cfg.global_head_dim,
        max_cache_size=cfg.max_position_embeddings,
        sliding_base=sliding_params["rope_theta"],
        global_base=global_params["rope_theta"],
        partial_rotary_factor=global_params.get("partial_rotary_factor", 0.25),
    ).to(DTYPE)

    n_kv = cfg.num_key_value_heads
    q_len = 4
    S = ring_size(W, q_len)            # 12
    assert max_context % block_size == 0, "global cache ctx must be a multiple of block_size"
    n_blocks = max_context // block_size
    ctx = max_context
    # seq spans >2 blocks (cross-block writes) AND > S (ring wraps) AND > W.
    seq = min(ctx, max(24, 2 * block_size + q_len))
    seq = (seq // q_len) * q_len
    assert seq % q_len == 0 and block_size % q_len == 0
    print(
        f"W={W} q_len={q_len} S={S} block_size={block_size} n_blocks={n_blocks} "
        f"ctx={ctx} seq={seq} (ring wraps: {seq > S}, blocks written: {(seq - 1) // block_size + 1})"
    )

    token_ids = torch.randint(0, cfg.vocab_size, (seq,))
    input_ids = token_ids.reshape(1, seq)

    # HF reference: one forward over the full prompt.
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, position_ids=torch.arange(seq).reshape(1, seq))
    hf_logits = hf_out.logits[0]  # (seq, vocab)

    # iOS chunked prefill against persistent caches. Global cache is FLAT
    # [n_global_storing, 1, C, 1, ctx]; the chunked flash BlockedSDPA walks the flat
    # slot in block_size chunks. Sliding cache is the flat ring.
    key_cache = torch.zeros(
        len(global_storing), 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE
    )
    value_cache = key_cache.clone()
    skey_cache = torch.zeros(len(sliding_storing), 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
    svalue_cache = skey_cache.clone()

    ios_logits = torch.zeros(seq, cfg.vocab_size, dtype=DTYPE)
    for start in range(0, seq, q_len):
        chunk = token_ids[start : start + q_len]
        ids = chunk.reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        # Precompute the combined cos/sin rows for this chunk's positions (what the
        # runner will pass), replacing the in-graph gather.
        rope_cos, rope_sin = rope.gather_cos_sin(pos)
        # Flat global cache: in_step is the single absolute write offset (no block
        # index). sliding_in_step is the ring offset (start % S).
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        cmask = global_mask(ctx, q_len, start)
        smask = sliding_mask(S, q_len, start, W)
        ple = ple_input_fp(hf, cfg, chunk)
        with torch.no_grad():
            out = ios(
                ids, rope_cos, rope_sin, in_step, sliding_in_step,
                cmask, smask,
                key_cache, value_cache, skey_cache, svalue_cache,
                ple,
            )
        # out shape (1, 1, q_len, vocab) -> (q_len, vocab)
        chunk_logits = out.reshape(q_len, cfg.vocab_size)
        ios_logits[start : start + q_len] = chunk_logits

    # Compare.
    max_abs = (ios_logits - hf_logits).abs().max().item()
    argmax_agree = (ios_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    # cosine / PSNR-ish
    num = (ios_logits * hf_logits).sum()
    den = ios_logits.norm() * hf_logits.norm()
    cos = (num / den).item()
    print(f"\nmax_abs_diff={max_abs:.3e}  argmax_agreement={argmax_agree:.3f}  cosine={cos:.6f}")

    # Per-position windowed-region check: positions > W must still match (this is
    # exactly where the OLD single-mask implementation diverged).
    beyond = slice(W, seq)
    beyond_agree = (ios_logits[beyond].argmax(-1) == hf_logits[beyond].argmax(-1)).float().mean().item()
    beyond_max = (ios_logits[beyond] - hf_logits[beyond]).abs().max().item()
    print(f"positions>{W}: argmax_agreement={beyond_agree:.3f}  max_abs_diff={beyond_max:.3e}")

    ok = max_abs < 1e-2 and argmax_agree == 1.0
    print("\nPARITY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-context",
        type=int,
        default=32,
        help="Global blocked-cache total context (multiple of block-size).",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Blocked-cache block size B. Small here so a short seq exercises "
        "multiple blocks + cross-block writes (literal 8192 would need a >16k-token "
        "seq, whose full-attention HF reference OOMs).",
    )
    args = ap.parse_args()
    sys.exit(0 if run(args.max_context, args.block_size) else 1)

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity for the blocked / flash global-attention path.

Default: a pure-torch flash / online-softmax attention, computed block-by-block
over the key axis, must match the flat per-head ``SDPA`` (≤1e-5) — proving the
online-softmax recurrence is algebraically identical to one big softmax. We
exercise ``n_blocks`` ∈ {1, 2, 16} and a ragged final block.

``--use-primitive``: the same parity, but the flash side calls the real
``primitives/ios/sdpa.py`` blocked variant, and we assert every compute op in its
traced graph has rank ≤ 4 (the on-device 4D compute-tensor limit).

    uv run python python/scripts/flash_attention_parity.py
    uv run python python/scripts/flash_attention_parity.py --use-primitive
"""

import argparse

import torch

from coreai_models.primitives.ios.sdpa import SDPA

torch.manual_seed(0)
DTYPE = torch.float32
NEG = float("-inf")


def build_causal_mask(ctx: int, q_len: int, start: int) -> torch.Tensor:
    """Flat causal mask in SDPA layout ``(1, ctx, 1, q_len)``.

    Query column ``i`` is at absolute position ``start + i`` and attends every
    key ``0 .. start+i`` (full causal). Matches ``gemma4_sliding_parity.global_mask``.
    """
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=DTYPE)
    for i in range(q_len):
        p = start + i
        m[0, : p + 1, 0, i] = 0.0
    return m


def block_bounds(ctx: int, block_size: int):
    """Ascending ``[start, end)`` block boundaries over the key axis."""
    bounds = []
    s = 0
    while s < ctx:
        bounds.append((s, min(s + block_size, ctx)))
        s += block_size
    return bounds


def flash_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal_mask: torch.Tensor,
    head_dim: int,
    block_size: int,
    scale: float,
    two_pass: bool = False,
) -> torch.Tensor:
    """Flash / online-softmax attention in the SDPA BC1S layout.

    Walks the key axis in ``block_size`` blocks, keeping the textbook running
    (max, denom, acc) recurrence — or, with ``two_pass``, a first pass for the
    global per-row (max, denom) and a second pass for the weighted value sum.
    Per-head matmuls stay rank ≤ 3 (``(B,q,d)@(B,d,blk)`` and ``(B,q,blk)@(B,blk,d)``).

    Shapes (mirror ``SDPA.forward``):
        query       (B, n_heads*head_dim, 1, q_len)
        key/value   (B, n_kv*head_dim,    1, ctx)
        causal_mask (1, ctx, 1, q_len)
    returns         (B, n_heads*head_dim, 1, q_len)
    """
    B = query.shape[0]
    q_len = query.shape[-1]
    ctx = key.shape[-1]

    queries = query.split(head_dim, dim=1)          # each (B, head_dim, 1, q_len)
    keys = list(key.split(head_dim, dim=1))          # each (B, head_dim, 1, ctx)
    values = list(value.split(head_dim, dim=1))
    n_heads = len(queries)
    n_kv = len(keys)
    kv_group = n_heads // n_kv

    bounds = block_bounds(ctx, block_size)
    outs = []

    for head_idx in range(n_heads):
        kv_idx = head_idx // kv_group
        # (B, q_len, head_dim)
        q_mat = queries[head_idx].squeeze(2).transpose(1, 2)
        # (B, head_dim, ctx) and (B, ctx, head_dim)
        k_all = keys[kv_idx].squeeze(2)
        v_all = values[kv_idx].squeeze(2).transpose(1, 2)
        # mask -> (1, q_len, ctx)
        mask_all = causal_mask.squeeze(2).transpose(1, 2)

        if not two_pass:
            m = torch.full((B, q_len, 1), NEG, dtype=DTYPE)
            l = torch.zeros((B, q_len, 1), dtype=DTYPE)
            acc = torch.zeros((B, q_len, head_dim), dtype=DTYPE)
            for s, e in bounds:
                s_t = (q_mat @ k_all[:, :, s:e]) * scale       # (B, q_len, blk)
                s_t = s_t + mask_all[:, :, s:e]
                rowmax = s_t.max(dim=-1, keepdim=True).values  # (B, q_len, 1)
                m_new = torch.maximum(m, rowmax)
                safe = ~torch.isneginf(m_new)
                corr = torch.where(
                    torch.isneginf(m), torch.zeros_like(m), torch.exp(m - m_new)
                )
                p_t = torch.where(
                    safe.expand_as(s_t), torch.exp(s_t - m_new), torch.zeros_like(s_t)
                )
                l = l * corr + p_t.sum(dim=-1, keepdim=True)
                acc = acc * corr + p_t @ v_all[:, s:e, :]      # (B, q_len, head_dim)
                m = m_new
            out_h = acc / l
        else:
            # Pass 1: global running max + denom over all blocks.
            m = torch.full((B, q_len, 1), NEG, dtype=DTYPE)
            l = torch.zeros((B, q_len, 1), dtype=DTYPE)
            scores = []
            for s, e in bounds:
                s_t = (q_mat @ k_all[:, :, s:e]) * scale
                s_t = s_t + mask_all[:, :, s:e]
                scores.append(s_t)
                m = torch.maximum(m, s_t.max(dim=-1, keepdim=True).values)
            for s_t in scores:
                safe = ~torch.isneginf(m)
                p_t = torch.where(
                    safe.expand_as(s_t), torch.exp(s_t - m), torch.zeros_like(s_t)
                )
                l = l + p_t.sum(dim=-1, keepdim=True)
            # Pass 2: weighted value sum.
            acc = torch.zeros((B, q_len, head_dim), dtype=DTYPE)
            for (s, e), s_t in zip(bounds, scores):
                safe = ~torch.isneginf(m)
                p_t = torch.where(
                    safe.expand_as(s_t), torch.exp(s_t - m), torch.zeros_like(s_t)
                )
                acc = acc + p_t @ v_all[:, s:e, :]
            out_h = acc / l

        # (B, q_len, head_dim) -> (B, head_dim, 1, q_len)
        outs.append(out_h.transpose(1, 2).unsqueeze(2))

    return torch.cat(outs, dim=1)


def make_inputs(n_heads, n_kv, head_dim, ctx, q_len, start):
    B = 1
    query = torch.randn(B, n_heads * head_dim, 1, q_len, dtype=DTYPE)
    key = torch.randn(B, n_kv * head_dim, 1, ctx, dtype=DTYPE)
    value = torch.randn(B, n_kv * head_dim, 1, ctx, dtype=DTYPE)
    mask = build_causal_mask(ctx, q_len, start)
    return query, key, value, mask


def to_blocked(key, value, mask, block_size):
    """Flat SDPA tensors -> the BlockedSDPA blocked-cache-slot layout.

    key/value (B,C,1,ctx) -> (n_blocks,C,1,block_size);
    mask (1,ctx,1,q_len)  -> (1,n_blocks,block_size,1,q_len).
    ``ctx`` must be a multiple of ``block_size`` (the static-graph invariant;
    ragged tails are covered by the Phase-0 pure-torch path).
    """
    Bb, C, _, ctx = key.shape
    assert ctx % block_size == 0
    n_blocks = ctx // block_size
    q_len = mask.shape[-1]
    # (1,C,1,ctx) -> (n_blocks, C, 1, block_size)
    kb = key.view(C, n_blocks, block_size).permute(1, 0, 2).reshape(n_blocks, C, 1, block_size)
    vb = value.view(C, n_blocks, block_size).permute(1, 0, 2).reshape(n_blocks, C, 1, block_size)
    mb = mask.view(1, n_blocks, block_size, 1, q_len).contiguous()
    return kb, vb, mb


def assert_ranks_le_4(module, args) -> int:
    """Export ``module`` and assert every *compute* tensor has rank ≤ 4 (the ANE
    compute-tensor constraint). Input placeholders (the blocked K/V *state* and
    blocked mask) are intentionally higher-rank tensor_buffers — the blocked-cache
    rank note (gemma4_long_prompt_design.md §3) allows that for state, only compute
    ops must be ≤4D. Returns the max compute rank seen."""
    from torch.export import export

    gm = export(module, args).graph_module
    max_rank = 0
    offenders = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        val = node.meta.get("val")
        vals = val if isinstance(val, (list, tuple)) else [val]
        for v in vals:
            shape = getattr(v, "shape", None)
            if shape is None:
                continue
            r = len(shape)
            max_rank = max(max_rank, r)
            if r > 4:
                offenders.append((node.name, str(node.target), tuple(shape)))
    if offenders:
        print("  RANK>4 compute ops:", offenders[:8])
    return max_rank


def run(use_primitive: bool) -> bool:
    BlockedSDPA = None
    if use_primitive:
        try:
            from coreai_models.primitives.ios.sdpa import BlockedSDPA
        except ImportError:
            print(
                "--use-primitive: BlockedSDPA not implemented in "
                "primitives/ios/sdpa.py yet (Phase 1). Run without the flag for "
                "the Phase 0 pure-torch parity."
            )
            return False

    n_heads, n_kv, head_dim = 4, 2, 16
    scale = head_dim**-0.5
    flat = SDPA(head_dim=head_dim)

    if use_primitive:
        # Phase 1: real blocked primitive (vectorized two-pass). Static graph =>
        # ctx is a multiple of block_size; n_blocks ∈ {1,2,16}. Ragged tails are a
        # Phase-0 concern.
        q_len = 8
        worst = 0.0
        all_ok = True
        ranks_ok = True
        blocked = BlockedSDPA(head_dim=head_dim)
        for n_blocks in (1, 2, 16):
            block_size = 16
            ctx = block_size * n_blocks
            for start, sub in ((max(0, ctx - 8), "end"), (0, "start")):
                query, key, value, mask = make_inputs(
                    n_heads, n_kv, head_dim, ctx, q_len, start
                )
                kb, vb, mb = to_blocked(key, value, mask, block_size)
                with torch.no_grad():
                    ref = flat(query, key, value, mask)
                    got = blocked(query, kb, vb, mb)
                diff = (got - ref).abs().max().item()
                worst = max(worst, diff)
                ok = diff <= 1e-5
                all_ok = all_ok and ok
                tag = "ok " if ok else "FAIL"
                print(
                    f"[{tag}] vectorized n_blocks={n_blocks:<2} {sub:<5} "
                    f"max_abs_diff={diff:.3e}"
                )
        # Rank check on the largest graph (n_blocks=16).
        ctx = 16 * 16
        query, key, value, mask = make_inputs(n_heads, n_kv, head_dim, ctx, q_len, ctx - 8)
        kb, vb, mb = to_blocked(key, value, mask, 16)
        max_rank = assert_ranks_le_4(blocked, (query, kb, vb, mb))
        ranks_ok = max_rank <= 4
        print(f"  [{'ok ' if ranks_ok else 'FAIL'}] max op rank = {max_rank} (≤4 required)")

        print(f"\nworst max_abs_diff={worst:.3e}")
        ok = all_ok and ranks_ok
        print("PARITY:", "PASS" if ok else "FAIL")
        return ok

    # (block_size, ctx, start) — n_blocks = ceil(ctx/block_size).
    #   ctx==block_size              -> 1 block
    #   ctx==2*block_size            -> 2 blocks
    #   ctx==16*block_size           -> 16 blocks
    #   ctx not a multiple of block  -> ragged final block
    # start chosen to exercise both a mid-prompt chunk (future blocks all -inf,
    # the NaN-guard path) and an end chunk (diagonal in the last block).
    cases = []
    for block_size, n_blocks in [(16, 1), (16, 2), (16, 16)]:
        ctx = block_size * n_blocks
        cases.append((block_size, ctx, max(0, ctx - 8), f"n_blocks={n_blocks} end"))
        cases.append((block_size, ctx, 0, f"n_blocks={n_blocks} start"))
    # ragged final blocks
    cases.append((16, 40, 24, "ragged ctx=40 (3 blks, last=8)"))
    cases.append((16, 100, 0, "ragged ctx=100 (7 blks, last=4)"))
    cases.append((8, 53, 30, "ragged ctx=53 (7 blks, last=5)"))

    q_len = 8
    worst = 0.0
    all_ok = True
    for block_size, ctx, start, label in cases:
        query, key, value, mask = make_inputs(n_heads, n_kv, head_dim, ctx, q_len, start)
        with torch.no_grad():
            ref = flat(query, key, value, mask)
            for two_pass in (False, True):
                got = flash_sdpa(
                    query, key, value, mask, head_dim, block_size, scale, two_pass
                )
                diff = (got - ref).abs().max().item()
                worst = max(worst, diff)
                ok = diff <= 1e-5
                all_ok = all_ok and ok
                form = "two-pass" if two_pass else "online  "
                tag = "ok " if ok else "FAIL"
                print(f"[{tag}] {form} {label:<32} max_abs_diff={diff:.3e}")

    print(f"\nworst max_abs_diff={worst:.3e}")
    print("PARITY:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--use-primitive", action="store_true")
    args = ap.parse_args()
    sys.exit(0 if run(args.use_primitive) else 1)

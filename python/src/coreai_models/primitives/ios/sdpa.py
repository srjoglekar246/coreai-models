# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn


class SDPA(nn.Module):
    """iOS-optimized Scaled Dot-Product Attention.

    Unlike PyTorch's fused SDPA, iOS requires each attention head to be computed
    individually to meet hardware constraints and ensure efficient compilation.
    This implementation processes heads sequentially rather than in parallel.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        scale: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        with torch.device("cpu"):
            if scale is None:
                self._scale_factor = nn.Buffer(torch.tensor(head_dim**-0.5), persistent=False)
            else:
                self._scale_factor = (
                    nn.Buffer(scale, persistent=False)
                    if isinstance(scale, torch.Tensor)
                    else nn.Buffer(torch.tensor(scale), persistent=False)
                )

    # Efficient implementation equivalent to the following:
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention for iOS.

        Args:
            query: Query tensor with shape (batch_size, n_heads*head_dim, 1, seq_len)
            key: Key tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            value: Value tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            causal_mask: Causal attention mask with shape (1, max_seq_len, 1, seq_len)

        Returns:
            torch.Tensor: Attention output with shape (batch_size, n_heads*head_dim, 1, seq_len)
        """

        # Apply the scale factor before QK^T for numerical stability
        key = key.transpose(-3, -1) * self._scale_factor
        queries = query.split(self.head_dim, dim=1)
        keys = list(key.split(self.head_dim, dim=-1))

        n_heads = len(queries)

        # permute key heads in advance
        for kv_idx in range(len(keys)):
            keys[kv_idx] = keys[kv_idx].permute(0, 2, 3, 1)

        kv_group_size = len(queries) // len(keys)

        scores = []

        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            q = queries[head_idx].permute(0, 2, 3, 1)
            k = keys[kv_idx]
            attn_score = q @ k
            attn_score = attn_score.permute(0, 3, 1, 2)
            scores.append(attn_score)

        full_scores = torch.cat(scores, dim=2)
        masked_scores = full_scores + torch.cat([causal_mask] * n_heads, dim=2)
        full_scores = masked_scores.softmax(1)

        scores = full_scores.split(1, dim=2)

        values = list(value.split(self.head_dim, dim=1))

        # transpose values in advance
        for kv_idx in range(len(values)):
            values[kv_idx] = values[kv_idx].permute(0, 2, 3, 1).squeeze(1)

        weights = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            s = scores[head_idx].permute(0, 2, 3, 1).squeeze(1)
            v = values[kv_idx]
            weight = (s @ v).unsqueeze(1)
            weight = weight.permute(0, 3, 1, 2)
            weights.append(weight)

        final_score = torch.cat(weights, dim=1)
        return final_score


class BlockedSDPA(nn.Module):
    """Blocked / flash global attention for large-context iOS.

    The flat ``SDPA`` materializes a score tensor whose key axis spans the full
    context; at large context that axis grows past the Neural Engine's supported
    tensor sizes and the global layers fall back to the GPU. ``BlockedSDPA`` instead
    reads the global K/V cache as ``n_blocks`` blocks of ``block_size`` (≤ 8192) and
    walks them with the textbook **online-softmax (flash)** recurrence —
    algebraically identical to one big softmax (parity ~1e-6) but with every score
    op's key axis kept at ``block_size``, so it stays Neural-Engine-resident.

    The block loop is **unrolled** and ``n_blocks`` is fixed per export (a static
    block count is required for the model to stay on the Neural Engine). Each block
    reuses the per-head shapes of the flat ``SDPA`` (rank-4 ``q@k`` with the leading
    size-1 head dim, rank-3 ``scores@v``); the only extra ops are the online-softmax
    glue (``maximum``/``exp``/mul/add/inner-axis reductions).

    Sliding layers (fixed ``S=576`` ring) are unaffected and keep using ``SDPA``.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        scale: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        with torch.device("cpu"):
            if scale is None:
                self._scale_factor = nn.Buffer(torch.tensor(head_dim**-0.5), persistent=False)
            else:
                self._scale_factor = (
                    nn.Buffer(scale, persistent=False)
                    if isinstance(scale, torch.Tensor)
                    else nn.Buffer(torch.tensor(scale), persistent=False)
                )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blocked scaled dot-product attention (batch size 1).

        Args:
            query: ``(1, n_heads*head_dim, 1, q_len)`` (BC1S, unchanged).
            key:   ``(n_blocks, n_kv*head_dim, 1, block_size)`` (a blocked-cache slot).
            value: ``(n_blocks, n_kv*head_dim, 1, block_size)``.
            causal_mask: ``(1, n_blocks, block_size, 1, q_len)`` — block ``t`` slice is
                the flat ``(1, block_size, 1, q_len)`` mask for that block.

        Returns:
            ``(1, n_heads*head_dim, 1, q_len)`` — identical layout to ``SDPA``.
        """
        n_blocks = key.shape[0]
        head_dim = self.head_dim
        queries = query.split(head_dim, dim=1)  # each (1, head_dim, 1, q_len)
        n_heads = len(queries)
        batch = query.shape[0]
        q_len = query.shape[-1]

        # Pre-split each block into per-kv-head key/value tiles + a per-block mask,
        # mirroring SDPA's permutes (each block slot is the flat BC1S shape with the
        # block batch dim folded out). Keys carry the scale (as in SDPA).
        keys_per_block = []   # keys_per_block[t][kv] = (1, 1, head_dim, block_size)
        values_per_block = []  # values_per_block[t][kv] = (1, block_size, head_dim)
        masks_per_block = []   # masks_per_block[t] = (1, 1, q_len, block_size)
        for t in range(n_blocks):
            k_t = key[t].unsqueeze(0).transpose(-3, -1) * self._scale_factor  # (1, B, 1, C)
            ks = [kk.permute(0, 2, 3, 1) for kk in k_t.split(head_dim, dim=-1)]
            vs = [
                vv.permute(0, 2, 3, 1).squeeze(1)
                for vv in value[t].unsqueeze(0).split(head_dim, dim=1)
            ]
            keys_per_block.append(ks)
            values_per_block.append(vs)
            masks_per_block.append(causal_mask[:, t].permute(0, 2, 3, 1))

        n_kv = len(keys_per_block[0])
        kv_group_size = n_heads // n_kv

        outs = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            q = queries[head_idx].permute(0, 2, 3, 1)  # (1, 1, q_len, head_dim)
            # Running max / denom / weighted-sum. Init max to -40000, the same
            # fp16-safe -inf proxy the masks use (-1e30 overflows fp16 at export).
            # Block 0 always contains unmasked keys (key 0 ≤ every query position),
            # so after block 0 ``m`` is a real score and the init only zeroes the
            # empty initial accumulator (corr = exp(-40000 - real) = 0); a finite
            # init also avoids exp(-inf - -inf) = NaN.
            m = torch.full((batch, 1, q_len, 1), -40000.0, dtype=q.dtype)
            l = torch.zeros((batch, 1, q_len, 1), dtype=q.dtype)
            acc = torch.zeros((batch, q_len, head_dim), dtype=q.dtype)
            for t in range(n_blocks):
                s_t = q @ keys_per_block[t][kv_idx]  # (1, 1, q_len, block_size)
                s_t = s_t + masks_per_block[t]
                m_new = torch.maximum(m, s_t.max(dim=-1, keepdim=True).values)
                corr = torch.exp(m - m_new)          # (1, 1, q_len, 1)
                p_t = torch.exp(s_t - m_new)         # (1, 1, q_len, block_size)
                l = l * corr + p_t.sum(dim=-1, keepdim=True)
                pv = p_t.squeeze(1) @ values_per_block[t][kv_idx]  # (1, q_len, head_dim)
                acc = acc * corr.squeeze(1) + pv
                m = m_new
            out_h = acc / l.squeeze(1)               # (1, q_len, head_dim)
            outs.append(out_h.transpose(1, 2).unsqueeze(2))

        return torch.cat(outs, dim=1)

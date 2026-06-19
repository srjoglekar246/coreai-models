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

    The flat ``SDPA`` forms one score tensor whose key axis spans the full context;
    above ~32k that tensor exceeds the accelerator's per-dimension size limit and the
    global layers fall back to the GPU. ``BlockedSDPA`` is a drop-in with the **same
    signature** (it reads the flat cache slot ``(1, C, 1, ctx)`` and flat mask
    ``(1, ctx, 1, q)``) but walks the key axis in ``block_size`` chunks with the
    textbook **online-softmax (flash)** recurrence — algebraically identical to one big
    softmax (parity ~1e-6) while every score op's key axis is only ``block_size``.

    This keeps the global layers on the accelerator up to ctx ~32768; beyond that the
    ``(1, C, 1, ctx)`` slot read and ``(1, ctx, 1, q)`` mask themselves exceed the
    ~65536 per-dimension limit (a hardware/compiler limit, not an attention-shape one).
    The block loop is unrolled (``n_blocks`` static per export bucket); each block keeps
    per-head rank-4 ``q@k`` / rank-3 ``scores@v`` (no rank-5 matmul).
    """

    def __init__(
        self,
        head_dim: int | None = None,
        scale: float | torch.Tensor | None = None,
        block_size: int = 8192,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        with torch.device("cpu"):
            scale_t = torch.tensor(head_dim**-0.5 if scale is None else scale)
            self._scale_factor = nn.Buffer(
                scale if isinstance(scale, torch.Tensor) else scale_t, persistent=False
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Online-softmax attention over a flat cache slot (batch size 1).

        Args:
            query: ``(1, n_heads*head_dim, 1, q_len)`` (BC1S).
            key/value: ``(1, n_kv*head_dim, 1, ctx)`` — the flat global cache slot.
            causal_mask: ``(1, ctx, 1, q_len)`` — same flat mask as ``SDPA``.

        Returns:
            ``(1, n_heads*head_dim, 1, q_len)`` — identical layout to ``SDPA``.
        """
        head_dim, block_size = self.head_dim, self.block_size
        ctx = key.shape[-1]
        n_blocks = (ctx + block_size - 1) // block_size  # static (ctx fixed per bucket)

        queries = query.split(head_dim, dim=1)           # each (1, head_dim, 1, q_len)
        n_heads = len(queries)
        q_len = query.shape[-1]
        kv_group_size = n_heads // (key.shape[1] // head_dim)

        outs = []
        for head_idx in range(n_heads):
            c0 = head_dim * (head_idx // kv_group_size)  # this head's K/V channel base
            q = queries[head_idx].permute(0, 2, 3, 1)    # (1, 1, q_len, head_dim)
            # Running max / denominator / weighted sum. -40000 is the fp16-safe -inf
            # the mask uses; block 0 always has an unmasked key, so it becomes real.
            m = torch.full((1, 1, q_len, 1), -40000.0, dtype=q.dtype)
            l = torch.zeros((1, 1, q_len, 1), dtype=q.dtype)
            acc = torch.zeros((1, q_len, head_dim), dtype=q.dtype)
            for t in range(n_blocks):
                lo, hi = t * block_size, min((t + 1) * block_size, ctx)
                k = (key[:, c0:c0 + head_dim, :, lo:hi] * self._scale_factor).permute(0, 2, 1, 3)
                v = value[:, c0:c0 + head_dim, :, lo:hi].squeeze(2).transpose(1, 2)  # (1, B, head_dim)
                s = q @ k + causal_mask[:, lo:hi].permute(0, 2, 3, 1)                # (1, 1, q_len, B)

                m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
                corr = torch.exp(m - m_new)
                p = torch.exp(s - m_new)
                l = l * corr + p.sum(dim=-1, keepdim=True)
                acc = acc * corr.squeeze(1) + p.squeeze(1) @ v
                m = m_new
            outs.append((acc / l.squeeze(1)).transpose(1, 2).unsqueeze(2))

        return torch.cat(outs, dim=1)

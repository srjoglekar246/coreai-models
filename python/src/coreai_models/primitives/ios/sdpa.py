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

        # Apply the scale factor before QK^T for numerical stability. Keep
        # everything in BC1S split-by-head form so the matmuls are expressed as
        # direct einsum contractions — no per-head permute/reshape into and out
        # of (B, 1, S, head_dim) layout, which would otherwise force memory
        # copies in the compiled graph (one per head, per matmul).
        key = key * self._scale_factor
        queries = query.split(self.head_dim, dim=1)  # each (B, head_dim, 1, S_q)
        keys = key.split(self.head_dim, dim=1)  # each (B, head_dim, 1, S_k)
        values = value.split(self.head_dim, dim=1)  # each (B, head_dim, 1, S_k)

        n_heads = len(queries)
        kv_group_size = n_heads // len(keys)

        # Q @ K^T per head, contracting head_dim (d) while preserving the
        # singleton (o): (B, head_dim, 1, S_q) x (B, head_dim, 1, S_k) -> (B, S_k, 1, S_q)
        scores = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            scores.append(torch.einsum("bdoq,bdok->bkoq", queries[head_idx], keys[kv_idx]))

        full_scores = torch.cat(scores, dim=2)
        masked_scores = full_scores + torch.cat([causal_mask] * n_heads, dim=2)
        full_scores = masked_scores.softmax(1)

        scores = full_scores.split(1, dim=2)

        # scores @ V per head, contracting the key axis (k):
        # (B, S_k, 1, S_q) x (B, head_dim, 1, S_k) -> (B, head_dim, 1, S_q)
        weights = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            weights.append(torch.einsum("bkoq,bdok->bdoq", scores[head_idx], values[kv_idx]))

        final_score = torch.cat(weights, dim=1)
        return final_score

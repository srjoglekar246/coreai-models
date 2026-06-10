# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma4 text decoder iOS implementation.

Supports E2B and E4B variants. Text decoder only (no vision/audio).

Key differences from other iOS models (Qwen3, Llama):
  - Dual head dimensions (sliding=256, global=512)
  - Shared KV cache (later layers reuse KV from earlier same-type layers)
  - V-norm (scale-free RMSNorm on values)
  - Per-Layer Embeddings (PLE) as external input
  - 4 RMSNorm layers per block (pre+post attention, pre+post MLP)
  - Layer scalar (learned per-layer buffer)
  - Proportional RoPE for global layers (partial_rotary_factor=0.25)
  - Double-wide MLP for E2B shared layers
  - GELU activation (not SiLU)
  - RMSNorm (not RMSNormPlusOne)
  - Merged KV cache (sliding+global along channel dim)
"""


import torch
import torch.nn as nn
from safetensors.torch import save_file

from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class Gemma4CombinedRoPE(RoPECache):
    """Single RoPE cache for Gemma4's dual head dims.

    Gemma4 has two RoPE variants — standard for sliding-attention layers
    (head_dim 256) and proportional (0.25 rotary) for global layers
    (head_dim 512). We concatenate both variants' cos/sin tables along the head
    dim into a single ``[max_pos, sliding_hd + global_hd]`` cache and gather
    once. Callers slice ``[:sliding_hd]`` for sliding layers and ``[sliding_hd:]``
    for global layers.
    """

    def __init__(
        self,
        sliding_head_dim: int,
        global_head_dim: int,
        max_cache_size: int,
        sliding_base: float,
        global_base: float,
        partial_rotary_factor: float = 0.25,
    ) -> None:
        self._sliding_head_dim = sliding_head_dim
        self._global_head_dim = global_head_dim
        self._sliding_base = sliding_base
        self._global_base = global_base
        self._partial_rotary_factor = partial_rotary_factor
        super().__init__(sliding_head_dim + global_head_dim, max_cache_size, sliding_base)

    @staticmethod
    def _emb(theta: torch.Tensor, max_cache_size: int) -> torch.Tensor:
        seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)
        freqs = seq_idx[:, None] * theta
        return torch.concatenate((freqs, freqs), dim=-1)

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        with torch.device("cpu"):
            # Sliding (standard RoPE).
            s_theta = 1.0 / (
                self._sliding_base
                ** (
                    torch.arange(0, self._sliding_head_dim, 2, dtype=torch.float32)
                    / self._sliding_head_dim
                )
            )
            s_emb = self._emb(s_theta, self._max_cache_size)

            # Global (proportional RoPE: only partial_rotary_factor of dims rotate).
            hd = self._global_head_dim
            rope_angles = int(self._partial_rotary_factor * hd // 2)
            nope_angles = hd // 2 - rope_angles
            inv_freq = 1.0 / (
                self._global_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / hd)
            )
            if nope_angles > 0:
                g_theta = torch.cat(
                    [inv_freq, torch.zeros(nope_angles, dtype=torch.float32)], dim=0
                )
            else:
                g_theta = inv_freq
            g_emb = self._emb(g_theta, self._max_cache_size)

            cos = torch.cat([torch.cos(s_emb), torch.cos(g_emb)], dim=-1)
            sin = torch.cat([torch.sin(s_emb), torch.sin(g_emb)], dim=-1)
            self.cos_cached = torch.nn.Buffer(cos.to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(sin.to(dtype=dtype), persistent=False)


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (for v_norm)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(self._eps, dtype=x.dtype)
        square = x * x
        mean_square = square.mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)
        return x * inv_rms


class GeGLUMLP(nn.Module):
    """MLP with GELU-gated activation (GeGLU) using Conv2d for iOS."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _, dim = x.shape
        x = x.reshape(batch_size * query_len, dim, 1, 1)

        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        down_tensor = self.down_proj(up_tensor * gate_tensor)

        return down_tensor.reshape(batch_size, query_len, 1, dim)


class Attention(nn.Module):
    """Gemma4 iOS attention supporting both sliding and global modes."""

    def __init__(
        self,
        config,
        layer_idx: int,
        is_sliding: bool,
        is_kv_shared: bool,
        kv_shared_source_idx: int | None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.is_kv_shared = is_kv_shared
        self.kv_shared_source_idx = kv_shared_source_idx

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if is_sliding else config.global_head_dim

        self.sdpa = SDPA(head_dim=self.head_dim, scale=1.0)

        self.q_proj = nn.Conv2d(dim, self.n_heads * self.head_dim, kernel_size=1, bias=False)
        if not is_kv_shared:
            self.k_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, dim, kernel_size=1, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not is_kv_shared:
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
        channel_begin: int = 0,
        channel_end: int | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads
        head_dim = self.head_dim

        x_conv = x.transpose(-3, -1)
        query = self.q_proj(x_conv)

        # Reshape Q for norm + RoPE, keeping (B, S, n_heads, head_dim) so norm
        # (over head_dim) and RoPE broadcast over heads at dim 2 — no n_heads<->S
        # transpose into head format is needed.
        query = query.transpose(-3, -1).reshape(batch_size, query_len, n_heads, head_dim)
        query = self.q_norm(query)
        query = apply_rope(query, rope_cos, rope_sin, head_axis=2)

        if self.is_kv_shared:
            assert cache is not None and self.kv_shared_source_idx is not None
            # Reshape Q back to BC1S for SDPA
            query = query.reshape(batch_size, query_len, 1, n_heads * head_dim).transpose(-3, -1)
            # Read K/V from merged cache at source layer's channel range
            key = cache.k_cache[self.kv_shared_source_idx, :, channel_begin:channel_end, :, :]
            value = cache.v_cache[self.kv_shared_source_idx, :, channel_begin:channel_end, :, :]
            output = self.sdpa(query, key, value, causal_mask)
        else:
            key = self.k_proj(x_conv)
            value = self.v_proj(x_conv)

            # Reshape K for norm + RoPE (B, S, n_kv_heads, head_dim)
            key = key.transpose(-3, -1).reshape(batch_size, query_len, n_kv_heads, head_dim)
            key = self.k_norm(key)
            key = apply_rope(key, rope_cos, rope_sin, head_axis=2)

            # K back to BC1S
            key = key.reshape(batch_size, query_len, 1, n_kv_heads * head_dim).transpose(-3, -1)

            # V: apply v-norm in (B, S, n_kv_heads, head_dim) then convert back to BC1S
            value = value.transpose(-3, -1).reshape(batch_size, query_len, n_kv_heads, head_dim)
            value = self.v_norm(value)
            value = value.reshape(batch_size, query_len, 1, n_kv_heads * head_dim).transpose(-3, -1)

            # Update merged cache at the correct channel sub-range
            if cache is not None:
                key, value = cache.update_and_fetch_channels(
                    self.layer_idx,
                    in_step,
                    key,
                    value,
                    query_len,
                    channel_begin,
                    channel_end,
                )

            # Q back to BC1S for SDPA
            query = query.reshape(batch_size, query_len, 1, n_heads * head_dim).transpose(-3, -1)
            output = self.sdpa(query, key, value, causal_mask)

        output = self.o_proj(output)
        return output.transpose(-3, -1)


class TransformerBlock(nn.Module):
    """Gemma4 iOS decoder layer with PLE injection and layer scalar."""

    def __init__(
        self,
        config,
        layer_idx: int,
        is_sliding: bool,
        is_kv_shared: bool,
        kv_shared_source_idx: int | None,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.self_attn = Attention(
            config=config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
            is_kv_shared=is_kv_shared,
            kv_shared_source_idx=kv_shared_source_idx,
        )
        self.mlp = GeGLUMLP(hidden_size, intermediate_size)

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Conv2d(
                hidden_size, self.hidden_size_per_layer_input, kernel_size=1, bias=False
            )
            self.per_layer_projection = nn.Conv2d(
                self.hidden_size_per_layer_input, hidden_size, kernel_size=1, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        cache: KVCacheHandler | None = None,
        channel_begin: int = 0,
        channel_end: int | None = None,
    ) -> torch.Tensor:
        # Self-attention (pre+post norm)
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            cache,
            channel_begin,
            channel_end,
        )
        h = x + self.post_attention_layernorm(r)

        # MLP (pre+post norm)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        h = h + self.post_feedforward_layernorm(r)

        # Per-Layer Embedding injection
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            batch_size, seq_len, _, ple_dim = per_layer_input.shape
            hidden_size = h.shape[-1]

            h_conv = h.reshape(batch_size * seq_len, hidden_size, 1, 1)
            gate = nn.functional.gelu(self.per_layer_input_gate(h_conv), approximate="tanh")

            pli_conv = per_layer_input.reshape(batch_size * seq_len, ple_dim, 1, 1)
            ple_out = self.per_layer_projection(gate * pli_conv)
            ple_out = ple_out.reshape(batch_size, seq_len, 1, hidden_size)
            h = h + self.post_per_layer_input_norm(ple_out)

        h = h * self.layer_scalar
        return h


class Gemma4Model(nn.Module):
    """Gemma4 iOS text decoder model (without LM head)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            ple_dim = num_layers * config.hidden_size_per_layer_input
            self.per_layer_model_projection = nn.Conv2d(
                hidden_size, ple_dim, kernel_size=1, bias=False
            )
            self.per_layer_model_projection_scale = hidden_size**-0.5
            self.per_layer_projection_norm = RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
            self.per_layer_input_scale = 2.0**-0.5

        layer_types = config.layer_types
        first_kv_shared_idx = num_layers - config.num_kv_shared_layers

        layers = []
        for i in range(num_layers):
            is_sliding = layer_types[i] == "sliding_attention"
            is_kv_shared = i >= first_kv_shared_idx and config.num_kv_shared_layers > 0

            kv_shared_source_idx = None
            if is_kv_shared:
                for j in range(first_kv_shared_idx - 1, -1, -1):
                    if layer_types[j] == layer_types[i]:
                        kv_shared_source_idx = j
                        break

            if config.use_double_wide_mlp and is_kv_shared:
                intermediate_size = config.intermediate_size * 2
            else:
                intermediate_size = config.intermediate_size

            layers.append(
                TransformerBlock(
                    config,
                    layer_idx=i,
                    is_sliding=is_sliding,
                    is_kv_shared=is_kv_shared,
                    kv_shared_source_idx=kv_shared_source_idx,
                    intermediate_size=intermediate_size,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Single combined RoPE cache (sliding + global tables concatenated),
        # gathered once per forward. Sliding layers use [:head_dim], global
        # layers [head_dim:].
        sliding_params = config.rope_parameters["sliding_attention"]
        global_params = config.rope_parameters["full_attention"]
        self.rope = Gemma4CombinedRoPE(
            sliding_head_dim=config.head_dim,
            global_head_dim=config.global_head_dim,
            max_cache_size=config.max_position_embeddings,
            sliding_base=sliding_params["rope_theta"],
            global_base=global_params["rope_theta"],
            partial_rotary_factor=global_params.get("partial_rotary_factor", 0.25),
        )

    def _compute_per_layer_inputs(
        self,
        ple_embeddings: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PLE: pre-gathered PLE embedding + projected main embedding."""
        num_layers = self.config.num_hidden_layers
        ple_dim = self.config.hidden_size_per_layer_input
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        hidden_size = inputs_embeds.shape[-1]

        per_layer_emb = ple_embeddings.reshape(batch_size, seq_len, num_layers, ple_dim)

        embeds_conv = inputs_embeds.reshape(batch_size * seq_len, hidden_size, 1, 1)
        per_layer_proj = self.per_layer_model_projection(embeds_conv)
        per_layer_proj = per_layer_proj.reshape(batch_size, seq_len, num_layers, ple_dim)
        per_layer_proj = per_layer_proj * torch.tensor(
            self.per_layer_model_projection_scale, dtype=per_layer_proj.dtype
        )
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)

        return (per_layer_proj + per_layer_emb) * torch.tensor(
            self.per_layer_input_scale, dtype=per_layer_proj.dtype
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = token_embeddings

        n_kv_heads = self.config.num_key_value_heads
        sliding_channels = n_kv_heads * self.config.head_dim
        global_channel_begin = sliding_channels
        global_channel_end = sliding_channels + n_kv_heads * self.config.global_head_dim

        layer_types = self.config.layer_types

        # Gather RoPE cos/sin ONCE (single composite), then slice the sliding
        # and global sub-ranges out of the combined table.
        head_dim = self.config.head_dim
        combined_cos, combined_sin = self.rope.gather_cos_sin(position_ids)
        sliding_cos = combined_cos[..., :head_dim]
        sliding_sin = combined_sin[..., :head_dim]
        global_cos = combined_cos[..., head_dim:]
        global_sin = combined_sin[..., head_dim:]

        per_layer_inputs = None
        if self.hidden_size_per_layer_input and ple_embeddings is not None:
            per_layer_inputs = self._compute_per_layer_inputs(ple_embeddings, h)

        for i, layer in enumerate(self.layers):
            per_layer_input = None
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, :, i : i + 1, :]

            is_sliding = layer_types[i] == "sliding_attention"
            if is_sliding:
                rope_cos, rope_sin = sliding_cos, sliding_sin
                ch_begin, ch_end = 0, sliding_channels
            else:
                rope_cos, rope_sin = global_cos, global_sin
                ch_begin, ch_end = global_channel_begin, global_channel_end

            h = layer(
                h, rope_cos, rope_sin, in_step, causal_mask,
                per_layer_input=per_layer_input,
                cache=cache,
                channel_begin=ch_begin,
                channel_end=ch_end,
            )

        return self.norm(h)


class Gemma4Extend(nn.Module):
    """Gemma4 iOS extend module: transformer + lm_head with merged KV cache."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Gemma4Model(config)
        self.embed_scale = config.hidden_size**0.5
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.ple_scale: torch.Tensor | None = None
        self.ple_zp: torch.Tensor | None = None
        self.tie_word_embeddings = config.tie_word_embeddings
        self.prefill_mode = False

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Merged KV cache: sliding+global concatenated along channel dim
        n_kv_heads = config.num_key_value_heads
        total_kv_channels = n_kv_heads * (config.head_dim + config.global_head_dim)
        self.kv_cache = KVCacheHandler(config.num_hidden_layers, total_kv_channels)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape

        # Scale embeddings by sqrt(hidden_size)
        transformer_input = transformer_input * torch.tensor(
            self.embed_scale, dtype=transformer_input.dtype
        )

        # Dequantize PLE if INT8
        if ple_embeddings is not None and ple_embeddings.dtype == torch.int8:
            ple_embeddings = dequantize_per_tensor(
                ple_embeddings, self.ple_scale, self.ple_zp, transformer_input.dtype
            )

        out = self.model(
            transformer_input,
            position_ids,
            in_step,
            causal_mask,
            cache=self.kv_cache,
            ple_embeddings=ple_embeddings,
        )

        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            return self.lm_head(out.transpose(-2, -3))

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )

        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)

        return (embedding_table @ out).transpose(-1, -2)


class Gemma4ForCausalLMForiOS(BaseForCausalLMForiOS):
    """Gemma4 iOS text-only CausalLM (extracted from multimodal HF model)."""

    _HF_MODEL_CLASS = None  # We override from_hf directly

    def __init__(self, config, model_device: str = "cpu") -> None:
        # INT8 embedding table (the iOS default); routes the token gather
        # through the fused_dequant_gather composite.
        super().__init__(config, model_device, disable_embedding_quantization=False)

    def _init_model(self, config) -> None:
        self.extend = Gemma4Extend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        ple_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
            ple_embeddings,
        )

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        """Extract text config from the multimodal Gemma4Config."""
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            if hasattr(text_config, "layer_types") and text_config.layer_types:
                text_config.layer_types = text_config.layer_types[:num_layers]
            orig_num_layers = text_config.num_hidden_layers
            orig_first_shared = orig_num_layers - text_config.num_kv_shared_layers
            new_shared = max(0, num_layers - orig_first_shared)
            text_config.num_kv_shared_layers = new_shared
            text_config.num_hidden_layers = num_layers

        return text_config

    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
    ) -> "Gemma4ForCausalLMForiOS":
        """Load from HF multimodal model, extracting text decoder only."""
        from transformers import Gemma4ForConditionalGeneration

        hf_model = Gemma4ForConditionalGeneration.from_pretrained(
            huggingface_model_id, torch_dtype=target_dtype
        )

        config = cls._get_reauthored_config(
            hf_model.config, max_context_length, num_layers=num_layers
        )

        full_sd = hf_model.state_dict()
        prefix = "model.language_model."
        text_sd: dict[str, torch.Tensor] = {}
        for k, v in full_sd.items():
            if k.startswith(prefix):
                text_sd["model." + k[len(prefix):]] = v

        if "lm_head.weight" not in text_sd and "model.embed_tokens.weight" in text_sd:
            text_sd["lm_head.weight"] = text_sd["model.embed_tokens.weight"]

        del hf_model, full_sd

        if num_layers is not None:
            from coreai_models.models.base import _is_layer_key_beyond

            text_sd = {
                k: v for k, v in text_sd.items() if not _is_layer_key_beyond(k, num_layers)
            }

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)

        model._mutate_state_dict(text_sd)

        for k, v in text_sd.items():
            if (
                v.dtype != target_dtype
                and v.is_floating_point()
                and "embedding_table" not in k
                and "zero_point" not in k
            ):
                raise ValueError(
                    f"tensor {k} in incorrect dtype {v.dtype}. Expected {target_dtype}."
                )

        strict = num_layers is None
        model.load_state_dict(text_sd, assign=True, strict=strict)

        # Register PLE scale/zp as buffers after load_state_dict
        if hasattr(model, "_ple_scale_pending"):
            model.extend.ple_scale = model._ple_scale_pending
            model.extend.ple_zp = model._ple_zp_pending
            del model._ple_scale_pending
            del model._ple_zp_pending

        if mmap_path is not None:
            from coreai_models.models.base import move_model_to_disk

            move_model_to_disk(model, path=mmap_path)

        return model

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Transform HF state dict for iOS Gemma4 model."""
        config = self.config
        num_layers = config.num_hidden_layers
        first_kv_shared_idx = num_layers - config.num_kv_shared_layers
        ple_dim = config.hidden_size_per_layer_input

        # Extract PLE embedding weight (externalized from graph)
        if ple_dim > 0:
            ple_key = "model.embed_tokens_per_layer.weight"
            if ple_key in state_dict:
                ple_weight = state_dict.pop(ple_key)
                expected_dim = num_layers * ple_dim
                if ple_weight.shape[1] > expected_dim:
                    ple_weight = ple_weight[:, :expected_dim].contiguous()
                self._ple_weight = ple_weight

                # Compute INT8 quantization for PLE with embed_scale folded in
                ple_embed_scale = config.hidden_size_per_layer_input**0.5
                _, ple_scale, ple_zp = quantize_per_tensor(
                    (ple_weight.float() * ple_embed_scale), nbits=8, symmetric=True
                )
                self._ple_scale_pending = ple_scale.to(torch.float16)
                self._ple_zp_pending = ple_zp

            # Truncate per_layer_model_projection if needed
            proj_key = "model.per_layer_model_projection.weight"
            if proj_key in state_dict:
                full_proj = state_dict[proj_key]
                expected_out = num_layers * ple_dim
                if full_proj.shape[0] > expected_out:
                    state_dict[proj_key] = full_proj[:expected_out, :]

        # Reshape weights for Conv2d and handle shared layers
        for i in range(num_layers):
            is_kv_shared = i >= first_kv_shared_idx and config.num_kv_shared_layers > 0

            for proj in ["q_proj", "o_proj"]:
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            if is_kv_shared:
                for proj in ["k_proj", "v_proj"]:
                    key = f"model.layers.{i}.self_attn.{proj}.weight"
                    state_dict.pop(key, None)
                state_dict.pop(f"model.layers.{i}.self_attn.k_norm.weight", None)
            else:
                for proj in ["k_proj", "v_proj"]:
                    weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                    if weight_key in state_dict:
                        state_dict[weight_key] = (
                            state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)
                        )

            for proj in ["gate_proj", "up_proj", "down_proj"]:
                weight_key = f"model.layers.{i}.mlp.{proj}.weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            if ple_dim > 0:
                for proj in ["per_layer_input_gate", "per_layer_projection"]:
                    weight_key = f"model.layers.{i}.{proj}.weight"
                    if weight_key in state_dict:
                        state_dict[weight_key] = (
                            state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)
                        )

        if ple_dim > 0:
            proj_key = "model.per_layer_model_projection.weight"
            if proj_key in state_dict:
                state_dict[proj_key] = state_dict[proj_key].unsqueeze(-1).unsqueeze(-1)

        # Wrap model weights under extend.model.* prefix
        new_state_dict = {}
        keys_to_pop = set()
        for k in state_dict:
            if k.startswith("model.") and "embed_tokens" not in k:
                new_state_dict[f"extend.{k}"] = state_dict[k]
                keys_to_pop.add(k)
        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        # Handle embeddings: quantize and set up for iOS layout
        embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(
                embedding_table, nbits=8, symmetric=True
            )
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)

        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point
        state_dict.pop("model.embed_tokens.weight")

        # Handle lm_head
        if not config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]
        state_dict.pop("lm_head.weight", None)

    def dump_ple_embedding(self, output_path: str, model_name: str) -> str:
        """Dump the externalized PLE embedding table as a quantized INT8 safetensors file."""
        import os

        ple_weight = self._ple_weight
        config = self.config
        ple_dim = config.hidden_size_per_layer_input
        embed_scale = str(ple_dim**0.5)

        ple_scaled = ple_weight.float() * float(embed_scale)
        ple_q, ple_scale, ple_zp = quantize_per_tensor(ple_scaled, nbits=8, symmetric=True)

        os.makedirs(output_path, exist_ok=True)
        ple_path = os.path.join(output_path, f"{model_name}_ple.safetensors")

        save_file(
            {"embed_tokens_per_layer": ple_q.contiguous()},
            ple_path,
            metadata={
                "embed_scale": embed_scale,
                "ple_scale": str(float(ple_scale)),
                "ple_zero_point": str(int(ple_zp)),
                "dtype": "SI8",
            },
        )
        return ple_path

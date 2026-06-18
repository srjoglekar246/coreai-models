# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Run the *real* iOS Gemma4 torch model on a needle prompt, in fp32, uncompressed.

Mimics the Swift runner end-to-end in Python (HF chat-template tokenization → chunked
prefill against the flat dual cache + chunked-flash global attention → greedy decode),
but with the full-fidelity torch model. This isolates the on-device recall failure:

  - if THIS recalls the needle  → the iOS model logic is correct; the gap is fp16 +
    4-bit palettization (the .aimodel) or the runner's tokenization/chat-template.
  - if THIS also fails           → a real attention/model bug the tiny-config parity
    (1 global layer, vocab 64) doesn't catch.

    uv run --with "transformers>=5.5.0" python python/scripts/gemma4_needle_torch.py \
        --prompt-file /tmp/needle_6k.txt --max-ctx 8192 --q-len 64 --max-new 12
"""

import argparse

import torch
from transformers import AutoTokenizer

from coreai_models.export.ios import QUERY_LENGTHS, sliding_ring_size
from coreai_models.models.ios.gemma4_text import (
    Gemma4CombinedRoPE,
    Gemma4ForCausalLMForiOS,
)

MODEL_ID = "google/gemma-4-E2B-it"
NEG = float("-inf")


def global_mask(ctx, q_len, aligned_step):
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=torch.float32)
    for i in range(q_len):
        p = aligned_step + i
        m[0, : p + 1, 0, i] = 0.0
    return m


def sliding_mask(S, q_len, aligned_step, window):
    m = torch.full((1, S, 1, q_len), NEG, dtype=torch.float32)
    for i in range(q_len):
        p = aligned_step + i
        for pos in range(max(0, p - window + 1), p + 1):
            m[0, pos % S, 0, i] = 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--q-len", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = ap.parse_args()
    q_len = args.q_len

    torch.manual_seed(0)
    DTYPE = torch.float16 if args.dtype == "fp16" else torch.float32

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = open(args.prompt_file).read()
    if args.no_chat_template:
        ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()
    else:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = enc["input_ids"][0].tolist()
    print(f"prompt tokens: {len(ids)} (chat_template={not args.no_chat_template})", flush=True)

    print("loading iOS model (fp32, uncompressed)...", flush=True)
    model = Gemma4ForCausalLMForiOS.from_hf(
        MODEL_ID, max_context_length=args.max_ctx, target_dtype=DTYPE
    ).eval()
    cfg = model.config

    rope = Gemma4CombinedRoPE(
        sliding_head_dim=cfg.head_dim,
        global_head_dim=cfg.global_head_dim,
        max_cache_size=args.max_ctx,
        sliding_base=cfg.rope_parameters["sliding_attention"]["rope_theta"],
        global_base=cfg.rope_parameters["full_attention"]["rope_theta"],
        partial_rotary_factor=cfg.rope_parameters["full_attention"].get(
            "partial_rotary_factor", 0.25
        ),
    ).to(DTYPE)

    n_kv = cfg.num_key_value_heads
    S = sliding_ring_size(cfg.sliding_window, max(QUERY_LENGTHS))
    W = cfg.sliding_window
    ctx = args.max_ctx
    n_g = model.extend.model.n_global_storing
    n_s = model.extend.model.n_sliding_storing
    ple_dim = cfg.hidden_size_per_layer_input
    ple_total = cfg.num_hidden_layers * ple_dim
    ple_scale = float(ple_dim) ** 0.5

    key_cache = torch.zeros(n_g, 1, n_kv * cfg.global_head_dim, 1, ctx, dtype=DTYPE)
    value_cache = key_cache.clone()
    skey = torch.zeros(n_s, 1, n_kv * cfg.head_dim, 1, S, dtype=DTYPE)
    svalue = skey.clone()

    def ple_for(token_ids):
        rows = model._ple_weight[token_ids].to(DTYPE) * ple_scale  # (n, ple_total)
        return rows.reshape(1, len(token_ids), 1, ple_total)

    @torch.no_grad()
    def step(chunk_ids, start, n_real):
        ids_t = torch.tensor(chunk_ids, dtype=torch.int32).reshape(1, q_len)
        pos = torch.arange(start, start + q_len, dtype=torch.int32).reshape(1, q_len)
        rope_cos, rope_sin = rope.gather_cos_sin(pos)
        in_step = torch.tensor([start], dtype=torch.int32)
        sliding_in_step = torch.tensor([start % S], dtype=torch.int32)
        cmask = global_mask(ctx, q_len, start)
        smask = sliding_mask(S, q_len, start, W)
        ple = ple_for(chunk_ids)
        out = model(
            ids_t, rope_cos, rope_sin, in_step, sliding_in_step,
            cmask, smask, key_cache, value_cache, skey, svalue, ple,
        )
        logits = out.reshape(q_len, cfg.vocab_size)
        return logits[n_real - 1]  # logits at the last real token of this chunk

    # Chunked prefill, q_len-aligned, pad the final partial chunk with token 0.
    seq = list(ids)
    generated = []
    last_logits = None
    pos = 0
    while pos < len(seq):
        chunk = seq[pos:pos + q_len]
        n_real = len(chunk)
        if n_real < q_len:
            chunk = chunk + [0] * (q_len - n_real)
        last_logits = step(chunk, pos, n_real)
        pos += n_real

    # Greedy decode.
    eos_ids = set(tok.all_special_ids) | {tok.eos_token_id}
    for _ in range(args.max_new):
        nxt = int(last_logits.argmax().item())
        if nxt in eos_ids:
            break
        generated.append(nxt)
        # one-token "chunk" (padded to q_len) appended at the current position.
        chunk = [nxt] + [0] * (q_len - 1)
        last_logits = step(chunk, len(seq), 1)
        seq.append(nxt)

    print("GENERATED:", repr(tok.decode(generated, skip_special_tokens=True)), flush=True)
    print("contains 7741-MAGENTA-9:", "7741-MAGENTA-9" in tok.decode(generated), flush=True)


if __name__ == "__main__":
    main()

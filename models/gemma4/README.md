# Gemma 4

Google's Gemma 4 text decoder for energy-efficient on-device inference via Core AI, optimized for the
accelerator and for **very large context windows** (up to 131,072 tokens) for on-device agentic tasks.
The Core AI recipe targets the text decoder only (no vision/audio).

## 1. Introduction

This recipe runs Gemma 4 **E2B** at long context on-device, correct to needle-in-a-haystack recall.

**TL;DR:** context lengths **up to 131,072 are supported and numerically correct** on the accelerator.
The practical sweet spot is **≤ 32k**, which runs fast at a small, flat memory footprint. 131,072 works
too, with two caveats — slower per-step latency at the top end, and a one-time model-preparation
overhead that is removed by full ahead-of-time precompilation (see [Next steps](#6-next-steps)).

Indicative performance (Apple silicon laptop, full E2B, 4-bit weights, greedy; see
[Performance](#5-export-build-and-run)):

| Prompt tokens | Decode tokens | Prompt tok/s | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 214 | 100 | 1128 | 25.5 |
| 814 | 100 | 3598 | 23.2 |
| 3014 | 100 | 615 | 11.8 |
| 7014 | 100 | 503 | 11.4 |
| 15014 | 100 | 420 | 10.3 |
| 28014 | 100 | 386 | 8.8 |

*Decode speed depends on the context window the model selects, not the exact prompt length: prompts
that land on the same context program decode at the same rate. Prompt tok/s peaks around ~800 tokens —
tiny prompts are bound by a fixed prefill overhead, longer ones by the quadratic cost of prefilling the
growing context. Steady-state memory (weights + caches + activations) is roughly flat across the table
at **~4.4 GB**, and excludes the one-time model-preparation overhead (see [Next steps](#6-next-steps)).*

**131,072 is also supported** at the same memory footprint, but with much higher per-step latency
(roughly ~1 second per generated token, and ~1.4 seconds per prefill chunk, at the maximum window) —
correct, but not interactive at the very top end.

## 2. Gemma 4 E2B

A small (effective ~2B) text decoder. Its dimensions and the architectural choices that shape the
on-device design:

- **35 decoder layers**, hidden size **1536**, double-wide MLP (intermediate size 6144), GeGLU. Vocab
  **262144**, tied input/output embedding (262144 × 1536 ≈ **402M** params). Final logit softcap 30.
- **Interleaved attention, 4 sliding : 1 global.** 28 layers are **sliding-window** (window 512 —
  attend only the last 512 keys); 7 are **global** (full history), placed at every 5th layer (indices
  4, 9, 14, 19, 24, 29, 34). Only the global layers' cost grows with context length.
- **Multi-query attention.** 8 query heads, **1 KV head** → the KV cache is a single head per storing
  layer. Sliding layers use head dim 256; global layers use head dim 512 with **partial-rotary** RoPE
  (only 128 of the 512 dims rotated). The two attention types use different RoPE bases.
- **Shared KV across layers.** The last 20 layers reuse the K/V of earlier same-type layers, so only
  **15 layers actually store K/V** (not 35). This is what makes the global cache affordable at long
  context.
- **Per-Layer Embeddings (PLE).** In addition to the token embedding, each token carries a small
  **256-dim** embedding *per layer*, mixed into that layer's input. The table is vocab × layers × 256
  = 262144 × 35 × 256 ≈ **2.35B params** (~2.35 GB at int8), i.e. ~67M params per layer — the dominant
  share of stored weights, gathered per token (35 × 256 per token) rather than run as matmuls.

## 3. On-device model I/O

Each decoder function is statically shaped, batch 1, named `extend_{ctx}_{q}` (decode) or
`prompt_opt_{ctx}_{q}` (prefill), where `ctx` is the context window and `q` the query length.

```
inputs:
  transformer_input    (1, q, 1, hidden)
  rope_cos, rope_sin   (1, q, 768)            runtime-precomputed RoPE rows
  in_step              (1,) int32             absolute global cache write offset
  sliding_in_step      (1,) int32             sliding ring write offset (= in_step % S)
  causal_mask          (1, ctx, 1, q)         flat global mask
  sliding_causal_mask  (1, S, 1, q)           windowed mask for the ring
  embedding_table, ple_embeddings             int8

states (persist across steps, both flat):
  key_cache,         value_cache          (n_global_storing, 1, n_kv·512, 1, ctx)
  sliding_key_cache, sliding_value_cache  (n_sliding_storing, 1, n_kv·256, 1, S=576)

outputs: out_logits
```

The runtime owns all dynamic state — write offsets, masks, and RoPE rows are computed per step and
passed in, so the graph itself is fully static.

## 4. Key changes

Two innovations keep long contexts feasible on the accelerator: a compact sliding-window cache, and
chunked (flash) global attention.

### 4.1 Sliding-window KV cache (ring)

The 28 sliding layers only ever attend the last 512 keys, so giving them a full-context cache would
waste memory. Instead the sliding layers share one compact **ring cache** of fixed depth `S = 576`
(> 512, with headroom for the query block), independent of context length. (The global layers keep an
ordinary flat full-context cache written at the absolute position; it is the only cache that grows with
context.)

- **Indexing / cache update.** The write position is `sliding_in_step = in_step % S`. Each step writes
  the new K/V row(s) at that offset; once the absolute position passes `S` the writes wrap and
  overwrite the oldest rows. The cache always holds the most recent ≤ S positions in ring order, never
  the full history.
- **Causal mask.** Because entries are stored in *ring* order (not contiguous time order), a plain
  lower-triangular mask is wrong. The runtime builds a windowed mask of shape `(1, S, 1, q)` that, for
  each query, admits exactly the ring slots holding keys in `[pos − 511, pos]` and masks both future
  positions and the stale wrapped-around slots beyond the 512-key window.

*See `primitives/ios/cache.py` and `models/ios/gemma4_text.py`.*

### 4.2 Blocked (flash) global attention

A global layer's attention score spans the full context on its key axis; at long context that single
tensor exceeds the sizes the accelerator supports, and the layer falls back to a slower path. The
`BlockedSDPA` primitive is a drop-in for standard attention that never materializes the full-width
score: it walks the key axis in fixed-width chunks with an online-softmax (flash) recurrence —
mathematically identical to one big softmax (parity ~1e-6).

```
# per head; ctx fixed per window, so the block count is static
m, l, o = -inf, 0, 0
# walk the long key axis in block_size-wide chunks; every op below stays within the accelerator's limits
for [lo:hi] in blocks of size block_size:
    k = scale * key[:, head, :, lo:hi]
    # sliced but never transposed on the context axis (that would force a whole-cache transpose)
    v = value[:, head, :, lo:hi]
    s = q @ k + mask[lo:hi]
    m_new = max(m, rowmax(s))
    # block-scaled weights + a normalized running output stay bounded at any ctx (plain flash overflows fp16 ~15k keys)
    p    = exp(s - m_new) / block_size
    prev = l * exp(m - m_new)
    l    = prev + rowsum(p)
    # convex update; algebraically one big softmax (parity ~1e-6)
    o    = (o * prev + p @ v) / l
    m    = m_new
return o
```

*See `BlockedSDPA` in `primitives/ios/sdpa.py`.*

## 5. Export, build, and run

```bash
# Export. Default context-window ladder up to a max context:
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --max-context-length 32768
```

The preset uses 4-bit palettization (channel group size 32) and an 8-bit embedding by default. Use
`--compression none` for full precision, or `--dry-run` to preview the resolved config.

```bash
# Run a Core AI language model (the exported model folder):
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## 6. Next steps

### Model-preparation overhead

**What it is.** Loading the exported model for the first time prepares its compute graphs. This is
memory-intensive — proportional to the number and size of context windows in the ladder — and is
repeated on each process launch, while the steady-state footprint (the ~4.4 GB above) is small.

**How to avoid it.** The durable fix is **full ahead-of-time precompilation** — compiling the model to
a ready-to-run asset so the device only loads it, with no per-launch graph preparation and low steady
memory. On-device deployments should use the precompiled asset once Core AI's tooling exposes loading
it directly in the runtime; the steady-state memory above is representative of that path.

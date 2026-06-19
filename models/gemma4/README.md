# Gemma 4

Google's Gemma 4 models for energy-efficient on-device inference via Core AI. The
Core AI recipe targets the text decoder only (no vision/audio) and is optimized for
the Neural Engine (iOS), with support for very large context windows (up to 131,072
tokens) for on-device agentic tasks.

## Supported Models

Gemma 4 E2B Instruct (iOS-style)

## Gated Access

Gemma 4 is gated on [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) (HF).
You will need to accept the terms of the
[license](https://huggingface.co/google/gemma-4-E2B-it), generate a HF token, and add
your HF token to your machine before exporting this model.

```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Setup to export models

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export models

Gemma 4 is exported for iOS (Neural Engine) with a fixed context length:

```bash
# Default context length
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS

# Large context (up to 32768 tokens)
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --max-context-length 32768
```

The iOS preset uses 4-bit palettization (channel group size 32) and an 8-bit Embedding
by default.

**Options:**

```bash
# Full precision
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --compression none

# Smaller palettization group
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS \
    --compression 4bit_weight_palettized_group8

# Custom context length (iOS requires a fixed value)
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --max-context-length 4096

# Preview resolved config without exporting
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --dry-run
```

## Large-context design

Gemma 4 interleaves two attention types per layer — **sliding-window** layers that
attend only the last 512 tokens, and **global** layers that attend the full history.
The Core AI recipe implements both correctly on-device and keeps long contexts
feasible on the Neural Engine:

- **Two compact KV caches.** A full-context **global** cache and a small,
  fixed-depth **sliding-window** ring cache, each holding only the layers and
  channels it needs (later layers reuse earlier same-type KV). The sliding ring is a
  constant size regardless of context length, so only the global cache grows with the
  prompt.

- **Chunked (flash) global attention.** Global attention is computed in fixed-width
  chunks over the cache with an online-softmax recurrence (mathematically identical to
  standard attention, parity ~1e-6). This keeps every attention tensor within the
  sizes the Neural Engine supports, so the global layers stay Neural-Engine-resident
  for context lengths up to 32,768 tokens — including correct recall of facts buried
  deep in a long prompt (needle-in-a-haystack). The model still exports and runs at
  larger context lengths (up to 131,072), with the global layers running partly off
  the Neural Engine beyond 32,768.

- **Per-context program ladder + right-sized cache.** The model is exported as a
  ladder of statically-shaped programs at power-of-two context lengths (256, 512, …,
  up to the configured maximum). At run time the engine picks the smallest program
  that covers the current position and allocates the global cache to *that* size,
  growing it (and carrying the written history forward) only as the context crosses
  into a larger bucket. A short prompt therefore runs a small graph and a small cache
  instead of always paying for the maximum context — short prompts decode several
  times faster and use a fraction of the memory of a full-context session.

Positions and RoPE are applied before caching, so cache layout does not affect
correctness. Decode throughput is highest at short contexts and decreases as the
context grows, since the global layers attend the entire history each step.

## Run a Core AI Language Model

### In your iOS and macOS applications via Foundation Models

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAILanguageModel(resourcesAt: modelURL)

let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "What is quantum computing?")

print(response)
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Performance

Measured on an **Apple M4 Max MacBook Pro** with the iOS (Neural Engine) export
running via `llm-runner`, 4-bit palettized weights, fully Neural-Engine-resident.
One real-text prompt per program-ladder bucket (sized so decode sits in that bucket);
a warm run is discarded first, then the measured run is reported. *Prompt tok/s* is
prefill over the whole prompt; *Decode tok/s* is per-step generation throughput. The
first use of each context size pays a one-time compile (cached thereafter).

| Prompt tokens | Decode context | Prompt tok/s | Decode tok/s |
| ------------: | -------------: | -----------: | -----------: |
|            95 |            256 |        1,545 |         24.1 |
|           294 |            512 |        3,152 |         24.1 |
|           767 |          1,024 |        5,077 |         23.6 |
|         1,623 |          2,048 |          760 |         12.8 |
|         3,349 |          4,096 |          569 |         12.5 |
|         6,785 |          8,192 |          503 |         12.8 |
|        13,847 |         16,384 |          435 |         10.9 |
|        19,290 |         32,768 |          424 |          9.4 |

Decode is fastest at short contexts (~24 tok/s for prompts that fit in ≤1k context),
steps down once the resident context grows past ~1k, then tapers gradually toward ~9
tok/s at 32k — the global layers attend the whole history each step, so cost scales
with the resident context. Because the cache is right-sized per session (above), a
short prompt keeps the fast path instead of paying the maximum-context cost.

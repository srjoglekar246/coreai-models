# Gemma 4

Google's Gemma 4 models for energy-efficient on-device inference via Core AI. The
Core AI recipe targets the text decoder only (no vision/audio) and is optimized for
the Neural Engine (iOS), with support for very large context windows (up to 131,072
tokens) for on-device agentic tasks.

## Supported Models

| Model                | Parameters         | macOS | iOS |
| -------------------- | ------------------ | ----- | --- |
| Gemma 4 E2B Instruct | E2B (effective 2B) | No    | Yes |

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

# Large context (up to 131,072 tokens)
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS --max-context-length 131072
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
The Core AI recipe implements both correctly on-device and compacts the KV cache so
that long contexts are feasible on the Neural Engine:

- **Two right-sized KV caches.** A full-context **global** cache and a small,
  fixed-depth **sliding-window** ring cache, each holding only the layers and
  channels it needs (later layers reuse earlier same-type KV). The sliding ring is a
  constant size regardless of context length, so only the global cache grows. At
  131,072 tokens the combined cache is ~812 MB instead of ~14 GB for a naive
  full-size cache.

- **Blocked global attention.** The global cache is stored block-wise and global
  attention is computed block-by-block with a flash / online-softmax recurrence
  (mathematically identical to standard attention). This keeps every attention
  tensor within the sizes the Neural Engine supports, so the model stays fully
  Neural-Engine-resident even at 131,072 tokens — including correct recall of facts
  buried deep in a long prompt (needle-in-a-haystack).

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

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.

## Performance

Measured on Apple silicon with the iOS (Neural Engine) export running via
`llm-runner`, 4-bit palettized weights. Decode runs fully on the Neural Engine;
throughput is highest at short contexts and scales down as the global layers attend
a longer history. Large-context exports incur a larger one-time compile on first
load (cached thereafter).

| Context        | Decode throughput (Neural Engine)                        |
| -------------- | -------------------------------------------------------- |
| Small (≤ 4k)   | ~20 tokens/s                                             |
| Large (~128k)  | lower — the full history is attended each decode step    |

# Gemma 4

Google's Gemma 4 models for energy-efficient on-device inference via Core AI. The
Core AI recipe targets the text decoder only (no vision/audio) and is optimized for
the Neural Engine (iOS).

## Supported Models

| Model              | Parameters  | macOS | iOS |
| ------------------ | ----------- | ----- | --- |
| Gemma 4 E2B Instruct | E2B (effective 2B) | No  | Yes |
| Gemma 4 E4B Instruct | E4B (effective 4B) | No  | Yes |

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
uv run coreai.llm.export google/gemma-4-E2B-it --platform iOS
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

Measured on an Apple M3 Max with the iOS (Neural Engine) export running via `llm-runner`.

| Stage             | Time      | Tokens | Throughput      |
| ----------------- | --------- | ------ | --------------- |
| Model load        | 79.6 ms   | —      | —               |
| Prompt (prefill)  | 115.3 ms  | 20     | 173.5 tokens/s  |
| Generation        | 1165.0 ms | 60     | 51.5 tokens/s   |
| Total             | 4.690 s   | —      | —               |


# blazr

[![CI](https://github.com/ml-rust/blazr/actions/workflows/ci.yml/badge.svg)](https://github.com/ml-rust/blazr/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/blazr.svg)](https://crates.io/crates/blazr) [![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Inference server for LLMs, with speech synthesis (TTS) and transcription (ASR). Supports standard HuggingFace models (Llama, Mistral, Qwen, Phi, Gemma, DeepSeek) and custom hybrid architectures (Mamba2, MLA, MoE). Loads SafeTensors, AWQ, GPTQ, and GGUF formats.

## Features

- **Multi-Architecture** - Llama, Mistral, Mamba2, MLA+MoE, hybrid models, and more
- **Multi-Format** - SafeTensors (F16/BF16), AWQ INT4, GPTQ INT4, GGUF (23 quantization levels)
- **Auto-Detection** - Detects architecture, format, and tokenizer from checkpoint tensor names
- **OpenAI-Compatible API** - Drop-in replacement with `/v1/completions` and `/v1/chat/completions`
- **Streaming** - Server-Sent Events (SSE) for real-time token generation
- **HuggingFace Hub** - Pull models directly from HuggingFace
- **Operations** - API-key auth, native TLS, Prometheus `/metrics`, request timeout, graceful shutdown on SIGTERM
- **CUDA Acceleration** - Optional GPU inference with optimized quantization kernels
- **Speech** - Text-to-speech (Kokoro, VoxCPM2, VITS) and Whisper transcription with word timestamps, over HTTP and the `speak`, `transcribe` and `sound-review` commands (`audio` feature)

## Quick Start

### Installation

#### Prebuilt binaries

Every [GitHub release](https://github.com/ml-rust/blazr/releases) carries one
archive per platform. Each holds the `blazr` binary, `LICENSE`, `README.md`,
`NOTICE`, `LICENSE-LLAMA` and `THIRD-PARTY-LICENSES.txt`.
All of them include the `audio` feature.

| Archive                    | Platform                    |
| -------------------------- | --------------------------- |
| `blazr-linux-x64.tar.gz`   | Linux x86_64, CPU           |
| `blazr-linux-arm64.tar.gz` | Linux aarch64, CPU          |
| `blazr-macos-arm64.tar.gz` | macOS on Apple silicon, CPU |
| `blazr-windows-x64.zip`    | Windows x86_64, CPU         |

The names carry no version, so the latest release always has the same URL:

```bash
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/blazr-linux-x64.tar.gz
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar xzf blazr-linux-x64.tar.gz
./blazr --help
```

`SHA256SUMS` lists the SHA-256 of every archive in the release. On macOS, use
`shasum -a 256 --check --ignore-missing SHA256SUMS`.

The Linux x86_64 archive needs glibc 2.35 or newer, and the aarch64 archive
glibc 2.39 or newer. No archive includes CUDA. Install with `--features
audio,cuda` for GPU inference: the build needs the CUDA toolkit, 12.8 or newer.

#### From crates.io

```bash
cargo install blazr --locked --features audio          # CPU
cargo install blazr --locked --features audio,cuda     # NVIDIA CUDA
```

#### From source

```bash
# Build (CPU-only)
cargo build --release

# Build with CUDA support (requires CUDA 12.8+ or 13.x)
cargo build --release --features cuda

# Add speech: TTS, ASR and the narration commands
cargo build --release --features audio
```

### Generate Text

```bash
blazr generate \
  --model meta-llama/Llama-3.2-1B \
  --prompt "Once upon a time" \
  --max-tokens 100

# Interactive REPL, with an optional first prompt
blazr run meta-llama/Llama-3.2-1B
```

### Start Server

`blazr serve` does not fetch models. Run `blazr pull` first, then serve the pulled name. `blazr list` shows the names.

```bash
blazr pull Qwen/Qwen2.5-0.5B
blazr serve --model <name> --port 8080
```

```bash
# Text completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

### Other Commands

```bash
blazr info ./models/mistral-7b             # Show model architecture and config
blazr list                                  # List available local models
blazr pull Qwen/Qwen2.5-0.5B               # Download from HuggingFace
```

## Supported Models

[docs/models.md](docs/models.md) holds the full matrix with an evidence level per cell.

| Family                                                                        | Evidence                                                   |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Llama 3.x                                                                     | SafeTensors tested                                         |
| Mistral 7B                                                                    | SafeTensors, AWQ, GPTQ, GGUF tested (production checklist) |
| Qwen3.5 hybrid                                                                | GGUF tested                                                |
| Mamba2, Mamba3, DeepSeek-V2/V3, Mamba+attention hybrids                       | Supported in code, no test names a checkpoint              |
| Qwen2/2.5, Phi, Gemma, StarCoder2, Mixtral, Falcon, Command-R, GPT-NeoX, DBRX | Routed to the Llama path, untested                         |

## Model Formats

blazr auto-detects the format from checkpoint contents:

### SafeTensors (HuggingFace)

```
model_dir/
├── model.safetensors          # or sharded: model-00001-of-00002.safetensors
├── config.json                # HuggingFace model config
└── tokenizer_config.json      # Chat template (optional)
```

### SafeTensors (oxidizr)

```
checkpoint_dir/
├── model.safetensors
└── config.json                # oxidizr config (optional, inferred from tensors)
```

### GGUF

```
model.gguf                     # Single file: weights + tokenizer + config
```

GGUF quantization formats run from Q2_K through Q8_0, plus the IQ and TQ series. [docs/models.md](docs/models.md) lists the CPU and CUDA kernel for each format.

## API Reference

### Endpoints

| Method | Path                       | Description                                    |
| ------ | -------------------------- | ---------------------------------------------- |
| GET    | `/health`                  | Health check                                   |
| GET    | `/v1/models`               | List loaded models                             |
| POST   | `/v1/completions`          | Text completion                                |
| POST   | `/v1/chat/completions`     | Chat completion                                |
| POST   | `/v1/embeddings`           | Embeddings                                     |
| POST   | `/v1/responses`            | OpenAI Responses API                           |
| POST   | `/v1/messages`             | Anthropic Messages API                         |
| POST   | `/v1/infill`               | Fill-in-the-middle                             |
| POST   | `/v1/rerank`               | Rerank documents                               |
| POST   | `/tokenize`, `/detokenize` | Tokenizer round trip                           |
| POST   | `/v1/audio/speech`         | Text to speech (`audio`)                       |
| POST   | `/v1/audio/transcriptions` | Speech to text, with word timestamps (`audio`) |
| POST   | `/v1/audio/translations`   | Speech to English text (`audio`)               |
| GET    | `/v1/audio/voices`         | A TTS model's voices (`audio`)                 |
| GET    | `/api/tags`, `/api/ps`     | Ollama-compatible model listing                |
| GET    | `/metrics`                 | Prometheus metrics                             |

### Request Parameters

```json
{
  "prompt": "text",
  "messages": [{ "role": "user", "content": "text" }],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop": ["\n\n"],
  "stream": false
}
```

### Response Format

Non-streaming responses follow the OpenAI format:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "choices": [
    {
      "text": "generated text",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  }
}
```

Streaming responses use SSE with `data: {...}` chunks and `data: [DONE]` sentinel.

## CLI Reference

```
blazr run           Interactive generation (REPL)
blazr generate      Generate text from a prompt
blazr chat          Interactive multi-turn chat
blazr serve         Start the inference server
blazr info          Display model architecture and configuration
blazr list          List available local models
blazr pull          Download a model from HuggingFace Hub
blazr bench         Measure prefill and decode throughput
blazr ps            Show the models a running server holds
blazr convert       Convert a model through compressr
blazr completions   Print shell completions
blazr speak         Narration from a script (audio)
blazr transcribe    Word timings from a Whisper model (audio)
blazr sound-review  Review rendered sound effects (audio)
```

### Generation Options

| Flag            | Default  | Description                                                       |
| --------------- | -------- | ----------------------------------------------------------------- |
| `--model`       | `default_model` | Model name or path                                             |
| `--prompt`      | none     | Input text. Required for `generate`.                              |
| `--max-tokens`  | 2048     | Maximum tokens to generate                                        |
| `--temperature` | 1.0      | Sampling temperature (0 = greedy)                                 |
| `--top-p`       | 1.0      | Nucleus sampling threshold                                        |
| `--top-k`       | 0        | Top-k sampling (0 = off)                                          |
| `--cpu`         | `cuda:0` | Force CPU inference. Default device: `device` config key, else `cuda:0` |
| `--mmproj`      | none     | Qwen3-VL mmproj GGUF for image inputs (`qwen35` models only)      |
| `--image`       | none     | Image file placed before the prompt. Needs `--mmproj`. Repeatable. `generate` only. |

### Server Options

| Flag                                        | Default            | Description                                                                                                    |
| ------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| `--model`                                   | none               | Local path or name under `$BLAZR_MODEL_DIR` (default `./models`); without it, models load on demand            |
| `--port`                                    | 8080               | Port to listen on                                                                                              |
| `--host`                                    | 0.0.0.0            | Host to bind to                                                                                                |
| `--data-parallel`                           | 1                  | Copies of the start model, one per CUDA device or CPU copy. Above 1 needs `--model` and tensor parallelism at 1. |
| `--chat-template`                           | model's own        | Built-in name (`chatml`, `llama3`, `mistral`, `phi3`, `gemma`, `deepseek`, `generic`) or a Jinja template file |
| `--mmproj`                                  | none               | Qwen3-VL mmproj GGUF loaded beside a `qwen35` model; enables `image_url` parts in `/v1/chat/completions`       |
| `--image-min-tokens` / `--image-max-tokens` | mmproj values      | Token range every image is resized into                                                                        |
| `--tls-cert` / `--tls-key`                  | none               | PEM certificate chain and private key; both together serve HTTPS                                               |
| `--api-key` / `--api-key-file`              | none               | Require a bearer key; also reads `BLAZR_API_KEY`                                                               |
| `--asr-model`                               | none               | `NAME=DIR` Whisper model for `/v1/audio/transcriptions` (`audio`); repeatable                                  |
| `--tts-model`                               | none               | `NAME=DIR` Kokoro, VoxCPM2 or VITS model for `/v1/audio/speech` (`audio`); repeatable                          |
| `--voice-dir`                               | `$BLAZR_VOICE_DIR` | Kokoro voice files searched before the model's `voices/` (`audio`)                                             |

## Image Inputs

With `--mmproj`, three routes take images beside the text. Each image takes
the place of its own part in the prompt; the reply describes it. Without
`--mmproj` such a request gets a 400 naming the flag, in the route's own
error shape. `blazr chat` attaches an image to the next turn with
`/image <path>`.

| Route                  | Image part                                                     |
| ---------------------- | -------------------------------------------------------------- |
| `/v1/chat/completions` | `image_url` part: data URI, `http(s)` URL, or base64           |
| `/v1/responses`        | `input_image` part: `image_url` as a data URL or `http(s)` URL |
| `/v1/messages`         | `image` block: a `base64` source, or a `url` source            |

## Chat Templates

`/v1/chat/completions`, `blazr chat` and `/apply-template` render messages
with the model's own Jinja chat template: GGUF `tokenizer.chat_template`, or
`chat_template` in `tokenizer_config.json`. A model that ships none gets the
built-in format for its model type. `--chat-template` overrides both with a
built-in name or a template file. `blazr info` and `/api/show` report which
one is in use (`gguf jinja`, `tokenizer_config jinja`, `builtin chatml`).

A Jinja template receives `messages`, `tools`, `add_generation_prompt`,
`bos_token`, `eos_token`, and per-request variables: `think` sets
`enable_thinking`, `reasoning_effort` passes through, and
`chat_template_kwargs` adds any other variable. A template that refuses the
conversation (`raise_exception`) answers the request with HTTP 400 and the
template's message.

## Audio

Build with `--features audio` for speech, or take a prebuilt binary: every
release archive includes it. The feature adds the `/v1/audio/*` routes to
`blazr serve` and three commands. The commands keep standard output for
their report and log to standard error.

### Commands

`blazr speak` turns a script into a voiceover WAV. It hears every phrase back
with Whisper and reports any phrase that does not match the script.

```bash
blazr speak script.txt --out assets/voiceover.wav
blazr speak "Every phrase is heard back." --engine kokoro --voice af_heart --json
```

`blazr transcribe` writes word timings from a local Whisper model: one words
file per narration.

```bash
blazr transcribe assets/voiceover.wav --script script.txt --json
```

`blazr sound-review` reviews a directory of rendered sound effects: sharpness,
hiss, tail and true peak, then a CLAP ranking of each base variant against
its text. `--no-model` skips the ranking.

```bash
blazr sound-review renders/sounds
```

Each command prints its full flag list with `--help`.

### Exit codes

| Code | Meaning                                                                       |
| ---- | ----------------------------------------------------------------------------- |
| 0    | Done: every phrase heard as written, every narration read, every sound passed |
| 1    | A problem remains, or the run failed                                          |
| 2    | A bad input: a missing file, a flag out of range, a bad manifest              |
| 3    | `speak --model` names a model blazr does not run; run it another way          |

### Models on first use

A model the commands need and do not find on disk downloads from the
Hugging Face hub on first use, once:

| Model                                   | Used by                                      |
| --------------------------------------- | -------------------------------------------- |
| `openai/whisper-base`                   | `transcribe` and `speak` for English         |
| `openai/whisper-small`                  | `transcribe` and `speak` for other languages |
| `hexgrad/Kokoro-82M`, `fs90/kokoro-g2p` | `speak` with Kokoro                          |
| `openbmb/VoxCPM2`                       | `speak` with VoxCPM2 (CUDA build with a GPU) |
| `laion/larger_clap_general`             | `sound-review`                               |

Downloads land in hub layout in the model folder: `$MOTION_VIDEO_MODELS`,
else the `models` entry of `~/.config/motion-video/config.json`, else
`~/.cache/motion-video/models`. The search also reads `$BOOSTR_MODELS_DIR`
and the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`~/.cache/huggingface/hub`). `blazr serve` fetches only Kokoro's G2P models
on first use, into the Hugging Face cache. Set `HF_HUB_OFFLINE=1` to forbid
every download: a missing model then fails with its name and the
`hf download` command that fetches it.

### CPU and CUDA builds

On the CUDA build (`--features audio,cuda`) with an NVIDIA GPU, `speak` runs
Kokoro, VITS, VoxCPM2 and Whisper on the first GPU. `transcribe` runs
Whisper there, and `sound-review` runs CLAP there. `speak --device cpu` keeps
every model of the run on the CPU. Every other build runs them all on the
CPU. A Kokoro or VITS take follows `--seed`: one seed gives the same take on
one device, and a different take on the CPU and the GPU.

`blazr serve` loads every `--asr-model` and `--tts-model` on the server's
device: the configured CUDA device on the CUDA build, else the CPU.

`speak` picks VoxCPM2 by default only on the CUDA build with an NVIDIA GPU
with 8 GB of memory. Elsewhere it picks Kokoro. `--engine voxcpm --device
cpu` runs VoxCPM2 on the CPU, at a small fraction of GPU speed.

### The motion-video skill

The [motion-video skill](https://github.com/farhan-syah/motion-video-skill)
calls `blazr speak`, `blazr transcribe` and `blazr sound-review` with its own
flags. Flags the commands do not read are accepted and ignored. When blazr
is missing or exits 3, the skill falls back to its own path.

## Tokenizer

Uses [splintr](https://github.com/ml-rust/splintr) for BPE tokenization. The vocabulary is detected from the model.

| Vocabulary    | Models                         | Vocab Size |
| ------------- | ------------------------------ | ---------- |
| `llama3`      | Llama 3.x                      | ~128k      |
| `mistral_v1`  | Mistral 7B v0.1, v0.2, Mixtral | 32k        |
| `mistral_v2`  | Mistral 7B v0.3, Codestral     | ~33k       |
| `mistral_v3`  | Mistral NeMo, Large 2, Pixtral | ~131k      |
| `cl100k_base` | GPT-4, GPT-3.5-turbo           | ~100k      |
| `o200k_base`  | GPT-4o                         | ~200k      |
| `deepseek_v3` | DeepSeek V3/R1                 | ~129k      |

GGUF files include an embedded tokenizer which is extracted automatically.

## Documentation

| Doc                                            | Content                                               |
| ---------------------------------------------- | ----------------------------------------------------- |
| [docs/api.md](docs/api.md)                     | Every HTTP endpoint, request field, and error         |
| [docs/configuration.md](docs/configuration.md) | Config file, CLI flags, environment variables         |
| [docs/deployment.md](docs/deployment.md)       | Install, systemd, Docker, reverse proxy, TLS, metrics |
| [docs/models.md](docs/models.md)               | Architecture, format, and quantization matrix         |
| [docs/tuning.md](docs/tuning.md)               | Batching, caching, parallelism knobs                  |
| [docs/architecture.md](docs/architecture.md)   | Layers, request path, modules                         |

## Architecture

blazr is a thin application layer built on [boostr](https://github.com/ml-rust/boostr) (ML framework) and [numr](https://github.com/ml-rust/numr) (numerical computing). All model architectures, tensor operations, and quantization kernels live in boostr. blazr provides the CLI, HTTP server, model loading, and request orchestration.

```
blazr (CLI + HTTP server + model lifecycle)
  |
boostr (model architectures + quant kernels + NN modules)
  |
numr (tensors + linalg + multi-backend: CPU/CUDA/WebGPU)
```

## Requirements

- Rust 1.95+ to build from source
- (Optional) CUDA 12.8+ or 13.x for GPU acceleration

## License

Apache-2.0 - see [LICENSE](LICENSE) for details.

Built with Llama. blazr embeds the Llama 3 tokenizer vocabulary through
[splintr](https://github.com/ml-rust/splintr). That vocabulary is under the
Llama 3.1 Community License: Llama 3.1 is licensed under the Llama 3.1
Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
Each release archive carries that licence as `LICENSE-LLAMA`, and the licence
of every compiled-in crate in `THIRD-PARTY-LICENSES.txt`.

## Related Projects

- [boostr](https://github.com/ml-rust/boostr) - ML framework (model architectures, quantization)
- [numr](https://github.com/ml-rust/numr) - Foundational numerical computing (tensors, linalg, multi-backend)
- [oxidizr](https://github.com/ml-rust/oxidizr) - Training framework for hybrid architectures
- [splintr](https://github.com/ml-rust/splintr) - High-performance BPE tokenizer
- [compressr](https://github.com/ml-rust/compressr) - Model conversion and compression

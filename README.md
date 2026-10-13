<div align="center">

# blazr

<h3>Inference server for LLMs, speech, and embeddings.</h3>

<p>
Serves OpenAI-compatible routes for SafeTensors, AWQ, GPTQ, and GGUF models.
</p>

<p>
  <a href="https://docs.rs/blazr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/blazr"><strong>Crate</strong></a>
  ·
  <a href="#supported-models"><strong>Models</strong></a>
  ·
  <a href="#api-reference"><strong>API</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/blazr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/blazr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/blazr">
    <img src="https://img.shields.io/crates/v/blazr" alt="crates.io">
  </a>
  <a href="https://docs.rs/blazr">
    <img src="https://img.shields.io/docsrs/blazr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/blazr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/blazr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/blazr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

blazr is one binary that runs text, speech, and embedding models. It loads checkpoints from SafeTensors, AWQ, GPTQ, and GGUF, and serves them over HTTP or the command line. Model architectures and quantization kernels come from [boostr](https://github.com/ml-rust/boostr). blazr supplies the CLI, the HTTP server, model loading, and request scheduling. See [docs/architecture.md](docs/architecture.md).

## Features

- **Architectures**: Llama, Mistral, Mamba2, MLA+MoE, hybrid models, and more
- **Formats**: SafeTensors (F16/BF16), AWQ INT4, GPTQ INT4, and GGUF with the quantization formats in [docs/models.md](docs/models.md#gguf-quantization-formats)
- **Auto-detection**: architecture, format, and tokenizer from checkpoint tensor names
- **OpenAI-compatible API**: `/v1/completions`, `/v1/chat/completions`, and `/v1/embeddings`
- **Streaming**: Server-Sent Events (SSE) for token-by-token output
- **Hugging Face Hub**: `blazr pull` downloads models
- **Operations**: API-key auth, native TLS, Prometheus `/metrics`, request timeout, graceful shutdown on SIGTERM
- **GPU**: CUDA (`cuda`) with quantization kernels, or WebGPU (`wgpu`) on Metal, Vulkan, and DX12
- **Embedding**: C ABI (`capi`) with a static and a shared library
- **Speech**: TTS (Kokoro, VoxCPM2, VITS) and Whisper transcription with word timestamps, behind the `audio` feature
- **Tokenizer**: [splintr](https://github.com/ml-rust/splintr) BPE, detected from the model. GGUF files carry an embedded tokenizer that blazr extracts. See [docs/models.md](docs/models.md#tokenizers).

## Quick Start

### Install

#### Prebuilt binaries

Each [GitHub release](https://github.com/ml-rust/blazr/releases) carries archives for Linux (CPU and Vulkan), macOS, Windows (CPU and DX12), Android, and iOS. The desktop archives include the `audio` feature. [docs/deployment.md](docs/deployment.md#prebuilt-archives) lists every archive.

The names carry no version, so the latest release keeps the same URL:

```bash
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/blazr-linux-x64.tar.gz
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar xzf blazr-linux-x64.tar.gz
./blazr --help
```

`SHA256SUMS` lists the SHA-256 of every archive. On macOS, use `shasum -a 256 --check --ignore-missing SHA256SUMS`.

The Linux x86_64 archives need glibc 2.35 or newer. The aarch64 archive needs glibc 2.39 or newer. No archive includes CUDA. Build with `--features audio,cuda` for NVIDIA GPU inference. The build needs the CUDA toolkit, 12.8 or newer. See [docs/deployment.md](docs/deployment.md) for install details.

#### From crates.io

```bash
cargo install blazr --locked --features audio          # CPU
cargo install blazr --locked --features audio,cuda     # NVIDIA CUDA
```

#### Build variants

| Build                                  | Cargo command                                                                             | Runtime                                                                                  |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| CPU (default)                          | `cargo build --release`                                                                   | CPU                                                                                      |
| Speech                                 | `cargo build --release --features audio`                                                  | CPU                                                                                      |
| CUDA                                   | `cargo build --release --features cuda`                                                   | CUDA device from `device`, CPU with `--cpu`                                              |
| WebGPU                                 | `cargo build --release --features wgpu`                                                   | wgpu adapter from `device`, CPU with `--cpu`                                             |
| Engine only (library)                  | `cargo check --lib --no-default-features`                                                 | Library with no CLI or HTTP stack                                                        |
| C library (`capi`)                     | `cargo build --release --lib --no-default-features --features capi,f16`                   | `libblazr.a` and `libblazr.so`                                                           |
| Android (`aarch64-linux-android`)      | `cargo check --lib --no-default-features --features capi --target aarch64-linux-android`  | Compile check passes on NDK 27. x86_64 emulator run passes. Not run on aarch64 hardware. |
| WebAssembly (`wasm32-unknown-unknown`) | `cargo check --lib --no-default-features --features wgpu --target wasm32-unknown-unknown` | Compile check passes. Not run in a browser.                                              |

- `cuda` takes priority over `wgpu` when both features are on.
- `cli` and `server` each enable `hub`, the Hugging Face client.
- `info`, `list`, `pull`, `run`, `chat`, `bench`, and `convert` need `cli`. `serve` and `ps` need `server`. `swarm` and `disagg` need `distributed` and `server`.
- Tensor parallelism needs `--features cuda,nccl`. A degree above 1 fails in any other build. It is tested on one GPU only. See [docs/tuning.md](docs/tuning.md#parallelism).

### Generate text

```bash
blazr generate \
  --model meta-llama/Llama-3.2-1B \
  --prompt "Once upon a time" \
  --max-tokens 100

# Interactive REPL
blazr run meta-llama/Llama-3.2-1B
```

### Start the server

`blazr serve` fetches only Kokoro's G2P files on first use. Run `blazr pull` first for LLM weights, then serve the pulled name. `blazr list` shows the names.

```bash
blazr pull Qwen/Qwen2.5-0.5B
blazr serve --model <name> --port 8080
```

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

Without `--model`, the server loads `default_model` when that key is set. Otherwise it starts with no model. Requests load the named model on first use. See [docs/deployment.md](docs/deployment.md#start-on-demand).

## Supported Models

[docs/models.md](docs/models.md) holds the full matrix, with an evidence level per cell.

| Family                                                                       | Evidence                                                              |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Llama 3.x                                                                    | SafeTensors tested against `transformers`                             |
| Mistral 7B                                                                   | SafeTensors, AWQ, GPTQ, GGUF tested                                   |
| Qwen2/2.5, Phi-3.5, Gemma 2, StarCoder2, GPT-NeoX/Pythia, Falcon-RW, Mixtral | SafeTensors tested against `transformers`: 32 of 32 greedy ids match  |
| Qwen2-MoE, DBRX, Command-R (Cohere)                                          | SafeTensors tested against `transformers` on tiny random-weight files |
| Qwen3.5 hybrid                                                               | GGUF tested                                                           |
| Mamba2, Mamba3, DeepSeek-V2/V3, Mamba+attention hybrids                      | Supported in code, no test names a checkpoint                         |
| AWQ, GPTQ and GGUF files of those families, Phi-2                            | Routed to the Llama path, untested                                    |
| InternLM2                                                                    | Routed to the Llama path. No tensor name map, so the load fails       |

## Model Formats

blazr detects the format from checkpoint contents.

```
model_dir/                         # SafeTensors (HuggingFace)
├── model.safetensors              # or sharded: model-00001-of-00002.safetensors
├── config.json                    # HuggingFace model config
└── tokenizer_config.json          # Chat template (optional)

checkpoint_dir/                    # SafeTensors (oxidizr)
├── model.safetensors
└── config.json                    # oxidizr config (optional, inferred from tensors)

model.gguf                         # GGUF: weights, tokenizer, and config in one file
```

GGUF quantization formats run from Q2_K through Q8_0, plus the IQ and TQ series. [docs/models.md](docs/models.md#gguf-quantization-formats) lists the CPU and CUDA kernel for each format.

GGUF files load through mmap on every target except `wasm32`. A failed map falls back to file reads.

## API Reference

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

[docs/api.md](docs/api.md) lists every route, with its request fields, responses, and errors. Streaming responses send `data: {...}` chunks and end with `data: [DONE]`.

## CLI Reference

| Command          | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `blazr run`      | Interactive generation (REPL)                        |
| `blazr generate` | Generate text from a prompt                          |
| `blazr chat`     | Interactive multi-turn chat                          |
| `blazr serve`    | Start the inference server                           |
| `blazr info`     | Display model architecture and config                |
| `blazr list`     | List available local models                          |
| `blazr pull`     | Download a model from HuggingFace Hub                |
| `blazr bench`    | Measure prefill and decode throughput                |
| `blazr ps`       | Show the models a running server holds               |
| `blazr convert`  | Convert a model through compressr                    |
| `blazr completions` | Print shell completions                           |
| `blazr speak`    | Narration from a script (`audio`)                    |
| `blazr transcribe` | Word timings from a Whisper model (`audio`)        |
| `blazr sound-review` | Review rendered sound effects (`audio`)          |
| `blazr swarm`    | Join or start a distributed inference swarm (`distributed`) |
| `blazr disagg`   | Run prefill and decode disaggregation (`distributed`) |

Run `blazr <command> --help` for every flag. [docs/configuration.md](docs/configuration.md) documents the flags, the config file, and the environment variables.

The `device` key accepts `cpu`, `cuda`, `cuda:N`, `wgpu`, and `wgpu:N`. Set it in the config file. See [docs/configuration.md](docs/configuration.md).

## Chat Templates and Images

Chat routes render messages with the model's own Jinja template. `--chat-template` replaces it with a built-in name (`chatml`, `llama3`, `mistral`, `phi3`, `gemma`, `deepseek`, `generic`) or a template file. See [docs/models.md](docs/models.md#chat-templates).

With `--mmproj`, the chat, responses, and messages routes accept images beside text. Without it, an image request gets a 400 that names the flag. See [docs/api.md](docs/api.md#image-and-audio-input).

## Speech

Build with `--features audio`, or use a desktop archive. The feature adds `/v1/audio/*` routes to `blazr serve` and three commands. The commands write their report to standard output and their logs to standard error.

| Command              | Purpose                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `blazr speak`        | Turns a script into a voiceover WAV. Whisper hears each phrase back and reports any mismatch. |
| `blazr transcribe`   | Writes word timings from a local Whisper model, one words file per narration.                 |
| `blazr sound-review` | Reviews rendered sound effects, then ranks each against its text with CLAP. `--no-model` skips the ranking. |

| Exit code | Meaning                                                                        |
| --------- | ------------------------------------------------------------------------------ |
| 0         | Done: every phrase heard as written, every narration read, every sound passed |
| 1         | A mismatch remains, or the run failed                                          |
| 2         | A bad input: a missing file, a flag out of range, a bad manifest               |
| 3         | `speak --model` names a model blazr does not run. Run it another way           |

A command fetches a missing model from the Hugging Face hub on first use. The model list, the search path, and the device rules are in [docs/models.md](docs/models.md#speech). The `motion-video` skill calls these commands. When blazr is missing or exits 3, the skill uses its own path.

## Embedding

The `capi` feature builds a C library for app embedders. Run `cargo build --release --lib --no-default-features --features capi,f16`. It writes `libblazr.a` and `libblazr.so` to `target/release/`. The header is [`include/blazr.h`](include/blazr.h).

| Function                                                      | Purpose                                                                 |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `blazr_version`                                               | Returns the library version. Do not free it.                            |
| `blazr_load_options_default`                                  | Returns the default `BlazrLoadOptions`.                                 |
| `blazr_model_load`                                            | Loads a GGUF file or a SafeTensors directory.                           |
| `blazr_model_unload`                                          | Unloads a model. A second unload returns `BLAZR_STATUS_INVALID_HANDLE`. |
| `blazr_generate_params_default`                               | Returns the default sampling parameters.                                |
| `blazr_generate`                                              | Generates text and calls the callback once per token.                   |
| `blazr_cancel`                                                | Stops the running generation from any thread.                           |
| `blazr_tokenize`                                              | Encodes text to token ids.                                              |
| `blazr_embed`                                                 | Returns one unit-length vector: the mean of the last hidden states.     |
| `blazr_last_error`                                            | Returns the message of the last failed call on this thread.             |
| `blazr_string_free`, `blazr_tokens_free`, `blazr_floats_free` | Free the strings, token arrays and float arrays blazr returns.          |

Every call returns a `BlazrStatus`. Zero and above mean success.

blazr ships no Kotlin or Swift bindings. An app calls the C functions through JNI on Android or a bridging header on iOS. [examples/capi/README.md](examples/capi/README.md) lists the link steps, and [`examples/capi/smoke.c`](examples/capi/smoke.c) calls every step.

### Android and iOS

The Android check needs NDK 27 variables. [examples/capi/README.md](examples/capi/README.md#android) lists them and the emulator steps. The x86_64 emulator run needs `-Wl,-z,max-page-size=16384` at link. The CI iOS job checks `aarch64-apple-ios` and `aarch64-apple-ios-sim` on macOS.

## Decision models

blazr serves System One decision models on `POST /v1/systemone`. A decision model answers typed questions about one state. It returns probabilities and does not generate text.

| Family     | Model                                                    | Weights                                             |
| ---------- | -------------------------------------------------------- | --------------------------------------------------- |
| `jebadiah` | Jebadiah                                                 | SafeTensors or GGUF                                 |
| `decider`  | Mapika decider                                           | SafeTensors or GGUF                                 |
| `kev`      | Kev: LoRA adapter and pointer head, served on its base   | Adapter directory, plus a base model                |
| `clef`     | Clef-Flash: full checkpoint and joint-schema head, text only | SafeTensors checkpoint with the head beside it  |

```bash
BLAZR_MODEL_DIR=/path/to/models blazr serve --model decider-4b --port 8080
```

Clef-Flash's text backbone takes 16.7 GiB in BF16. It does not fit a 12 GB GPU. Kev and Clef are not tested on CUDA. Clef 27B is not tested and not claimed. Other families are not served.

See [docs/decision-models.md](docs/decision-models.md) for run commands for each family, family detection, calibration, limits, and the tested matrix.

## Requirements

- Rust 1.95+ to build from source
- (Optional) CUDA 12.8+ or 13.x for the `cuda` feature
- (Optional) NCCL with `cuda` for tensor parallelism
- (Optional) A Metal, Vulkan, or DX12 driver for the `wgpu` feature

## Documentation

| Doc                                            | Content                                               |
| ---------------------------------------------- | ----------------------------------------------------- |
| [docs/api.md](docs/api.md)                     | Every HTTP endpoint, request field, and error         |
| [docs/configuration.md](docs/configuration.md) | Config file, CLI flags, environment variables         |
| [docs/deployment.md](docs/deployment.md)       | Install, systemd, Docker, reverse proxy, TLS, metrics |
| [docs/models.md](docs/models.md)               | Architecture, format, and quantization matrix         |
| [docs/decision-models.md](docs/decision-models.md) | System One decision models: families, run, limits |
| [docs/tuning.md](docs/tuning.md)               | Batching, caching, parallelism knobs                  |
| [docs/architecture.md](docs/architecture.md)   | Layers, request path, modules                         |

## Related Projects

- [boostr](https://github.com/ml-rust/boostr) - ML framework (model architectures, quantization)
- [numr](https://github.com/ml-rust/numr) - Foundational numerical computing (tensors, linalg, multi-backend)
- [oxidizr](https://github.com/ml-rust/oxidizr) - Training framework for hybrid architectures
- [splintr](https://github.com/ml-rust/splintr) - BPE tokenizer
- [compressr](https://github.com/ml-rust/compressr) - Model conversion and compression

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).

Built with Llama. blazr embeds the Llama 3 tokenizer vocabulary through [splintr](https://github.com/ml-rust/splintr). That vocabulary is under the Llama 3.1 Community License. Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Each release archive carries that licence as `LICENSE-LLAMA`, and the licence of every compiled-in crate in `THIRD-PARTY-LICENSES.txt`.

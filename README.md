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
  <a href="#what-works"><strong>Hardware</strong></a>
  ·
  <a href="#supported-models"><strong>Models</strong></a>
  ·
  <a href="#api-reference"><strong>API</strong></a>
  ·
  <a href="#quick-start"><strong>Quick Start</strong></a>
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

blazr is one binary that runs text, speech, and embedding models. It serves them over HTTP or the command line. Model architectures and quantization kernels come from [boostr](https://github.com/ml-rust/boostr). See [docs/architecture.md](docs/architecture.md).

## What works

| Hardware | Get it from | Needs |
| --- | --- | --- |
| CPU | The Linux x86_64, Linux aarch64, or Windows x86_64 CPU archive, or the default build | Nothing |
| NVIDIA through CUDA | Source build with `--features cuda`. No archive includes it | CUDA toolkit 12.8 or newer (13.x works), `nvcc` on `PATH`, compute capability 7.5 (Turing) or newer |
| Linux GPU through Vulkan | `blazr-linux-x64-gpu.tar.gz`, or `--features wgpu` | Vulkan driver |
| Windows GPU through DX12 | `blazr-windows-x64-gpu.zip`, or `--features wgpu` | DX12 driver |
| Apple silicon through Metal | `blazr-macos-arm64.tar.gz`, or `--features wgpu` | macOS |

CI runs no GPU test. It compile-checks `wgpu` and does not build `cuda`.

| Capability | What you get |
| --- | --- |
| Formats | SafeTensors (F16, BF16), AWQ INT4, GPTQ INT4, and [GGUF quantizations](docs/models.md#gguf-quantization-formats) |
| Models | See [Supported Models](#supported-models) |
| Detection | Architecture, format, and tokenizer from checkpoint tensor names |
| HTTP APIs | OpenAI-compatible, Anthropic Messages, and Ollama routes. See [API Reference](#api-reference) |
| Streaming | Server-Sent Events for token-by-token output |
| Downloads | `blazr pull` from the Hugging Face Hub |
| Operations | API-key auth, native TLS, Prometheus `/metrics`, request timeout, graceful shutdown on SIGTERM |
| Speech | Kokoro, VoxCPM2, and VITS synthesis, and Whisper with word timestamps. Needs `audio` |
| C library | Static and shared library. Needs `capi` |
| Tokenizer | [splintr](https://github.com/ml-rust/splintr) BPE, detected from the model. blazr extracts a GGUF file's embedded tokenizer. See [docs/models.md](docs/models.md#tokenizers) |

## Quick Start

### Install a prebuilt archive

| Archive | Platform | GPU | Needs |
| --- | --- | --- | --- |
| `blazr-linux-x64.tar.gz` | Linux x86_64 | None | glibc 2.35 or newer |
| `blazr-linux-x64-gpu.tar.gz` | Linux x86_64 | Vulkan | glibc 2.35 or newer |
| `blazr-linux-arm64.tar.gz` | Linux aarch64 | None | glibc 2.39 or newer |
| `blazr-macos-arm64.tar.gz` | macOS on Apple silicon | Metal | Nothing |
| `blazr-windows-x64.zip` | Windows x86_64 | None | Nothing |
| `blazr-windows-x64-gpu.zip` | Windows x86_64 | DX12 | Nothing |

- Every archive above includes the `audio` feature.
- Android and iOS archives hold `libblazr.a`. See [C library](#c-library).
- Archive names carry no version, so the latest release keeps the same URL.
- [docs/deployment.md](docs/deployment.md#prebuilt-archives) lists every archive.

```bash
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/blazr-linux-x64.tar.gz
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar xzf blazr-linux-x64.tar.gz
./blazr --help
```

On macOS, use `shasum -a 256 --check --ignore-missing SHA256SUMS`. [docs/deployment.md](docs/deployment.md#prebuilt-archives) has the Windows steps.

### Install with cargo

Needs Rust 1.95 or newer.

```bash
cargo install blazr --locked --features audio          # CPU
cargo install blazr --locked --features audio,cuda     # NVIDIA CUDA
```

### Run a model

`blazr run`, `generate`, `chat`, and `bench` take a model path. They never fetch a model. `blazr pull` downloads one first.

```bash
blazr pull Qwen/Qwen2.5-0.5B      # writes ./models/Qwen2.5-0.5B
blazr generate --model ./models/Qwen2.5-0.5B --prompt "Once upon a time" --max-tokens 100
blazr run ./models/Qwen2.5-0.5B   # interactive REPL
```

`blazr pull` writes to `--output`, then `BLAZR_MODEL_DIR`, then the `model_dir` config key, then `./models`.

### Start the server

`blazr serve --model` takes a path, or a name that `blazr list` shows.

```bash
blazr serve --model Qwen2.5-0.5B --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

- `blazr serve` does not download LLM weights. Run `blazr pull` first.
- `blazr serve` fetches only Kokoro's G2P files, on first use.
- Without `--model`, the server loads the `default_model` key when set. Otherwise it starts with no model.
- Requests load the named model on first use. See [docs/deployment.md](docs/deployment.md#start-on-demand).

### Build from source

| Build | Command |
| --- | --- |
| CPU | `cargo build --release` |
| CPU with speech | `cargo build --release --features audio` |
| NVIDIA CUDA | `cargo build --release --features cuda` |
| WebGPU | `cargo build --release --features wgpu` |
| Multi-node | `cargo build --release --features distributed` |
| Multi-GPU tensor parallelism | `cargo build --release --features cuda,nccl` |

- Backends: `cuda` takes priority over `wgpu` when both are on.
- Devices: the `device` config key takes `cpu`, `cuda`, `cuda:N`, `wgpu`, and `wgpu:N`. `--cpu` forces the CPU where a command offers it.
- CUDA binaries: set `NUMR_CUDA_ARCH=all` for a binary that runs on another machine.
- Tensor parallelism: needs NCCL. A degree above 1 fails in any other build. It is tested on one GPU only. See [docs/tuning.md](docs/tuning.md#parallelism).
- Library use: `--no-default-features` builds the engine with no CLI or HTTP stack. The `blazr` binary needs the default `cli` and `server`.
- Commands by feature: [docs/architecture.md](docs/architecture.md#commands-by-feature).

## Supported Models

| Family | Status | Tested formats | Evidence |
| --- | --- | --- | --- |
| Llama 3.x | Tested | SafeTensors | Matches `transformers` |
| Mistral 7B | Tested | SafeTensors, AWQ, GPTQ, GGUF | Production checklist names it. CPU runs of GGUF, AWQ, and GPTQ give coherent text. No reference |
| Qwen2/2.5, Phi-3.5, Gemma 2, StarCoder2, GPT-NeoX/Pythia, Falcon-RW, Mixtral | Tested | SafeTensors | 32 of 32 greedy ids match `transformers` |
| Qwen2-MoE, DBRX, Command-R (Cohere) | Tested on tiny random-weight files only | SafeTensors | 32 of 32 ids and the logits match `transformers` |
| Qwen3.5 hybrid | Tested | GGUF | A repo test generates from a GGUF file |
| oxidizr MLA, MLA+MoE, Mamba2 + MLA+MoE hybrids | Tested on tiny seeded checkpoints only | SafeTensors | Generates through the loader |
| Mamba2, Mamba3 | In code, untested | None | No test names a checkpoint |
| Llama AWQ, GPTQ, and GGUF files, and Yi, CodeLlama, Solar | In code, untested | None | No test names a checkpoint |
| Phi-2, and AWQ, GPTQ, or GGUF files of the other families above | Routes to the Llama path, untested | None | No test covers the route |
| DeepSeek-V2/V3 | Refused at load | None | No DeepSeek model implementation |
| InternLM2 | Load fails | None | No tensor name map |

The Evidence column covers the forward pass on a raw prompt. It says nothing about chat. `tests/chat_stop_reference.rs` checks chat prompt ids and stop ids against `transformers` for Qwen2.5, Qwen3, Gemma 2, and Phi-3.5, and one real Qwen3 turn that ends on `<|im_end|>`.

A base checkpoint is not a chat model. `Qwen2.5-0.5B` and `gemma-2-2b` never learned their turn markers. Through `/v1/chat/completions` they run past the first sentence and print control text where a marker belongs, and `transformers` prints the same kind of control text from the same prompt. Use an `-Instruct`, `-it`, or chat checkpoint for chat.

[docs/models.md](docs/models.md) holds the full matrix, with an evidence level per cell.

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

This table lists the main routes. [docs/api.md](docs/api.md) lists every route, with its request fields, responses, and errors.

| Method | Path                       | Description                                    |
| ------ | -------------------------- | ---------------------------------------------- |
| GET    | `/health`                  | Health check                                   |
| GET    | `/v1/models`               | List available models                          |
| POST   | `/v1/completions`          | Text completion                                |
| POST   | `/v1/chat/completions`     | Chat completion                                |
| POST   | `/v1/embeddings`           | Embeddings                                     |
| POST   | `/v1/responses`            | OpenAI Responses API                           |
| POST   | `/v1/messages`             | Anthropic Messages API                         |
| POST   | `/v1/infill`               | Fill-in-the-middle                             |
| POST   | `/v1/rerank`               | Rerank documents                               |
| POST   | `/v1/systemone`            | Decision model questions                       |
| POST   | `/tokenize`, `/detokenize` | Tokenizer round trip                           |
| POST   | `/v1/audio/speech`         | Text to speech (`audio`)                       |
| POST   | `/v1/audio/transcriptions` | Speech to text, with word timestamps (`audio`) |
| POST   | `/v1/audio/translations`   | Speech to English text (`audio`)               |
| GET    | `/v1/audio/voices`         | A TTS model's voices (`audio`)                 |
| POST   | `/api/chat`, `/api/generate` | Ollama chat and prompt completion, NDJSON streams |
| POST   | `/api/embed`, `/api/embeddings` | Ollama embeddings                            |
| GET    | `/api/tags`, `/api/ps`     | Ollama-compatible model listing                |
| GET    | `/metrics`                 | Prometheus metrics                             |

| API | Stream format |
| --- | --- |
| OpenAI | `data: {...}` chunks, ending with `data: [DONE]` |
| Anthropic | Named `event:` lines, each followed by a `data: {...}` chunk. No `[DONE]` |
| Ollama | One JSON object per line |

## CLI Reference

| Command          | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `blazr run`      | Interactive generation (REPL)                        |
| `blazr generate` | Generate text from a prompt                          |
| `blazr chat`     | Interactive multi-turn chat                          |
| `blazr serve`    | Start the inference server                           |
| `blazr info`     | Display model architecture and config                |
| `blazr list`     | List available local models                          |
| `blazr pull`     | Download a model from the Hugging Face Hub           |
| `blazr bench`    | Measure prefill and decode throughput                |
| `blazr ps`       | Show the models a running server holds               |
| `blazr convert`  | Convert a model. Needs `compressr` on `PATH`         |
| `blazr completions` | Print shell completions                           |
| `blazr speak`    | Narration from a script (`audio`)                    |
| `blazr transcribe` | Word timings from a Whisper model (`audio`)        |
| `blazr sound-review` | Review rendered sound effects (`audio`)          |
| `blazr decision` | Decision model tools, such as `calibrate`            |
| `blazr swarm`    | Join or start a distributed inference swarm (`distributed`) |
| `blazr disagg`   | Run prefill and decode disaggregation (`distributed`) |

Run `blazr <command> --help` for every flag. [docs/configuration.md](docs/configuration.md) documents the flags, the config file, and the environment variables.

## Chat Templates and Images

Chat routes render messages with the model's own Jinja template. `--chat-template` replaces it with a built-in name (`chatml`, `llama3`, `mistral`, `phi3`, `gemma`, `deepseek`, `generic`) or a template file. See [docs/models.md](docs/models.md#chat-templates).

With `--mmproj`, the chat, responses, and messages routes accept images beside text. Without it, an image request gets a 400 that names the flag. See [docs/api.md](docs/api.md#image-and-audio-input).

## Speech

Build with `--features audio`, or use a desktop archive. The feature adds `/v1/audio/*` routes to `blazr serve` and three commands.

| Command | Purpose |
| --- | --- |
| `blazr speak` | Turns a script into a voiceover WAV. Whisper hears each phrase back and reports any mismatch. |
| `blazr transcribe` | Writes word timings from a local Whisper model, one words file per narration. |
| `blazr sound-review` | Reviews rendered sound effects, then ranks each against its text with CLAP. `--no-model` skips the ranking. |

The commands write reports to standard output and logs to standard error.

| Exit code | Meaning |
| --- | --- |
| 0 | Done: every phrase heard as written, every narration read, every sound passed |
| 1 | A mismatch remains, or the run failed |
| 2 | A bad input: a missing file, a flag out of range, a bad manifest |
| 3 | `speak --model` names a model blazr does not run. The caller runs that model elsewhere |

A command fetches a missing model from the Hugging Face Hub on first use. [docs/models.md](docs/models.md#speech) holds the model list, search path, and device rules.

## C library

The `capi` feature builds a C library for app embedders.

```bash
cargo build --release --lib --no-default-features --features capi,f16
```

- Output: `libblazr.a` and `libblazr.so` in `target/release/`.
- Header: [`include/blazr.h`](include/blazr.h).
- Status: every call returns a `BlazrStatus`. Zero and above mean success.
- Bindings: blazr ships no Kotlin or Swift bindings. Call the C functions through JNI or a bridging header.
- Functions and link steps: [examples/capi/README.md](examples/capi/README.md). [`examples/capi/smoke.c`](examples/capi/smoke.c) calls every step.

| Target | Compile check | Run |
| --- | --- | --- |
| Android `aarch64-linux-android` | Passes on NDK 27 | x86_64 emulator passes. Not run on aarch64 hardware |
| iOS `aarch64-apple-ios`, `aarch64-apple-ios-sim` | CI checks both | Not stated |
| WebAssembly `wasm32-unknown-unknown`, `wgpu` feature | Passes | Not run in a browser |

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

Limits:

- Clef-Flash memory: the text backbone takes 16.7 GiB in BF16. It does not fit a 12 GB GPU.
- CUDA: Kev and Clef are not tested on it.
- Clef 27B: not tested and not claimed.
- Other families: not served.

[docs/decision-models.md](docs/decision-models.md) has run commands, family detection, calibration, and the tested matrix.

## Documentation

| Doc                                            | Content                                               |
| ---------------------------------------------- | ----------------------------------------------------- |
| [docs/api.md](docs/api.md)                     | Every HTTP endpoint, request field, and error         |
| [docs/configuration.md](docs/configuration.md) | Config file, CLI flags, environment variables         |
| [docs/deployment.md](docs/deployment.md)       | Install, systemd, Docker, reverse proxy, TLS, metrics |
| [docs/models.md](docs/models.md)               | Architecture, format, and quantization matrix         |
| [docs/decision-models.md](docs/decision-models.md) | System One decision models: families, run, limits |
| [docs/tuning.md](docs/tuning.md)               | Batching, caching, parallelism knobs                  |
| [docs/architecture.md](docs/architecture.md)   | Layers, request path, modules, features, commands     |
| [examples/capi/README.md](examples/capi/README.md) | C library: functions, link steps, target checks   |

## Related Projects

- [boostr](https://github.com/ml-rust/boostr): ML framework, model architectures, quantization
- [numr](https://github.com/ml-rust/numr): tensors, linear algebra, CPU and GPU backends
- [voxr](https://github.com/ml-rust/voxr): audio and speech pipelines behind `audio`
- [oxidizr](https://github.com/ml-rust/oxidizr): training framework for hybrid architectures
- [splintr](https://github.com/ml-rust/splintr): BPE tokenizer
- [compressr](https://github.com/ml-rust/compressr): model conversion and compression

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).

Built with Llama.

blazr embeds the Llama 3 tokenizer vocabulary through [splintr](https://github.com/ml-rust/splintr). That vocabulary is under the Llama 3.1 Community License.

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

Each release archive carries that licence as `LICENSE-LLAMA`, and the licence of every compiled-in crate in `THIRD-PARTY-LICENSES.txt`.

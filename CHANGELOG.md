# Changelog

All notable changes to blazr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
blazr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

0.1.0 is the first release. It replaces `0.1.0-beta.1`, a candle-based
preview on crates.io that shares no code with it.

---

## [Unreleased]

### Changed

- **`speak`** — the CUDA build runs Kokoro and VITS on the GPU, as it runs
  Whisper and VoxCPM2. `--device cpu` keeps them on the CPU.
- **`speak`** — a Kokoro take follows `--seed`: one seed gives the same take
  on one device.
- **`serve`** — `--tts-model` loads Kokoro, VITS and VoxCPM2 on the server's
  device, as `--asr-model` loads Whisper. A CUDA build serves them from the
  configured CUDA device instead of the CPU or device 0.

### Fixed

- **`serve`** — `/v1/audio/transcriptions` and `/v1/audio/speech` make the
  server's CUDA device current on the worker thread before they run a
  model. Transcription runs on a blocking worker instead of the request's
  async task, and on the configured CUDA device instead of device 0.

---

## [0.1.2] - 2026-10-11

### Changed

- **`speak`** — the speech check summary counts the parts marked `ok ?`,
  with their part numbers and kinds, instead of saying every phrase was
  heard back as written. The exit code stays 0.
- **`speak`** — a VoxCPM2 stray pause advises `--reroll` for its part, and
  each stray pause counts against a take, so retries and re-rolls keep the
  take with fewer.
- **`speak`** — a voice designed from an opening of fewer than 25 words
  prints that its reference is short.
- **Dependencies** — voxr and voxr-narrate 0.3.

---

## [0.1.1] - 2026-10-10

### Added

- **`--version`** names the optional features the binary is built with,
  such as `blazr 0.1.1 (audio, cuda)`.

### Changed

- **`speak`** — a build without CUDA reads the GPU with `nvidia-smi`, so it
  picks the same default engine as the motion-video skill. A run that
  resolves to VoxCPM2 on a GPU that fits it, with no `--device`, exits 3, so
  the caller runs VoxCPM2 another way.
- **`speak` and `transcribe`** — a Whisper model that holds only ONNX or
  `.bin` weights, named by `--model` or the config's `asr.model`, exits 3
  instead of 1.

---

## [0.1.0] - 2026-10-09

blazr runs on boostr and numr, serves an OpenAI-, Anthropic- and
Ollama-compatible API, and adds speech behind the `audio` feature: TTS and
ASR routes, and the `speak`, `transcribe` and `sound-review` commands.
Prebuilt CPU binaries ship for Linux, macOS and Windows. The CUDA build
installs from crates.io.

Built on boostr 0.5, numr 0.10, splintr 0.21, voxr 0.2 and voxr-narrate 0.2.
Minimum Rust: 1.95.

### Added

- **Commands** — `serve`, `generate`, `run` (REPL), `chat`, `info`, `list`,
  `pull`, `bench`, `ps`, `convert` (through compressr) and `completions`.
  `swarm` and `disagg` run distributed inference behind the `distributed`
  feature.
- **Narration commands** (`audio`)
  - `blazr speak` turns a script into a voiceover WAV with Kokoro,
    VoxCPM2, a VITS (MMS-TTS) model, or any TTS command. It hears every
    phrase back with Whisper and reports the phrases that do not match.
  - `blazr transcribe` writes word timings from a local Whisper model: one
    words file per narration, and the video timeline from a manifest.
  - `blazr sound-review` checks rendered sound effects for sharpness, hiss,
    tail and true peak, and ranks variants against their text with CLAP.
  - Exit codes: 0 done, 1 a problem remains or the run failed, 2 a bad
    input, 3 a `speak --model` blazr does not run.
  - The commands accept the motion-video skill's global flags and ignore the
    ones they do not read, so the skill calls blazr with its own arguments.
- **Model download on first use** — a Whisper, Kokoro, Kokoro G2P, VoxCPM2,
  VITS or CLAP model missing from disk downloads once from the Hugging Face
  hub, in hub layout. Only the files the loader reads are fetched.
  `HF_HUB_OFFLINE=1` turns downloads off.
- **API routes** — `/v1/completions`, `/v1/chat/completions`,
  `/v1/embeddings`, `/v1/responses`, `/v1/messages` and
  `/v1/messages/count_tokens`, `/v1/infill`, `/rerank` and `/v1/rerank`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/models` and
  `/v1/models/{id}`, `/api/slots`, `/v1/lora`, `/metrics`, `/health`, and the
  Ollama-compatible `/api/tags`, `/api/show`, `/api/ps`, `/api/delete`,
  `/api/copy` and `/api/pull`.
- **Audio routes** (`audio`)
  - `POST /v1/audio/speech` synthesizes with Kokoro, VoxCPM2 or VITS, as
    `wav`, `pcm` or `f32`. `"stream": true` streams the samples.
  - `POST /v1/audio/transcriptions` and `/v1/audio/translations` run
    Whisper. `timestamp_granularities[]=word` adds word timestamps to
    `verbose_json`.
  - `GET /v1/audio/voices` lists a TTS model's voices. `--voice-dir` is
    searched before the TTS model directory's `voices/`.
  - `serve` flags: `--asr-model`, `--tts-model`, `--voice-dir`, and the
    VoxCPM2 batching flags `--tts-max-batch`, `--tts-batch-window-ms`,
    `--tts-no-continuous-batching` and `--tts-continuous-capacity`.
  - A VoxCPM2 model directory's `adapters/<one>.safetensors` folds a LoRA
    adapter into the weights at load. A lone VoxCPM2 GGUF loads without
    `config.json`.
- **Chat** — the model's own Jinja chat template renders every prompt, with
  built-in templates as the fallback. Tool calling, a reasoning split
  (`reasoning_content`), grammar and JSON-schema constrained output, logit
  bias and `n` completions.
- **Images** — `qwen35` models with a Qwen3-VL `--mmproj` take images on the
  chat, Responses and Messages routes.
- **Batching** — `qwen35` hybrid GDN/attention models decode in batched
  lanes, with chunked prefill interleaved into occupied lanes. Other models
  with paged attention decode in a continuous batch.
- **Engine** — CUDA graph decode reused across requests, speculative
  decoding, LoRA, MoE offload, prefix caching, tensor parallelism (`nccl`)
  and data parallelism.
- **Models** — LLaMA, Mistral, Mamba2/3, DeepSeek MLA+MoE, hybrid
  Mamba+attention and `qwen35` (Bonsai-2) architectures. Weights load from
  SafeTensors and GGUF. LLaMA and Mistral also load from AWQ and GPTQ.
- **Server** — API keys (`--api-key`, `--api-key-file`, `BLAZR_API_KEY`),
  HTTPS (`--tls-cert`, `--tls-key`), Prometheus metrics, inference slots,
  per-request `keep_alive` with model eviction, an in-flight token budget,
  and a hot-reloaded user config at `~/.blazr/config.yaml`.
- **TLS** — rustls with the ring provider. Hugging Face downloads go over
  rustls, so the Linux binary needs no system OpenSSL.
- **Releases** — prebuilt archives for `linux-x64`, `linux-arm64`,
  `macos-arm64` and `windows-x64`, with `SHA256SUMS`. Each carries `NOTICE`
  (the Llama attribution and the notices for ported code), `LICENSE-LLAMA`
  and `THIRD-PARTY-LICENSES.txt`, the licence of every compiled-in crate.

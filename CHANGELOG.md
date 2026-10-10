# Changelog

All notable changes to blazr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
blazr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The 0.1.0 entry is a delta against `0.1.0-beta.1`, the one earlier release on
crates.io. That release ran on candle and predates this file. Every later
entry is a delta against the one below it.

---

## [0.1.0] - 2026-10-09

blazr runs on boostr and numr, serves an OpenAI-, Anthropic- and
Ollama-compatible API, and adds speech behind the `audio` feature: TTS and
ASR routes, and the `speak`, `transcribe` and `sound-review` commands.
Prebuilt CPU binaries ship for Linux, macOS and Windows. The CUDA build
installs from crates.io.

### Breaking

- **Backend** — candle is replaced by boostr and numr. Every model
  architecture, tensor op and quantization kernel now comes from boostr.
- **Library API** — the beta's `hub`, `inference` and `model::{mamba2,
  mamba3, mla, moe, rope, transformer}` modules and `get_device` are removed.
  The crate exposes `cli`, `config`, `engine`, `loader`, `model`, `server`,
  `tokenizer` and `tls`, plus `distributed` behind its feature.
- **`blazr info`** takes the model as a positional argument:
  `blazr info <MODEL>`, not `--model`.
- **`blazr serve`** has no `-p` short flag for `--port`. `--model` is
  optional: without it, models load on demand.
- **`blazr generate`** drops `--vocab`: the tokenizer comes from the model.
  `--max-tokens` defaults to 2048 and `--top-k` to 0 (off).
- **`blazr decode`** is hidden and reads a file with a model:
  `decode --model <MODEL> <INPUT>`. `--ids` and `--vocab` are gone.
- **Speech is opt-in** — the audio routes and commands build only with
  `--features audio`. The prebuilt binaries include it.
- **Minimum Rust** — 1.89.
- **Repository** — `github.com/ml-rust/blazr`.

### Added

- **Release archives** carry `NOTICE` (the Llama attribution and the
  notices for ported code), `LICENSE-LLAMA` and
  `THIRD-PARTY-LICENSES.txt`, the licence of every compiled-in crate.
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
- **Audio routes** (`audio`)
  - `POST /v1/audio/speech` synthesizes with Kokoro, VoxCPM2 or VITS, as
    `wav`, `pcm` or `f32`. `"stream": true` streams the samples.
  - `POST /v1/audio/transcriptions` and `/v1/audio/translations` run
    Whisper. `timestamp_granularities[]=word` adds word timestamps to
    `verbose_json`.
  - `GET /v1/audio/voices` lists a TTS model's voices.
  - `serve` flags: `--asr-model`, `--tts-model`, `--voice-dir`, and the
    VoxCPM2 batching flags `--tts-max-batch`, `--tts-batch-window-ms`,
    `--tts-no-continuous-batching` and `--tts-continuous-capacity`.
  - A VoxCPM2 model directory's `adapters/<one>.safetensors` folds a LoRA
    adapter into the weights at load. A lone VoxCPM2 GGUF loads without
    `config.json`.
- **Commands** — `run` (REPL), `chat`, `list`, `pull`, `bench`, `ps`,
  `convert` (through compressr) and `completions`. `swarm` and `disagg` run
  distributed inference behind the `distributed` feature.
- **API routes** — `/v1/embeddings`, `/v1/responses`, `/v1/messages` and
  `/v1/messages/count_tokens`, `/v1/infill`, `/rerank` and `/v1/rerank`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/models/{id}`,
  `/api/slots`, `/v1/lora`, `/metrics`, and the Ollama-compatible
  `/api/tags`, `/api/show`, `/api/ps`, `/api/delete`, `/api/copy` and
  `/api/pull`.
- **Chat** — the model's own Jinja chat template renders every prompt, with
  built-in templates as the fallback. Tool calling, a reasoning split
  (`reasoning_content`), grammar and JSON-schema constrained output, logit
  bias and `n` completions.
- **Images** — `qwen35` models with a Qwen3-VL `--mmproj` take images on the
  chat, Responses and Messages routes.
- **Lane batching** — `qwen35` hybrid GDN/attention models decode in
  batched lanes, with chunked prefill interleaved into occupied lanes. Other
  models with paged attention decode in a continuous batch.
- **Engine** — CUDA graph decode reused across requests, speculative
  decoding, LoRA, MoE offload, prefix caching, tensor parallelism (`nccl`)
  and data parallelism.
- **Models** — `qwen35` (Bonsai-2) hybrid GGUF models, and more GGUF
  architectures.
- **Server** — API keys (`--api-key`, `--api-key-file`, `BLAZR_API_KEY`),
  HTTPS (`--tls-cert`, `--tls-key`), Prometheus metrics, inference slots,
  per-request `keep_alive` with model eviction, an in-flight token budget,
  and a hot-reloaded user config at `~/.blazr/config.yaml`.
- **Releases** — prebuilt archives for `linux-x64`, `linux-arm64`,
  `macos-arm64` and `windows-x64`, with `SHA256SUMS`.

### Changed

- **Audio stack** — audio decoding, the TTS engines and Whisper moved from
  boostr-audio to voxr. The narration logic comes from voxr-narrate.
- **Dependencies** — boostr 0.3, numr 0.8, splintr 0.21, voxr 0.1 and
  voxr-narrate 0.1.
- **TLS** — one rustls crypto provider, ring, for the whole process.
  Hugging Face downloads go over rustls, so the Linux binary needs no system
  OpenSSL. The archived `rustls-pemfile` is replaced by `rustls-pki-types`'
  PEM parser.
- **`--voice-dir`** is searched before the TTS model directory's `voices/`.
  The voice directory that the beta bundled under `assets/` is gone: Kokoro
  voices come from the model directory.

### Fixed

- **HTTPS** — `serve --tls-cert --tls-key` no longer panics while building
  the TLS config. `--tls-cert` without `--tls-key`, or the reverse, is an
  error instead of a silent HTTP listener.
- **`blazr convert`** passes `--output` to compressr, which has no `--out`.
  `--verbose` sets compressr's log level instead of passing a flag compressr
  rejects. compressr is found on `PATH` on Windows too.
- **Voices** — a voice in `--voice-dir` reaches the Kokoro engine, and a
  voice in the model's `voices/` passes the request check.
- **Chat templates** — message content can no longer inject template
  markers, and special tokens obey their allow-lists on every encode path.
- **Loader** — `VarMap` memory no longer leaks, and short/long RoPE factors
  load from GGUF and SafeTensors configs.
- **Batched decode** — paged decode reads each row's own KV range, and
  `qwen35` graph decode tracks its RoPE position apart from the KV length.

# Changelog

All notable changes to blazr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
blazr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

0.1.0 is the first release. It replaces `0.1.0-beta.1`, a candle-based
preview on crates.io that shares no code with it.

---

## [Unreleased]

### Added

- **`loader::load_safetensors_with_report`** — the regular SafeTensors load, plus a `TensorLoadReport`: the tensors uploaded, the tensors left on disk, and the compute dtype. `loader::model_torch_dtype` gives the dtype a model's `config.json` names.
- **`tests/decision_loader_real_file.rs`** — a Qwen3.5 decision model (Jebadiah, the Mapika decider) through blazr's loader, then `LabelTokenHead` against reference fixtures, from SafeTensors or GGUF, on CPU or CUDA. It skips unless `BLAZR_DECISION_MODEL_DIR` or `BLAZR_DECISION_GGUF`, and `BLAZR_DECISION_FIXTURE_DIR`, are set.
- **`POST /v1/systemone`** — typed decisions from a System One model (Jebadiah, the Mapika decider), each answer in the family's own wire shape. The family comes from `--decision-family`, a sidecar beside the weights, or the GGUF header. Each model gets one decision worker thread: requests with a short shared prefix merge into one run, longer ones run once over their prefix. Limits answer `413` and a full queue `503`. Keys: `inference.decision.*` and the `--decision-*` flags. `GET /v1/models` adds a `decision` object read without loading weights.
- **`blazr_decision_*` metrics** — requests, rows, prefill tokens, latency and queued rows, by model.
- **`blazr decision calibrate`** — fits one temperature per answer type (choice, noul, score) of a decision model on a JSON-lines label file, with boostr's fitters. A line is a labelled request (`state`, `questions`, `labels`), rendered by the model's renderer and run at temperature 1 on its decision worker, or a precomputed `decider/calibrate.py` record. A records-only file loads no weights. A type with fewer than `--min-rows` answers keeps the vendor temperature. The fit goes to `blazr_calibration.json` in the sidecar directory, or `--out`. `--dry-run` writes nothing. `blazr serve` and `/v1/models` take each type that file sets over the vendor's `temperature_by_type` or `temperatures.json`. `calibration.source` is then `blazr_calibration.json`, and `/v1/models` adds `fitted_at` and per-type `sources`. An invalid file stops the model from loading.
- **`boostr::inference::decision::fit_temperature_mixed`** — one temperature over listed and isolated Score answers together, the Score fit of `blazr decision calibrate`.
- **LoRA variants** — a PEFT adapter applies to a loaded `qwen35` model as the variant `<model>:<name>`, built with boostr's `LoadedModel::derive_with_adapter`: shared base weights, the adapter unmerged, its own executor and KV cache. `serve --lora <dir>[=<name>]` (repeatable) and `inference.lora` register variants of the start model; `POST /v1/lora` registers one at run time. A request selects a variant by `model: "<model>:<name>"` or `lora_adapter`; an unknown adapter is a `404`. `run` and `generate` take one `--lora`. `/v1/models` and `GET /v1/lora` list each variant's base, path, rank and adapted module count. Eviction drops a variant before its base; a registered variant builds again on its next request. `/v1/systemone` serves variants. Data-parallel start models refuse adapters.
- **`tests/lora_variant_real_file.rs`** — a Qwen3.5 base and a PEFT adapter through the scheduler's variant path, checked against `transformers` + `peft` logits. It skips unless `BLAZR_LORA_BASE_DIR`, `BLAZR_LORA_ADAPTER_DIR`, `BLAZR_LORA_REF` and `BLAZR_LORA_BASE_REF` are set.
- **Kev on `POST /v1/systemone`** — a Kev directory (a LoRA adapter and a pointer head in `head.pt`, no base weights) in the model directory serves by its name. blazr finds the base with `--decision-base` (`inference.decision.base`), else by the adapter's `base_model_name_or_path` in the model directory or a Hugging Face hub cache, and never downloads it. The base loads once under its own name. The Kev model is the LoRA variant of it, and boostr's `PointerHead` reads the answers at the fitted temperature `head.pt` holds. Prompts and answers follow Kev's own code: every question on its own row, at most 255 options, states over 65,536 tokens refused (`--decision-truncate-states` cuts them instead), 4-digit answers, and Kev's `usage`. `/v1/models` reports `family: "kev"`, `readout: "pointer"`, the base, the adapter rank and the temperature.
- **`tests/decision_kev_fixtures.rs`** and **`tests/decision_kev_route.rs`** — Kev's rows, positions and answers against fixtures from Kev's own code, without weights; then the route end to end on the CPU in F32. They skip unless `BLAZR_DECISION_MODEL_DIR` and `BLAZR_DECISION_FIXTURE_DIR` are set, and the route test also `BLAZR_DECISION_BASE`.
- **Clef on `POST /v1/systemone`** — a Clef or Clef-Flash directory serves by its name: a full fine-tuned Qwen3.5 checkpoint with `joint_head_config.json` and `joint_head.safetensors` beside it, or any checkpoint named Clef with `--decision-family clef`. The backbone loads through blazr's own loader; boostr's `JointSchemaHead` reads every hidden row of the record and scores all questions together, in F32, with no temperature. Prompts, option order, limits, error words and answers follow the release's `joint_schema_model.py`: one record of at most 16,384 tokens with the state cut to fit, choice probabilities in the order the request wrote the keys, `confidence` the chosen option's probability, and a one-option field accepted. Control-token text in a request stays text, unlike the release. Requests with images or video are refused. `/v1/models` reports `family: "clef"`, `readout: "joint_schema"` and the head's width and dtype. `blazr decision calibrate` refuses Clef. Clef-Flash's text backbone is 16.7 GiB in BF16: it serves on the CPU with that much free RAM, or on one GPU that holds it.
- **`tests/decision_clef_fixtures.rs`** and **`tests/decision_clef_route.rs`** — Clef's records and spans against the release's fixtures, and the head through blazr's readout over the fixtures' saved hidden rows, without the backbone; then the route end to end on the CPU. They skip unless `BLAZR_DECISION_MODEL_DIR` and `BLAZR_DECISION_FIXTURE_DIR` are set, and the route test also `BLAZR_CLEF_FULL=1`.

### Changed

- `Executor::get_embeddings` is a plain function. `/v1/embeddings` and `/rerank` run it on a blocking thread, never inside an async task.
- `POST /v1/lora` reads adapters through boostr's `PeftAdapter`. A load now fails when `adapter_config.json` is missing or lacks `r` or `lora_alpha`. It also fails on an unpaired `lora_A`/`lora_B`, a rank mismatch, `.bin` weights, DoRA, `fan_in_fan_out`, `lora_bias` and `modules_to_save`. `use_rslora`, `rank_pattern` and `alpha_pattern` now set each module's scaling.

### Removed

- `engine::load_lora_adapter`, `engine::LoraAdapter`, `engine::LoraAdapterRegistry`, and `Executor::{load_lora, unload_lora, list_loras, lora_registry}`. `Scheduler::{register_variant, remove_variant, variants, select_model}` replace them.

### Fixed

- **Sharded SafeTensors directories** — a directory with `model.safetensors.index.json` loads only the shards the index names (boostr's `SafeTensorsLoader`). Every `.safetensors` file of the directory loaded before, so a head or adapter file beside the shards was read as part of the model.
- **`lora_adapter` on generation requests** — it now runs the request on the adapter's variant. It was logged and ignored before: the request generated with the base weights, and an unknown name fell back to the base.
- **Qwen3.5 SafeTensors configs** — a `qwen3_5` (nested `text_config`) or `qwen3_5_text` `config.json` goes to boostr's `qwen35_config_from_hf` before any other parse. It fell through to tensor-name detection or a partial parse before.
- **Config dtype** — the compute dtype reads `torch_dtype`, then `dtype`, then both under `text_config`. Only the top-level `torch_dtype` was read before, so transformers v5 and `Qwen3_5ForConditionalGeneration` configs fell back to the backend default.
- **Qwen3.5 MTP and vision tensors** — they stay on disk in every SafeTensors path. They were uploaded to the device and then dropped before.
- **Qwen3.5 edit precision** — `1 + w` and `-exp(A_log)` round once, to the compute dtype. Each 1-D tensor waits in F32 when the compute dtype cannot hold its file value. A BF16 model rounded these edits twice before, and the F32 `A_log` was rounded to BF16 first.
- **Sharded checkpoints named by an index** — a directory with `model.safetensors.index.json` is SafeTensors whatever its shard names, such as `model.safetensors-00001-of-00002.safetensors`. It failed with `No supported model files found` before.
- **Offloading load** — `load_safetensors_with_offloading` calls `restructure_hf_checkpoint`, as the other SafeTensors paths do. It takes the layer count from the config and the layer prefix from the tensor names. Non-Llama layouts failed to load through it before.

---

## [0.3.0] - 2026-10-12

### Added

- **`tests/arch_reference.rs`** — greedy decode of real checkpoints against `transformers` on CPU in F32. It covers Llama, Qwen2, Gemma 2, GPT-NeoX, Falcon, Mixtral, Qwen2-MoE, DBRX and Cohere. Each test reads its own `BLAZR_TEST_*_MODEL` variable and skips when unset.

### Changed

- **Dependencies** — boostr 0.8, splintr 0.22, and voxr and voxr-narrate 0.6. `blazr::boostr` re-exports boostr 0.8.

### Fixed

- **Checkpoint tokenizer** — a SafeTensors checkpoint now runs with its own `tokenizer.json`. Qwen2, Phi-3, Gemma and StarCoder2 failed to load before: their padded embedding size matched no bundled vocabulary. A `tokenizer.json` that does not load is an error. The bundled vocabulary serves only a checkpoint without one.
- **Non-Llama tensor layouts** — the SafeTensors loaders call boostr's `restructure_hf_checkpoint`. GPT-NeoX, Falcon, DBRX, Phi-3, StarCoder2, Mixtral and Qwen2-MoE checkpoints failed with a missing tensor before.
- **Architecture detection** — tensor-name detection runs only when `config.json` describes no model. A Falcon or DBRX checkpoint failed with `No layers detected in checkpoint` before.
- **Paged attention** — `Executor::new` refuses paged attention for an ALiBi or softcapped model. The paged kernel applied neither, and the output was wrong.
- **`SafeTensorsBytes`** — `tokenizer_json` is now the tokenizer, not only its BOS rule.

### Documentation

- **Android** — `examples/capi/README.md` gives the x86_64 emulator build, link and run steps, including `-Wl,-z,max-page-size=16384` for 16 KB pages. The emulator run passes on CPU and on wgpu through the host GPU. No aarch64 hardware or physical device has run blazr.

---

## [0.2.0] - 2026-10-11

### Added

- **`wgpu` feature** — the WebGPU runtime over Metal, Vulkan, DX12 and
  WebGPU. Select it with `device: wgpu` or `device: wgpu:N`. `--version`
  lists it. With `cuda` also on, CUDA wins.
- **`device` key** — takes `wgpu` and `wgpu:N` beside `cpu`, `cuda` and
  `cuda:N`.
- **`cli`, `server` and `hub` features** — `cli` holds the command-line
  front end. `server` holds the HTTP front end. `hub` holds the Hugging
  Face client that both enable.
- **Engine library** — `--no-default-features` builds the engine alone,
  without the CLI and the HTTP stack.
- **C API** — the `capi` feature builds `libblazr.a` and `libblazr.so`
  with the C ABI in `include/blazr.h`. `examples/capi/` shows the link
  steps. No Kotlin or Swift bindings exist.
- **Engine entry point** — `engine::load_executor_from_bytes` loads a GGUF
  file, or a SafeTensors file plus its JSON, from memory.
- **Compute dtype** — the `--dtype f32|bf16|f16` flag of `run`, `generate`,
  `chat`, `bench` and `serve` sets the compute dtype. So does the
  `inference.dtype` user file key. The flag wins over the key. Without
  either, each model's `torch_dtype` applies, else `f32`.
- **SafeTensors BOS** — a SafeTensors checkpoint prepends BOS by the rule
  in its own files: `tokenizer_config.json` `add_bos_token`, then the
  Llama-family `tokenizer_class` default, then the `tokenizer.json`
  post-processor.
- **Token text streaming** — each request decodes its generated tokens
  through one streaming decoder primed with the prompt. A SentencePiece
  token keeps its leading space, and a character split over several ids
  streams whole.
- **Release archives** — `release.yml` builds `linux-x64`, `linux-x64-gpu`,
  `linux-arm64`, `macos-arm64`, `windows-x64`, `windows-x64-gpu`,
  `android-arm64` and `ios-arm64` archives, with `SHA256SUMS`. The `-gpu`
  archives add `wgpu`. The macOS archive runs `wgpu` on Metal. No archive
  includes CUDA.
- **Target builds** — `aarch64-linux-android` compiles with `capi`.
  `wasm32-unknown-unknown` compiles with `wgpu`. No model has run on a
  device or in a browser.

### Changed

- **Dependencies** — boostr 0.7, numr 0.12, splintr 0.21.1, and voxr and voxr-narrate 0.5.
- **Default features** — the default set is `f16`, `cli` and `server`. The
  `blazr` binary requires `cli` and `server`. `info`, `list`, `pull`,
  `run`, `chat`, `bench` and `convert` need `cli`. `serve` and `ps` need
  `server`.
- **Server dependencies** — axum, tower-http, TLS, reqwest, the Prometheus
  exporter and tokio's `full` feature set sit behind `server`. Without it,
  tokio uses `sync`, `time` and `rt`.
- **Metrics** — the engine records through the `metrics` facade. The
  Prometheus exporter moves behind `server`. Metrics, SLO checks and the
  test recorder move into the `telemetry` module.
- **GGUF loader** — GGUF files load through mmap on every target except
  `wasm32`. For Llama 3.2 1B Q4_K_M on x86_64 Linux, peak RSS fell from
  2,977,316 kB to 1,929,888 kB.
- **Pooling** — the embedding pooling helpers move from `server` to
  `engine::pooling`.
- **Browser builds** — on `wasm32-unknown-unknown`, `getrandom` uses the
  `wasm_js` backend and `uuid` uses the `js` feature.
- **Dependencies** — `dotenvy` is removed.

---

## [0.1.3] - 2026-10-11

### Added

- **User config file** — `~/.blazr/config.yaml` holds the `server`,
  `inference` and `generation` sections beside the user keys. `run`,
  `serve`, `generate`, `chat`, `bench`, `list`, `info`, `pull`, `swarm` and
  `disagg` read it. Each value comes from the CLI flag, then the
  environment variable, then the file, then the built-in default.
- **Config `inference` section** — the file's `inference` keys apply over
  every model's own `config.json`. For `inference.*` keys the model's
  `config.json` sits between the file and the built-in default.
- **Config keys** — `default_model` names the model when a command names
  none. `device` takes `cpu`, `cuda` or `cuda:N`. `model_dir` sets the model
  directory after `BLAZR_MODEL_DIR`. `num_ctx` sets the initial KV cache
  size, `serve` included.
- **`server` keys** — `serve` reads `port`, `host`, `tls_cert`, `tls_key`,
  `max_concurrent_requests`, `request_timeout_secs`, `cors_enabled`,
  `cors_origins`, `request_logging`, `max_body_size`,
  `max_inflight_tokens`, `slo` and `reasoning_format` from the file. These
  values were fixed before. The swarm leader and the disaggregated router
  read the same section.
- **Config hot reload** — `serve` swaps in each version of the file that
  parses, as a whole. A file that fails to parse keeps the running config
  and logs an error. A deleted file resets every key to its default.
- **Live keys** — `generation` keys and `server.reasoning_format` apply to
  the next request. Every other changed key logs a warning to restart
  `serve`.
- **Data-parallel replicas** — `serve --data-parallel <N>` or
  `inference.data_parallel` loads `N` copies of the start model. A CUDA
  build puts copy `i` on CUDA device `device + i`. A CPU build loads CPU
  copies. Each copy runs its own batching scheduler.
- **Data-parallel checks** — `serve` refuses `0` replicas. It refuses more
  than one replica with no start model, or with tensor parallelism above 1.
  A CUDA build refuses more replicas than the devices from the `device`
  index up.
- **Cache-aware router** — with replicas, a request goes to the replica
  that holds its longest known prompt prefix. A request with no known
  prefix goes to the least loaded replica. A warm replica more than 4
  requests above the least loaded one gives up the request.
- **SLO batch-growth hold** — while the rolling P95 TTFT or P95 ITL is over
  its `server.slo` target, the continuous-batching schedulers admit no
  sequences beyond the running count. An idle scheduler still admits one.
  Running sequences keep decoding.
- **SLO samples** — ITL and end-to-end latency samples now feed the SLO
  checker beside TTFT.
- **Metrics** — `blazr_slo_batch_growth_held` is `1` while a breach holds
  batch growth. `blazr_data_parallel_replicas` reports the replica count.
  `blazr_replica_active_requests` reports in-flight requests per
  `replica`.
- **Recorded metrics** — `blazr_queue_depth` counts queued sequences in
  the batching schedulers plus requests waiting for an HTTP slot.
  `blazr_slo_violations_total` counts breaches with `metric` and
  `percentile` labels. `serve` recorded neither before.
- **`/health`** — the response carries `data_parallel_replicas`.
- **Flags** — `swarm --host` and `disagg --host` set the HTTP API address.
  `swarm --api-key` sets the leader's API key, with `BLAZR_API_KEY` as the
  fallback.

### Changed

- **Dependencies** — boostr 0.6, voxr and voxr-narrate 0.4, and through them numr 0.11.
- **CLI sampling defaults** — with no flag and no config, `--temperature`
  now falls back to `1.0` instead of `0.7`. `--top-p` falls back to `1.0`
  instead of `0.9`. This applies to `run`, `generate` and `chat`. The
  sampling flags take the `generation` section before the built-in value.
- **`chat --num-ctx`** — the flag falls back to `num_ctx`, else `2048`
  instead of `4096`.
- **Flag fallbacks** — `--max-tokens`, `--top-k`, `--min-p`,
  `--repeat-penalty`, `--repeat-last-n`, `--num-ctx`, `--host`, `--port`,
  `--reasoning-format`, `--tls-cert` and `--tls-key` fall back to the file
  instead of fixed clap defaults. `--paged-attention` and `--graphs` fall
  back to `inference.paged_attention` and `inference.graphs`.
- **`inference.graphs`** — `run` and `generate` read the key. `serve` logs
  a warning when it is set, because `serve` does not capture graphs.
- **Model argument** — `run`, `generate`, `chat`, `bench`, `serve`,
  `swarm` and `disagg` take `default_model` when no model is given. A
  command that needs a model and finds neither exits with an error that
  names `default_model`.
- **Request defaults** — a request field the request omits takes the live
  `generation` section. `max_tokens` on completions, chat and infill now
  defaults to `2048` instead of `256`. `max_output_tokens` on
  `/v1/responses` defaults to `2048` instead of `4096`.
- **Config file errors** — a config file that does not read or parse exits
  with an error that names the file and line. Before, `serve` logged a
  warning and ran on defaults. An unknown key logs a warning that names
  it.
- **Older keys** — the top-level `port`, `temperature` and `max_tokens`
  still parse as `server.port`, `generation.temperature` and
  `generation.max_tokens`. Setting an older key and its section key
  together is an error.
- **Device** — `run` and `generate` without `--cpu` run on the CUDA index
  from `device`, else device 0. In a CUDA build, `serve`, `chat`, `bench`,
  `swarm` and `disagg` refuse `device: cpu`.
- **Model directory** — `list`, `info`, `pull`, `serve` and the cluster
  roles resolve it from `BLAZR_MODEL_DIR`, then `model_dir`, then
  `./models`.
- **TLS** — the certificate and the key each come from a flag or from
  `server.tls_cert` and `server.tls_key`. One without the other is an
  error.
- **Concurrency limit** — middleware replaces the tower layer for
  `server.max_concurrent_requests`. A request over the limit waits for a
  slot and counts in `blazr_queue_depth`. `/health` and `/metrics` skip
  the limit, so probes and scrapes answer while the server is saturated.
- **Tensor parallelism** — the degree comes from `BLAZR_TP_SIZE`, then
  `inference.tensor_parallel_size` in the file, then the model's
  `config.json`, then `1`. `serve` checks it before any model loads. A
  degree of `0`, or a `BLAZR_TP_SIZE` that is not a whole number, is an
  error. The loader checks that the head counts divide by the degree
  before weights load.
- **Model `config.json`** — blazr ignores its `server` and `generation`
  sections. Its `torch_dtype` still sets the compute dtype when neither
  `--dtype` nor `inference.dtype` is set.
- **`speak`** — the CUDA build runs Kokoro and VITS on the GPU, as it runs
  Whisper and VoxCPM2. `--device cpu` keeps them on the CPU.
- **`speak`** — a Kokoro take follows `--seed`: one seed gives the same take
  on one device.
- **`serve`** — `--tts-model` loads Kokoro, VITS and VoxCPM2 on the server's
  device, as `--asr-model` loads Whisper. A CUDA build serves them from the
  configured CUDA device instead of the CPU or device 0.

### Removed

- **`decode` command** — the hidden command had no implementation.
- **MoE expert offload** — the subsystem never moved experts to host
  memory, so it is gone with its `inference.moe_offload` and
  `inference.moe_gpu_experts` keys. A config that still sets either key
  parses and ignores it. In `config.yaml` each logs an unknown-key warning.
- **Unused `inference` keys** — `weights_path`, `device`, `flash_attention`
  and `kv_cache` had no reader and are gone. A model `config.json` that
  sets them still parses and ignores them.
  In `config.yaml` each logs an unknown-key warning.

### Fixed

- **Image input** — a raw base64 `image_url` that does not decode to a PNG,
  JPEG, WebP or GIF image fails with `Invalid image_url: expected a data:
  URI, an http(s) URL, or base64 of a PNG, JPEG, WebP, or GIF image`.
  Before, it passed on as `application/octet-stream`. A local file path
  fails with an `Invalid image_url` error.
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
  - Exit codes: 0 done, 1 a mismatch remains or the run failed, 2 a bad
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

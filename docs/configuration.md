# blazr config reference

This page lists the config keys, CLI flags, and environment variables that blazr reads.

## Precedence

Blazr applies the first source that sets a value, in this order.

| Order | Source                               | Scope                                                                  |
| ----- | ------------------------------------ | ---------------------------------------------------------------------- |
| 1     | CLI flag                             | Flags listed under [CLI flags](#cli-flags)                             |
| 2     | Environment variable                 | Variables listed under [Environment variables](#environment-variables) |
| 3     | `~/.blazr/config.yaml`               | Keys listed under [User config file](#user-config-file)                |
| 4     | Built-in default                     | Defaults listed in the tables on this page                             |

For `inference.*` keys, the order is: flag, environment variable (`BLAZR_TP_SIZE` only), `config.yaml`, model `config.json`, default. The `inference` section of `config.yaml` replaces the matching value in every model.

Exceptions to the order:

- `--token` takes its value from `BLAZR_SWARM_TOKEN` through the clap `env` attribute. A flag on the command line wins over the variable.
- `--voice-dir` wins over `BLAZR_VOICE_DIR`.
- `pull --output` wins over `BLAZR_MODEL_DIR`, then `model_dir`.
- `--mmproj`, `--image-min-tokens`, and `--image-max-tokens` replace the matching `inference` values when set.
- The compute dtype has its own order. See [Compute dtype](#compute-dtype).

## User config file

Path: `~/.blazr/config.yaml`. Format: YAML, parsed with `serde_saphyr`.

These commands read this file at start:

- `run`, `serve`, `generate`, `chat`, and `bench`
- `list`, `info`, and `pull`
- `swarm` and `disagg`

A missing file gives the built-in defaults. A file that cannot be read or parsed exits with an error. A parse error names the file and line.

Example:

```yaml
default_model: mistral-7b
device: cuda:0
num_ctx: 4096
model_dir: /srv/models

server:
  port: 9090
  max_concurrent_requests: 32

inference:
  max_batch_size: 8
  paged_attention: true

generation:
  temperature: 0.8
  max_tokens: 1024
```

| Key             | Type    | Default  | Meaning                                                                                                                  |
| --------------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `default_model` | string  | unset    | Model that a command uses when it names none                                                                             |
| `device`        | string  | GPU 0 of the build, else `cpu` | `cpu`, `cuda`, `cuda:N`, `wgpu`, or `wgpu:N`. `run` and `generate` with `cpu` use the CPU. `serve`, `chat`, `bench`, `swarm`, and `disagg` refuse `cpu` in a `cuda` or `wgpu` build. A value for a backend the build lacks is an error. |
| `num_ctx`       | integer | `2048`   | Initial KV cache size in tokens for `run`, `generate`, `chat`, `bench`, and `serve`                                      |
| `model_dir`     | string  | `./models` | Model search directory. `BLAZR_MODEL_DIR` overrides it.                                                               |
| `port`          | integer | unset    | Older name of `server.port`                                                                                              |
| `temperature`   | float   | unset    | Older name of `generation.temperature`                                                                                   |
| `max_tokens`    | integer | unset    | Older name of `generation.max_tokens`                                                                                    |

The `server`, `inference`, and `generation` sections hold the other keys. Old top-level keys still parse. Setting an old key and its section key together is an error.

### Compute dtype

The compute dtype is the one dtype of the float weights, the activations, and the KV cache. Blazr resolves it once per model, before any weight loads. It casts every F32, F16, and BF16 weight to it.

| Order | Source                                  | Commands                                            |
| ----- | --------------------------------------- | --------------------------------------------------- |
| 1     | `--dtype <f32\|bf16\|f16>`               | `run`, `generate`, `chat`, `bench`, and `serve`     |
| 2     | `inference.dtype` in `config.yaml`      | every command that loads a model                    |
| 3     | `torch_dtype` in the model `config.json` | SafeTensors loads. See [torch_dtype](#torch_dtype). |
| 4     | `f32`                                   | the backend default                                 |

```yaml
inference:
  dtype: f32
```

| Value  | Meaning                                       |
| ------ | --------------------------------------------- |
| `f32`  | 32-bit float. Every backend runs it.          |
| `bf16` | bfloat16. Needs the `f16` build feature.      |
| `f16`  | IEEE half. Needs the `f16` build feature.     |

Long names `float32`, `bfloat16`, and `float16` also parse.

Rules:

- An unknown value exits with an error. The error names the key or flag and the accepted values.
- `bf16` or `f16` in a build without the `f16` feature exits with an error that names the feature.
- wgpu computes in `f32` only. A request for `bf16` or `f16` logs a warning that names its flag or key. The model then runs `f32`.
- A GGUF model runs `f32`. An AWQ or GPTQ model runs `f16`. A request for another dtype logs a warning, and the model keeps its own dtype.
- A speculative draft model loads at the dtype of its target model.
- The CPU runs `f32`, `bf16`, and `f16`. CUDA runs all three.

### Validation

- An unknown key logs a warning that names the key. Blazr ignores the key.
- A YAML error exits with an error that names the file and line.
- An empty environment variable counts as unset.
- A `BLAZR_TP_SIZE` that is not a whole number exits with an error.

### Removed keys

These keys no longer apply. Each one logs an unknown-key warning in `config.yaml`.

| Key                         | Status                                                  |
| --------------------------- | ------------------------------------------------------- |
| `inference.flash_attention` | Removed                                                 |
| `inference.kv_cache`        | Removed                                                 |
| `inference.moe_offload`     | Removed with the MoE expert offload subsystem           |
| `inference.moe_gpu_experts` | Removed with the MoE expert offload subsystem           |
| `inference.weights_path`    | Removed                                                 |
| `inference.device`          | Removed. Use the top-level `device` key.                |

## Model config.json

Blazr reads `config.json` from the model directory for SafeTensors checkpoints. Blazr tries each parser in order. A parse error moves to the next parser. Blazr logs nothing when a parser fails.

1. `BlazrConfig`: the top-level model keys, plus the `inference` section.
2. `UniversalConfig`: the top-level model keys only.
3. `HuggingFaceConfig`: the Hugging Face keys listed below.
4. Values detected from the checkpoint, when `config.json` is missing or no parser accepts it.

A type error in the `inference` section fails parser 1. Parser 2 then reads the file without that section.

Blazr ignores `server` and `generation` in `config.json`. It ignores unknown `inference` keys without a warning. Only `config.yaml` sets `inference.data_parallel`.

Example:

```json
{
  "model_type": "llama",
  "vocab_size": 32000,
  "hidden_size": 4096,
  "num_layers": 32,
  "max_seq_len": 8192,
  "inference": {
    "max_context_len": 4096,
    "max_batch_size": 4,
    "paged_attention": true,
    "block_size": 16,
    "prefix_cache": true
  }
}
```

### Required model keys

| Key           | Type    | Meaning                 |
| ------------- | ------- | ----------------------- |
| `model_type`  | string  | Architecture name       |
| `vocab_size`  | integer | Vocabulary size         |
| `hidden_size` | integer | Hidden dimension        |
| `num_layers`  | integer | Layer count             |
| `max_seq_len` | integer | Maximum sequence length |

Blazr passes these optional keys to boostr. This page does not list their nested keys.

- `intermediate_size`
- `rms_norm_eps`
- `attention`
- `ssm`
- `moe`
- `hybrid_layers`
- `tie_word_embeddings`
- `grow_vocab`
- `vision`
- `audio`
- `gdn`
- `qwen35_attention`
- `hadamard`

### Hugging Face keys

Parser 3 reads these keys. A key marked with `or` accepts either name.

| Key                                        | Type         | Required | Maps to                          |
| ------------------------------------------ | ------------ | -------- | -------------------------------- |
| `vocab_size`                               | integer      | yes      | Vocabulary size                  |
| `hidden_size`                              | integer      | yes      | Hidden dimension                 |
| `num_hidden_layers` or `num_layers`        | integer      | yes      | Layer count                      |
| `max_position_embeddings` or `max_seq_len` | integer      | yes      | Maximum sequence length          |
| `model_type`                               | string       | no       | Architecture name                |
| `architectures`                            | string array | no       | Architecture class names         |
| `num_attention_heads`                      | integer      | no       | Query head count                 |
| `num_key_value_heads` or `num_kv_heads`    | integer      | no       | Key and value head count         |
| `head_dim`                                 | integer      | no       | Head dimension                   |
| `intermediate_size`                        | integer      | no       | MLP width                        |
| `rope_theta`                               | float        | no       | RoPE base                        |
| `sliding_window`                           | integer      | no       | Sliding attention window         |
| `rms_norm_eps`                             | float        | no       | RMSNorm epsilon                  |
| `rope_scaling`                             | object       | no       | RoPE scaling parameters          |
| `tie_word_embeddings`                      | boolean      | no       | Ties input and output embeddings |
| `num_local_experts`                        | integer      | no       | Expert count                     |
| `num_experts_per_tok`                      | integer      | no       | Experts per token                |

Parser 3 also reads `attention_bias`, `alibi`, `multi_query`, `new_decoder_architecture`, `parallel_attn`, `vision_config`, and `audio_config`. This page does not trace their use.

### torch_dtype

Blazr reads `torch_dtype` from `config.json` for every SafeTensors load. It is step 3 of the [compute dtype](#compute-dtype) order. `--dtype` and `inference.dtype` in `config.yaml` win over it.

| `torch_dtype`               | Resulting dtype                          |
| --------------------------- | ---------------------------------------- |
| `bfloat16`                  | `bf16`                                   |
| `float16`                   | `f16`                                    |
| `float32`                   | `f32`                                    |
| Any other value, or missing | The next step: `f32`, the backend default |

The weights take the resolved dtype, whatever dtype the file stores. A `torch_dtype` of `float32` over BF16 weights runs in `f32`.

An `inference.dtype` in the model `config.json` is ignored. GGUF loads use `f32`.

### inference section

Blazr reads this section from `config.json`. The `inference` section of `config.yaml` replaces these values for every model. The defaults are the `InferenceConfig` defaults.

| Key                                  | Type     | Default             | Meaning                                                                                                                      |
| ------------------------------------ | -------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `max_context_len`                    | integer  | unset               | Overrides the model maximum sequence length. Blazr does not compare the two values.                                          |
| `max_batch_size`                     | integer  | `1`                 | Batch size. Values of 1 or less use the single-sequence path.                                                                |
| `paged_attention`                    | boolean  | `false`             | Enables the paged KV cache. `run` and `generate` also take `--paged-attention`.                                              |
| `block_size`                         | integer  | `16`                | Tokens per block for paged attention                                                                                         |
| `num_blocks`                         | integer  | `0`                 | KV cache blocks to pre-allocate for paged attention                                                                          |
| `kv_pool_blocks`                     | integer  | `0`                 | Blocks in the shared pool. `0` computes the count from `max_context_len` and `max_batch_size`.                               |
| `prefix_cache`                       | boolean  | `false`             | Reuses KV blocks for repeated prompt prefixes. Requires `paged_attention`.                                                   |
| `max_cached_blocks`                  | integer  | `10000`             | Cached prefix blocks to retain. Used when `prefix_cache` is `true`.                                                          |
| `gpu_prefix_cache`                   | boolean  | `false`             | GPU prefix cache lookup. Falls back to CPU lookup without CUDA.                                                              |
| `gpu_prefix_cache_ram_tier`          | integer  | `5000`              | RAM tier entries for the GPU prefix cache                                                                                    |
| `prefill_chunk_size`                 | integer  | `0`                 | Prompt tokens per prefill chunk. `0` processes the whole prompt at once.                                                     |
| `speculative`                        | object   | unset               | Speculative decoding. Blazr uses it when the section is set.                                                                 |
| `speculative.draft_model`            | string   | required in section | Draft model path. A relative path also resolves against the model directory.                                                 |
| `speculative.num_speculative_tokens` | integer  | `5`                 | Tokens speculated per step.                                                                                                  |
| `speculative.adaptive_depth`         | boolean  | `false`             | Adjusts speculation depth from the acceptance rate. Passed to boostr `SpeculativeConfig`.                                    |
| `tensor_parallel_size`               | integer  | `1`                 | Tensor parallel degree. `BLAZR_TP_SIZE` overrides it for `serve`.                                                            |
| `data_parallel`                      | integer  | `1`                 | Replicas of the start model for `serve`. Only `config.yaml` sets it. See [Data-parallel replicas](deployment.md#data-parallel-replicas). |
| `graphs`                             | boolean  | `false`             | CUDA graph capture for greedy decode. `run` and `generate` read it. `serve` logs a warning when it is set.                   |
| `mmproj`                             | path     | unset               | Qwen3-VL mmproj GGUF for image inputs                                                                                        |
| `image_min_tokens`                   | integer  | unset               | Fewest output tokens per image. Unset keeps the mmproj value.                                                                |
| `image_max_tokens`                   | integer  | unset               | Most output tokens per image. Unset keeps the mmproj value.                                                                  |
| `decision.family`                    | string   | detected            | `jebadiah`, `decider` or `clef` for the start model, else every model, over sidecar, head and GGUF detection.                |
| `decision.config`                    | path     | unset               | Sidecar (`decider_config.json`, `prompt_contract.json`, `jebadiah.json`, or Clef's `joint_head_config.json` with `joint_head.safetensors` beside it) for a model that ships none. Same scope as `family`. |
| `decision.shared_min_tokens`         | integer  | `768`               | Shortest shared prefix one request runs once. Shorter prefixes run as full rows merged with other requests.                  |
| `decision.lane_budget_mb`            | integer  | `8192`              | Lane-state budget of one decision forward pass in MiB. `0` sets none. Kept prefix states count against it.                   |
| `decision.prefix_cache_mb`           | integer  | `2048`              | Device memory in MiB for the prefix states each decision model's worker keeps across requests. `0` keeps none. See [Prefix-state cache](api.md#prefix-state-cache). |
| `decision.max_rows`                  | integer  | `1024`              | Most prompt rows one request expands to. More answers `413`.                                                                 |
| `decision.max_row_tokens`            | integer  | state cap + `4096`  | Most tokens in one row. More answers `413`. Kev's default is its own row limit, `73728`. Clef's is its record limit, `16384`. |
| `decision.max_request_tokens`        | integer  | `1048576`           | Most tokens over all rows of one request. More answers `413`.                                                                |
| `decision.max_queue_rows`            | integer  | `4096`              | Most rows queued or running per model. More answers `503`.                                                                   |
| `decision.batch_wait_ms`             | integer  | `0`                 | Milliseconds the decision worker waits after a request for more to merge.                                                    |
| `decision.allow_contract_mismatch`   | boolean  | `false`             | Serve a Jebadiah model whose prompt contract does not match the bundled renderer.                                            |
| `decision.base`                      | string   | adapter's base      | Base a Kev adapter applies to: a path, or a model name in the model directory. Same scope as `family`.                       |
| `decision.truncate_states`           | boolean  | `false`             | Kev: read the first 65,536 tokens of a longer state instead of answering `422`. Responses then say `truncated`.             |
| `lora`                               | string array | `[]`            | `serve` only, from `config.yaml`: PEFT LoRA adapters of the start model, each `<dir>[=<name>]`. Each one is served as the variant `<model>:<name>`. `--lora` replaces the list. See [LoRA adapters](api.md#lora-adapters). |

Blazr applies no range check to these values in the config code.

### server section

Blazr reads this section from `config.yaml`. `blazr serve`, the swarm leader, and the disaggregated router use it. Flags override `port`, `host`, `tls_cert`, `tls_key`, and `reasoning_format`.

| Key                       | Type           | Default                        | Flag                 | Meaning                                                                                  |
| ------------------------- | -------------- | ------------------------------ | -------------------- | ---------------------------------------------------------------------------------------- |
| `port`                    | integer        | `8080`                         | `--port`             | Port to listen on                                                                        |
| `host`                    | string         | `0.0.0.0`                      | `--host`             | Address to bind to                                                                       |
| `max_concurrent_requests` | integer        | `16`                           | none                 | HTTP requests in flight. Extra requests wait for a slot.                                 |
| `request_timeout_secs`    | integer        | `300`                          | none                 | Request timeout. An exceeded request returns `408`.                                      |
| `cors_enabled`            | boolean        | `true`                         | none                 | Enables CORS headers                                                                     |
| `cors_origins`            | string array   | empty, which allows any origin | none                 | Allowed origins                                                                          |
| `request_logging`         | boolean        | `true`                         | none                 | Logs one line per request                                                                |
| `max_body_size`           | integer, bytes | `10485760`                     | none                 | Request body limit. An exceeded request returns `413`.                                   |
| `tls_cert`                | path           | unset                          | `--tls-cert`         | PEM certificate file                                                                     |
| `tls_key`                 | path           | unset                          | `--tls-key`          | PEM private key file                                                                     |
| `max_inflight_tokens`     | integer        | `0`, which means unlimited     | none                 | In-flight token budget. An exceeded request returns `503`.                               |
| `slo`                     | object         | unset                          | none                 | Latency thresholds. See [SLO thresholds](#slo-thresholds).                               |
| `reasoning_format`        | string         | `deepseek`                     | `--reasoning-format` | Default reasoning split: `deepseek` or `none`. Reloads live.                             |

#### SLO thresholds

Each threshold is optional and in milliseconds. A sample over its threshold logs a warning and adds 1 to `blazr_slo_violations_total`.

| Key                                         | Measures                    |
| ------------------------------------------- | --------------------------- |
| `ttft_p50_ms`, `ttft_p95_ms`, `ttft_p99_ms` | Time to first token         |
| `itl_p50_ms`, `itl_p95_ms`, `itl_p99_ms`    | Inter-token latency         |
| `e2e_p99_ms`                                | End-to-end request latency  |

While the P95 TTFT or P95 ITL is over its threshold, batching admits no sequences beyond the running count. An idle scheduler still admits one sequence. The window holds the last 256 samples, none older than 30 seconds. A change to `server.slo` needs a restart.

### generation section

Blazr reads this section from `config.yaml`. It sets the default for each sampling key. The matching `run`, `generate`, and `chat` flags override it. Requests that omit a value read the live section, and a reload changes it.

Basic sampling:

| Key              | Type         | Default | Meaning                                            |
| ---------------- | ------------ | ------- | -------------------------------------------------- |
| `max_tokens`     | integer      | `2048`  | Maximum tokens to generate                         |
| `temperature`    | float        | `1.0`   | Sampling temperature. `0` selects greedy decoding. |
| `top_p`          | float        | `1.0`   | Nucleus sampling threshold                         |
| `top_k`          | integer      | `0`     | Top-k sampling. `0` disables it.                   |
| `min_p`          | float        | `0.05`  | Min-p threshold. `0` disables it.                  |
| `seed`           | integer      | unset   | Random seed                                        |
| `stop_sequences` | string array | `[]`    | Stop sequences                                     |
| `verbose_prompt` | boolean      | `false` | Prints prompt tokens before generation             |
| `grammar`        | string       | unset   | GBNF grammar for constrained generation            |
| `lora_adapter`   | string       | unset   | `serve`: registered LoRA adapter a request without `lora_adapter` runs on. Unregistered for the model: `404`. See [LoRA adapters](api.md#lora-adapters). |

Penalties:

| Key                 | Type                     | Default | Meaning                                     |
| ------------------- | ------------------------ | ------- | ------------------------------------------- |
| `repeat_penalty`    | float                    | `1.1`   | Repetition penalty. `1.0` disables it.      |
| `repeat_last_n`     | integer                  | `64`    | Recent tokens the repetition penalty covers |
| `frequency_penalty` | float                    | `0.0`   | Penalty scaled by token count               |
| `presence_penalty`  | float                    | `0.0`   | Penalty for tokens seen at all              |
| `logit_bias`        | map of token ID to float | `{}`    | Additive logit bias per token               |

Log probabilities:

| Key            | Type    | Default | Meaning                                                             |
| -------------- | ------- | ------- | ------------------------------------------------------------------- |
| `logprobs`     | boolean | `false` | Returns log probabilities per token                                 |
| `top_logprobs` | integer | `5`     | Top log probabilities per position. Used when `logprobs` is `true`. |

Mirostat and dynamic temperature:

| Key                 | Type    | Default | Meaning                                                         |
| ------------------- | ------- | ------- | --------------------------------------------------------------- |
| `mirostat_mode`     | integer | `0`     | `0` disables. `1` selects Mirostat v1, `2` selects Mirostat v2. |
| `mirostat_tau`      | float   | `5.0`   | Target entropy                                                  |
| `mirostat_eta`      | float   | `0.1`   | Learning rate                                                   |
| `dynatemp_range`    | float   | `0.0`   | Dynamic temperature range. `0` disables it.                     |
| `dynatemp_exponent` | float   | `1.0`   | Entropy to temperature curve exponent                           |

DRY and typical sampling:

| Key                     | Type         | Default | Meaning                                                      |
| ----------------------- | ------------ | ------- | ------------------------------------------------------------ |
| `dry_multiplier`        | float        | `0.0`   | DRY penalty multiplier. `0` disables it.                     |
| `dry_base`              | integer      | `2`     | Minimum n-gram length to penalize                            |
| `dry_allowed_length`    | integer      | `0`     | Tokens scanned back for repeats. `0` scans the full history. |
| `dry_sequence_breakers` | string array | `[]`    | Strings that break n-gram matching                           |
| `typical_p`             | float        | `0.0`   | Typical sampling threshold. `0` disables it.                 |

JSON mode:

| Key         | Type    | Default | Meaning                                             |
| ----------- | ------- | ------- | --------------------------------------------------- |
| `json_mode` | boolean | `false` | Retries up to 3 times when output is not valid JSON |

## CLI flags

Flag short forms come from the clap definitions. A default that names a key takes that key's value when the flag is absent.

When the model is omitted, `run`, `generate`, `chat`, `bench`, and `serve` use `default_model`.

### serve

| Flag                                  | Default               | Meaning                                                                                                                      |
| ------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--model <NAME\|PATH>`          | `default_model`       | Model to load at startup. Blazr loads no model when unset.                                                                   |
| `--chat-template <NAME\|FILE>`        | each model's template | Chat template for every served model. Built-in names: `chatml`, `llama3`, `mistral`, `phi3`, `gemma`, `deepseek`, `generic`. |
| `--port <PORT>`                       | `server.port`, else `8080` | Port to listen on                                                                                                       |
| `--host <HOST>`                       | `server.host`, else `0.0.0.0` | Host to bind to                                                                                                      |
| `--data-parallel <N>`                 | `inference.data_parallel`, else `1` | Copies of the start model, one per device. See [Data-parallel replicas](deployment.md#data-parallel-replicas).   |
| `--api-key <KEY>`                     | `BLAZR_API_KEY`       | API key for authentication. Flag wins over the variable.                                                                     |
| `--api-key-file <PATH>`               | unset                 | File of API keys, one per line. Blank lines and lines starting with `#` are skipped.                                         |
| `--tls-cert <PATH>`                   | `server.tls_cert`     | PEM certificate file. Enables HTTPS with a key from `--tls-key` or `server.tls_key`.                                         |
| `--tls-key <PATH>`                    | `server.tls_key`      | PEM private key file. Needs a certificate from `--tls-cert` or `server.tls_cert`.                                            |
| `--vision-model <NAME=PATH>`          | none                  | Standalone vision embedder for `/v1/embeddings`. Repeatable.                                                                 |
| `--asr-model <NAME=PATH>`             | none                  | Whisper model for `/v1/audio/transcriptions`. `audio` feature. Repeatable.                                                   |
| `--tts-model <NAME=PATH>`             | none                  | TTS model for `/v1/audio/speech`. `audio` feature. Repeatable.                                                               |
| `--tts-max-batch <N>`                 | `8`                   | VoxCPM2 batch width. Rounded down to 1, 2, 4, or 8. `audio` feature.                                                         |
| `--tts-batch-window-ms <MS>`          | `20`                  | Wait after the first VoxCPM2 request. `audio` feature.                                                                       |
| `--tts-no-continuous-batching`        | off                   | Closes the VoxCPM2 batch between steps. `audio` feature.                                                                     |
| `--tts-continuous-capacity <N>`       | `1024`                | Cache positions held per seated VoxCPM2 batch slot. `audio` feature.                                                         |
| `--voice-dir <DIR>`                   | `BLAZR_VOICE_DIR`     | Kokoro voice file directory                                                                                                  |
| `--reasoning-format <deepseek\|none>` | `server.reasoning_format`, else `deepseek` | Default split of reasoning output in chat responses                                                 |
| `--dtype <f32\|bf16\|f16>`             | `inference.dtype`, else `torch_dtype`, else `f32` | Compute dtype of every served model. See [Compute dtype](#compute-dtype).                         |
| `--mmproj <PATH>`                     | mmproj value          | Qwen3-VL mmproj GGUF for image inputs                                                                                        |
| `--image-min-tokens <N>`              | mmproj value          | Fewest output tokens per image                                                                                               |
| `--image-max-tokens <N>`              | mmproj value          | Most output tokens per image                                                                                                 |
| `--decision-family <jebadiah\|decider\|clef>` | `inference.decision.family`, else detected | Family of the start model on `/v1/systemone`                                                   |
| `--decision-config <FILE>`            | `inference.decision.config` | Sidecar of a model that ships none, such as a bare `.gguf`, or a Clef head's `joint_head_config.json`                  |
| `--decision-shared-min-tokens <N>`    | `inference.decision.shared_min_tokens`, else `768` | Shortest shared prefix one request runs once                                                    |
| `--decision-lane-budget-mb <MIB>`     | `inference.decision.lane_budget_mb`, else `8192` | Lane-state budget of one decision forward pass                                                    |
| `--decision-prefix-cache-mb <MIB>`    | `inference.decision.prefix_cache_mb`, else `2048` | Device memory for the prefix states each decision model keeps. `0` keeps none                    |
| `--decision-max-rows <N>`             | `inference.decision.max_rows`, else `1024` | Most prompt rows per request                                                                             |
| `--decision-max-row-tokens <N>`       | `inference.decision.max_row_tokens`, else state cap + `4096` | Most tokens per row                                                                    |
| `--decision-max-request-tokens <N>`   | `inference.decision.max_request_tokens`, else `1048576` | Most tokens per request                                                                    |
| `--decision-max-queue-rows <N>`       | `inference.decision.max_queue_rows`, else `4096` | Most rows queued or running per model                                                             |
| `--decision-batch-wait-ms <MS>`       | `inference.decision.batch_wait_ms`, else `0` | Wait after a request for more to merge                                                                 |
| `--allow-contract-mismatch`           | `inference.decision.allow_contract_mismatch` | Serve a Jebadiah model whose prompt contract does not match                                            |
| `--decision-base <PATH\|NAME>`         | `inference.decision.base`, else the adapter's `base_model_name_or_path` | Base a Kev adapter applies to                                               |
| `--decision-truncate-states`          | `inference.decision.truncate_states` | Kev: read the first 65,536 state tokens instead of answering `422`                                             |
| `--lora <DIR[=NAME]>`                 | `inference.lora`      | PEFT LoRA adapter of the start model, served as the variant `<model>:<name>`. The name defaults to the directory name. Repeatable. Given flags replace `inference.lora`. Needs `--model` or `default_model`. Refused with `--data-parallel` above `1`. qwen35 models only. See [LoRA adapters](api.md#lora-adapters). |

### run and generate

`run` and `generate` take the same flags. `run` takes the model as a positional argument. `generate` takes `-m, --model` and `-p, --prompt`, both required.

| Flag                           | Default                                          | Meaning                                                                                         |
| ------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `-p`, `--prompt <TEXT>`        | unset for `run`, required for `generate`         | Prompt. With a prompt, the command generates once and exits. Without one, `run` opens the REPL. |
| `--chat-template <NAME\|FILE>` | model template, then built-in for the model type | Chat template. `run` and `chat` take this flag. `generate` does not.                            |
| `-v`, `--verbose`              | off                                              | Shows token IDs and timing. `run` and `chat` take this flag. `generate` does not.               |
| `--max-tokens <N>`             | `generation.max_tokens`, else `2048`             | Maximum tokens to generate                                                                      |
| `--temperature <F>`            | `generation.temperature`, else `1.0`             | Sampling temperature. `0` selects greedy decoding.                                              |
| `--top-p <F>`                  | `generation.top_p`, else `1.0`                   | Nucleus sampling threshold                                                                      |
| `--top-k <N>`                  | `generation.top_k`, else `0`                     | Top-k sampling. `0` disables it.                                                                |
| `--min-p <F>`                  | `generation.min_p`, else `0.05`                  | Min-p threshold                                                                                 |
| `--repeat-penalty <F>`         | `generation.repeat_penalty`, else `1.1`          | Repetition penalty. `1.0` disables it.                                                          |
| `--repeat-last-n <N>`          | `generation.repeat_last_n`, else `64`            | Recent tokens the repetition penalty covers                                                     |
| `--gpu-layers <N>`             | `0`                                              | Layers on GPU. `-1` loads all layers on GPU. `0` selects automatically. CUDA path only.         |
| `--cpu`                        | off                                              | Runs on the CPU runtime                                                                         |
| `--num-ctx <N>`                | `num_ctx`, else `2048`                           | Initial KV cache size. The cache grows as needed.                                               |
| `--paged-attention`            | off                                              | Paged KV cache. CUDA path only.                                                                 |
| `--graphs`                     | off                                              | CUDA graph capture for greedy decode. CUDA path only.                                           |
| `--dtype <f32\|bf16\|f16>`      | `inference.dtype`, else `torch_dtype`, else `f32` | Compute dtype of the weights, activations, and KV cache. See [Compute dtype](#compute-dtype). |
| `--mmproj <PATH>`              | mmproj value                                     | Qwen3-VL mmproj GGUF for image inputs                                                           |
| `--image-min-tokens <N>`       | mmproj value                                     | Fewest output tokens per image                                                                  |
| `--image-max-tokens <N>`       | mmproj value                                     | Most output tokens per image                                                                    |
| `--verbose-prompt`             | off                                              | Prints prompt tokens before generation. `generate` only.                                        |
| `--image <PATH>`               | none                                             | Image placed before the prompt. Needs `--mmproj`. Repeatable. `generate` only.                  |
| `--lora <DIR[=NAME]>`          | none                                             | PEFT LoRA adapter, applied unmerged to the loaded qwen35 model. The command decodes with that variant. Once only. `inference.lora` does not apply. |

Rules that apply to `run` and `generate`:

- Without `--cpu`, the command runs on the GPU index from `device`, else device 0 of the build. The GPU is CUDA or wgpu, by build.
- A `device` value for a backend the build lacks exits with an error. The error names `--cpu` and `device: cpu`.
- `--gpu-layers`, `--paged-attention`, and `--graphs` have no effect with `--cpu`.
- `--graphs` needs `--temperature 0`, `--repeat-penalty 1.0`, and no frequency or presence penalty. The default `--repeat-penalty` of `1.1` makes the run ineligible. An ineligible run logs a warning and decodes eagerly.
- `--image` needs `--mmproj` and `--prompt`.

### chat

| Flag                           | Default                                          | Meaning                               |
| ------------------------------ | ------------------------------------------------ | ------------------------------------- |
| `-s`, `--system <TEXT>`        | unset                                            | System prompt                         |
| `--chat-template <NAME\|FILE>` | model template, then built-in for the model type | Chat template                         |
| `--max-tokens <N>`             | `generation.max_tokens`, else `2048`             | Maximum tokens per turn               |
| `--temperature <F>`            | `generation.temperature`, else `1.0`             | Sampling temperature                  |
| `--top-p <F>`                  | `generation.top_p`, else `1.0`                   | Nucleus sampling threshold            |
| `--num-ctx <N>`                | `num_ctx`, else `2048`                           | Context window size                   |
| `-v`, `--verbose`              | off                                              | Shows token IDs and timing            |
| `--dtype <f32\|bf16\|f16>`      | `inference.dtype`, else `torch_dtype`, else `f32` | Compute dtype. See [Compute dtype](#compute-dtype). |
| `--mmproj <PATH>`              | mmproj value                                     | Qwen3-VL mmproj GGUF for image inputs |
| `--image-min-tokens <N>`       | mmproj value                                     | Fewest output tokens per image        |
| `--image-max-tokens <N>`       | mmproj value                                     | Most output tokens per image          |

`chat` has no `--cpu` flag. In a `cuda` build it runs on the CUDA index from `device`. A `device: cpu` value is an error in that build. The `/image <path>` command needs `--mmproj`. Other sampling keys take the `generation` section.

### bench

| Flag                   | Default | Meaning                                                          |
| ---------------------- | ------- | ---------------------------------------------------------------- |
| `--num-ctx <N>`        | `2048`  | Context window size                                              |
| `--decode-tokens <N>`  | `128`   | Tokens generated per run                                         |
| `--runs <N>`           | `3`     | Measurement runs                                                 |
| `--json <PATH>`        | unset   | Writes results as JSON to this file                              |
| `--concurrency <N>...` | unset   | Concurrency levels to sweep, for example `--concurrency 1 2 4 8` |
| `--dtype <f32\|bf16\|f16>` | `inference.dtype`, else `torch_dtype`, else `f32` | Compute dtype. See [Compute dtype](#compute-dtype). |

`bench` uses fixed values that no flag sets. Prompts are 32, 128, and 512 tokens. It runs one warmup run before measuring. Decoding is greedy. Each concurrency level runs its requests one after another.

### list, info, pull, ps, convert, completions

| Command       | Flag or argument                | Default                            | Meaning                                                       |
| ------------- | ------------------------------- | ---------------------------------- | ------------------------------------------------------------- |
| `list`        | `-v`, `--verbose`               | off                                | Shows detailed information                                    |
| `info`        | `<MODEL>`                       | required                           | Model name or path                                            |
| `pull`        | `<REPO>`                        | required                           | Hugging Face repository ID                                    |
| `pull`        | `--file <NAME>`                 | unset                              | Fetches one file from the repository                          |
| `pull`        | `-o`, `--output <DIR>`          | `BLAZR_MODEL_DIR`, then `model_dir`, then `./models` | Output directory                                              |
| `ps`          | `--server <URL>`                | `http://localhost:8080`            | Server to query                                               |
| `convert`     | `<INPUT>`                       | required                           | Input model path                                              |
| `convert`     | `-o`, `--output <PATH>`         | auto-generated                     | Output path                                                   |
| `convert`     | `-f`, `--format <FORMAT>`       | `gguf`                             | Target format: `gguf`, `awq`, `gptq`, `fp8`, or `safetensors` |
| `convert`     | `-q`, `--quantization <METHOD>` | unset                              | Quantization method, for example `q4_k_m`                     |
| `convert`     | `-v`, `--verbose`               | off                                | Debug logs from `compressr`                                   |
| `completions` | `<SHELL>`                       | required                           | Prints shell completions to standard output                   |

`convert` runs the `compressr` binary found on `PATH`.

### decision calibrate

| Flag                  | Default                                    | Meaning                                                                   |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| `-m`, `--model <M>`   | required                                   | Decision model name or path                                               |
| `--labels <FILE>`     | required                                   | JSON-lines label file. See [Fitted temperatures](api.md#fitted-temperatures) |
| `--min-rows <N>`      | `50`                                       | Fewest labelled answers a type needs to be fitted                         |
| `--out <FILE>`        | `blazr_calibration.json` in the sidecar directory | Output file. A vendor sidecar name is refused                      |
| `--dry-run`           | off                                        | Prints the fit and writes nothing                                         |
| `--decision-*`        | as `serve`                                 | Family, sidecar and worker limits, as [serve](#serve) takes them          |

The model loads on the build's runtime and the `device` key's GPU. A label file of records only loads no weights. A Clef model is refused: its head serves no temperature. The label format is in [Fitted temperatures](api.md#fitted-temperatures), and the workflow is in [Decision models](decision-models.md#calibration).

### swarm and disagg (distributed feature)

| Command  | Flag                   | Default             | Meaning                                              |
| -------- | ---------------------- | ------------------- | ---------------------------------------------------- |
| `swarm`  | `--role <ROLE>`        | `worker`            | `leader` or `worker`                                 |
| `swarm`  | `--token <TOKEN>`      | `BLAZR_SWARM_TOKEN` | Shared secret for swarm membership. Required.        |
| `swarm`  | `-m`, `--model <NAME>` | `default_model`     | Model to serve. Leader only.                         |
| `swarm`  | `--leader <ADDR>`      | unset               | Leader address. Worker only.                         |
| `swarm`  | `--swarm-port <PORT>`  | `9090`              | Port for the nexar transport                         |
| `swarm`  | `--port <PORT>`        | `server.port`, else `8080` | HTTP API port. Leader only.                   |
| `swarm`  | `--host <HOST>`        | `server.host`, else `0.0.0.0` | HTTP API host. Leader only.                |
| `swarm`  | `--api-key <KEY>`      | `BLAZR_API_KEY`     | API key for the leader's HTTP API                    |
| `swarm`  | `--mdns`               | off                 | Enables LAN discovery                                |
| `disagg` | `--role <ROLE>`        | `router`            | `router`, `prefill`, or `decode`                     |
| `disagg` | `-m`, `--model <NAME>` | `default_model`     | Model. Required for `prefill` and `decode`.          |
| `disagg` | `--listen-addr <ADDR>` | `0.0.0.0:9190`      | Address this node listens on                         |
| `disagg` | `--router-addr <ADDR>` | unset               | Router address. Required for `prefill` and `decode`. |
| `disagg` | `--prefill <ADDR>...`  | unset               | Prefill worker addresses. Router only.               |
| `disagg` | `--decode <ADDR>...`   | unset               | Decode worker addresses. Router only.                |
| `disagg` | `--port <PORT>`        | `server.port`, else `8080` | HTTP API port. Router only.                   |
| `disagg` | `--host <HOST>`        | `server.host`, else `0.0.0.0` | HTTP API host. Router only.                |

The swarm leader and the disaggregated router take the `server` section. Every role loads its model from `model_dir` on the `device` index.

The narration commands `speak`, `transcribe`, and `sound-review` need the `audio` feature. Their flags live in `src/cli/narration/` and this page does not list them.

## Environment variables

Blazr reads no `.env` file. Export these variables before the command runs.

| Variable              | Default                       | Read by                                                                                  | Meaning                                                                                                                                                               |
| --------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BLAZR_MODEL_DIR`     | `./models`                    | `serve`, `list`, `info`, `pull`, `swarm`, the server pull handler, disaggregated forward | Model directory                                                                                                                                                       |
| `BLAZR_API_KEY`       | unset                         | `serve --api-key`, `swarm --api-key`                                                                        | API key                                                                                                                                                               |
| `BLAZR_SWARM_TOKEN`   | unset                         | `swarm --token`                                                                          | Swarm shared secret                                                                                                                                                   |
| `BLAZR_VOICE_DIR`     | unset                         | `serve --voice-dir`                                                                      | Kokoro voice directory when the flag is unset                                                                                                                         |
| `BLAZR_TP_SIZE`       | unset                         | `serve`                                                                                  | Tensor parallel degree. Overrides `inference.tensor_parallel_size`. A value that is not a whole number is an error. `serve` returns an error above `1` in a build without `cuda` and `nccl`. The scheduler also returns an error above `1` without both features. |
| `BLAZR_TP_RANK`       | unset                         | Scheduler model load, `cuda` and `nccl` builds                                           | Refused. Every rank runs as a thread of one process, so a set value is a load error                                                                                   |
| `BOOSTR_MODELS_DIR`   | unset                         | Model search                                                                             | Model root, searched before the hub caches. Relative paths resolve against the working directory.                                                                     |
| `HF_HUB_CACHE`        | unset                         | Model search                                                                             | Hugging Face hub cache, searched first                                                                                                                                |
| `HF_HOME`             | `~/.cache/huggingface`        | Model search                                                                             | Hugging Face home. Blazr searches `$HF_HOME/hub`.                                                                                                                     |
| `XDG_CACHE_HOME`      | `~/.cache`                    | Model search                                                                             | Cache root. Blazr uses `$XDG_CACHE_HOME/huggingface` when `HF_HOME` is unset.                                                                                         |
| `HF_HUB_OFFLINE`      | unset                         | Model fetch                                                                              | Blocks hub fetches when set to a value other than `0`, `false`, or empty                                                                                              |
| `RUST_LOG`            | `blazr=info,tower_http=debug` | Logging                                                                                  | Log filter. `convert --verbose` sets `RUST_LOG=debug` for `compressr` when this variable is unset.                                                                    |
| `PATH`                | system                        | `convert`                                                                                | Finds the `compressr` binary                                                                                                                                          |
| `MOTION_VIDEO_MODELS` | unset                         | Narration commands                                                                       | Model directory for narration                                                                                                                                         |
| `XDG_CONFIG_HOME`     | `~/.config`                   | Narration commands                                                                       | Root for the narration config file                                                                                                                                    |

The `.env.example` file lists `HF_TOKEN`. No code in `src/` reads that variable.

## Hot reload

Blazr reloads one file while `blazr serve` runs: `~/.blazr/config.yaml`.

- The watcher checks the file's modification time every 5 seconds.
- A change or a new file reloads the whole file. A deleted file resets every key to its default.
- A file that fails to parse leaves the running config unchanged. The watcher logs an error.
- `generation` keys and `server.reasoning_format` apply to the next request that omits a value.
- Every other key needs a restart. The reload logs one warning for each changed key.

Restart `blazr serve` after any other change:

- Changes to `config.json` in the model directory. Blazr reads it when it loads the model.
- Changes to CLI flags.
- Changes to environment variables.

## Related

- [Architecture](architecture.md)

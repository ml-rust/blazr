# blazr config reference

This page lists the config keys, CLI flags, and environment variables that blazr reads.

## Precedence

Blazr applies the first source that sets a value, in this order.

| Order | Source                               | Scope                                                                  |
| ----- | ------------------------------------ | ---------------------------------------------------------------------- |
| 1     | CLI flag                             | Flags listed under [CLI flags](#cli-flags)                             |
| 2     | Environment variable                 | Variables listed under [Environment variables](#environment-variables) |
| 3     | `config.json` in the model directory | The `inference` section and the `dtype` value                          |
| 4     | Built-in default                     | Defaults listed in the tables on this page                             |

The file `~/.blazr/config.yaml` belongs at position 3. No code path applies it. See [User config file](#user-config-file).

Exceptions to the order:

- `--api-key` and `--token` take their value from `BLAZR_API_KEY` and `BLAZR_SWARM_TOKEN` through the clap `env` attribute. A flag on the command line wins over the variable.
- `--voice-dir` wins over `BLAZR_VOICE_DIR`.
- `pull --output` wins over `BLAZR_MODEL_DIR`.
- `--mmproj`, `--image-min-tokens`, and `--image-max-tokens` replace the matching `inference` values from `config.json` when set.
- The `dtype` value from `config.json` is overwritten by `torch_dtype`. See [Model config.json](#model-configjson).

## User config file

Path: `~/.blazr/config.yaml`. Format: YAML, parsed with `serde_saphyr`.

`blazr serve` reads this file at startup. No other command reads it.

A missing file gives the built-in defaults. A file that cannot be read or parsed logs a warning and gives the built-in defaults.

Blazr parses this file. No code path reads its keys in this source tree. A key set in this file changes no behavior.

Example:

```yaml
default_model: mistral-7b
device: cuda
num_ctx: 4096
model_dir: /srv/models
port: 9090
temperature: 0.8
max_tokens: 1024
```

| Key             | Type    | Default | Meaning                      |
| --------------- | ------- | ------- | ---------------------------- |
| `default_model` | string  | unset   | Default model name or path   |
| `device`        | string  | unset   | Device: `cuda` or `cpu`      |
| `num_ctx`       | integer | unset   | Default context window size  |
| `model_dir`     | string  | unset   | Model search directory       |
| `port`          | integer | unset   | Default server port          |
| `temperature`   | float   | unset   | Default sampling temperature |
| `max_tokens`    | integer | unset   | Default maximum tokens       |

## Model config.json

Blazr reads `config.json` from the model directory for SafeTensors checkpoints. Blazr tries each parser in order. A parse error moves to the next parser. Blazr logs nothing when a parser fails.

1. `BlazrConfig`: the top-level model keys, plus the `inference`, `server`, and `generation` sections.
2. `UniversalConfig`: the top-level model keys only.
3. `HuggingFaceConfig`: the Hugging Face keys listed below.
4. Values detected from the checkpoint, when `config.json` is missing or no parser accepts it.

A type error in any blazr section fails parser 1. Parser 2 then reads the file and drops the sections.

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

Blazr reads `torch_dtype` from `config.json` for every SafeTensors load. The result overwrites `inference.dtype`.

| `torch_dtype`               | Resulting dtype |
| --------------------------- | --------------- |
| `bfloat16`                  | `bf16`          |
| `float16`                   | `f16`           |
| `float32`                   | `f32`           |
| Any other value, or missing | `f16`           |

GGUF loads use `f32`.

### inference section

Blazr reads this section from `config.json`. The defaults are the `InferenceConfig` defaults.

| Key                                  | Type             | Default             | Read by code | Meaning                                                                                        |
| ------------------------------------ | ---------------- | ------------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| `weights_path`                       | path             | unset               | No           | Path to model weights                                                                          |
| `device`                             | string or object | `cuda:0`            | No           | Device                                                                                         |
| `max_context_len`                    | integer          | unset               | Yes          | Overrides the model maximum sequence length. Blazr does not compare the two values.            |
| `dtype`                              | string           | `f16`               | Yes          | Inference dtype: `f32`, `f16`, or `bf16`. Overwritten for SafeTensors loads.                   |
| `flash_attention`                    | boolean          | `true`              | No           | Enables flash attention                                                                        |
| `kv_cache`                           | boolean          | `true`              | No           | Enables the KV cache                                                                           |
| `max_batch_size`                     | integer          | `1`                 | Yes          | Batch size. Values of 1 or less use the single-sequence path.                                  |
| `paged_attention`                    | boolean          | `false`             | Yes          | Enables the paged KV cache                                                                     |
| `block_size`                         | integer          | `16`                | Yes          | Tokens per block for paged attention                                                           |
| `num_blocks`                         | integer          | `0`                 | Yes          | KV cache blocks to pre-allocate for paged attention                                            |
| `kv_pool_blocks`                     | integer          | `0`                 | Yes          | Blocks in the shared pool. `0` computes the count from `max_context_len` and `max_batch_size`. |
| `prefix_cache`                       | boolean          | `false`             | Yes          | Reuses KV blocks for repeated prompt prefixes. Requires `paged_attention`.                     |
| `max_cached_blocks`                  | integer          | `10000`             | Yes          | Cached prefix blocks to retain. Used when `prefix_cache` is `true`.                            |
| `gpu_prefix_cache`                   | boolean          | `false`             | Yes          | GPU prefix cache lookup. Falls back to CPU lookup without CUDA.                                |
| `gpu_prefix_cache_ram_tier`          | integer          | `5000`              | Yes          | RAM tier entries for the GPU prefix cache                                                      |
| `prefill_chunk_size`                 | integer          | `0`                 | Yes          | Prompt tokens per prefill chunk. `0` processes the whole prompt at once.                       |
| `speculative`                        | object           | unset               | Yes          | Speculative decoding. Blazr uses it when the section is set.                                   |
| `speculative.draft_model`            | string           | required in section | Yes          | Draft model path. A relative path also resolves against the model directory.                   |
| `speculative.num_speculative_tokens` | integer          | `5`                 | No           | Tokens speculated per step. No reader found.                                                   |
| `speculative.adaptive_depth`         | boolean          | `false`             | No           | Adjusts speculation depth from the acceptance rate. No reader found.                           |
| `tensor_parallel_size`               | integer          | `1`                 | No           | Parsed. No caller reads it. Tensor parallelism reads `BLAZR_TP_SIZE`.                          |
| `moe_offload`                        | string           | unset               | No           | MoE offload strategy                                                                           |
| `moe_gpu_experts`                    | integer          | `0`                 | No           | MoE experts kept on GPU                                                                        |
| `graphs`                             | boolean          | `false`             | No           | CUDA graph capture for greedy decode. Set by `run --graphs`. No other reader found.            |
| `mmproj`                             | path             | unset               | Yes          | Qwen3-VL mmproj GGUF for image inputs                                                          |
| `image_min_tokens`                   | integer          | unset               | Yes          | Fewest output tokens per image. Unset keeps the mmproj value.                                  |
| `image_max_tokens`                   | integer          | unset               | Yes          | Most output tokens per image. Unset keeps the mmproj value.                                    |

Blazr applies no range check to these values in the config code.

### server section

Blazr parses this section from `config.json`. No code path reads it. `blazr serve` builds its server values from CLI flags and the values below.

| Key                       | Type           | Value in `blazr serve`         | Source                         |
| ------------------------- | -------------- | ------------------------------ | ------------------------------ |
| `port`                    | integer        | `8080`                         | `--port`                       |
| `host`                    | string         | `0.0.0.0`                      | `--host`                       |
| `max_concurrent_requests` | integer        | `16`                           | Fixed                          |
| `request_timeout_secs`    | integer        | `300`                          | Fixed                          |
| `cors_enabled`            | boolean        | `true`                         | Fixed                          |
| `cors_origins`            | string array   | empty, which allows any origin | Fixed                          |
| `request_logging`         | boolean        | `true`                         | Fixed                          |
| `max_body_size`           | integer, bytes | `10485760`                     | Fixed                          |
| `tls_cert`                | path           | unset                          | `--tls-cert`                   |
| `tls_key`                 | path           | unset                          | `--tls-key`                    |
| `max_inflight_tokens`     | integer        | `0`, which means unlimited     | Fixed                          |
| `slo`                     | object         | unset                          | Parsed. No code path reads it. |
| `reasoning_format`        | string         | `deepseek`                     | `--reasoning-format`           |

### generation section

Blazr parses this section from `config.json`. No code path reads it. The `run`, `generate`, and `chat` commands take their sampling values from CLI flags.

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
| `lora_adapter`   | string       | unset   | Name of a loaded LoRA adapter                      |

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

Note that `temperature` defaults to `1.0` in this section, but the `run`, `generate`, and `chat` flags default to `0.7`.

## CLI flags

Flag short forms come from the clap definitions. Defaults in parentheses come from the same definitions.

### serve

| Flag                                  | Default               | Meaning                                                                                                                      |
| ------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--model <NAME\|PATH>`          | unset                 | Model to load at startup. Blazr loads no model when unset.                                                                   |
| `--chat-template <NAME\|FILE>`        | each model's template | Chat template for every served model. Built-in names: `chatml`, `llama3`, `mistral`, `phi3`, `gemma`, `deepseek`, `generic`. |
| `--port <PORT>`                       | `8080`                | Port to listen on                                                                                                            |
| `--host <HOST>`                       | `0.0.0.0`             | Host to bind to                                                                                                              |
| `--api-key <KEY>`                     | `BLAZR_API_KEY`       | API key for authentication                                                                                                   |
| `--api-key-file <PATH>`               | unset                 | File of API keys, one per line. Blank lines and lines starting with `#` are skipped.                                         |
| `--tls-cert <PATH>`                   | unset                 | PEM certificate file. Requires `--tls-key`. Enables HTTPS.                                                                   |
| `--tls-key <PATH>`                    | unset                 | PEM private key file. Requires `--tls-cert`.                                                                                 |
| `--vision-model <NAME=PATH>`          | none                  | Standalone vision embedder for `/v1/embeddings`. Repeatable.                                                                 |
| `--asr-model <NAME=PATH>`             | none                  | Whisper model for `/v1/audio/transcriptions`. `audio` feature. Repeatable.                                                   |
| `--tts-model <NAME=PATH>`             | none                  | TTS model for `/v1/audio/speech`. `audio` feature. Repeatable.                                                               |
| `--tts-max-batch <N>`                 | `8`                   | VoxCPM2 batch width. Rounded down to 1, 2, 4, or 8. `audio` feature.                                                         |
| `--tts-batch-window-ms <MS>`          | `20`                  | Wait after the first VoxCPM2 request. `audio` feature.                                                                       |
| `--tts-no-continuous-batching`        | off                   | Closes the VoxCPM2 batch between steps. `audio` feature.                                                                     |
| `--tts-continuous-capacity <N>`       | `1024`                | Cache positions held per seated VoxCPM2 batch slot. `audio` feature.                                                         |
| `--voice-dir <DIR>`                   | `BLAZR_VOICE_DIR`     | Kokoro voice file directory                                                                                                  |
| `--reasoning-format <deepseek\|none>` | `deepseek`            | Default split of reasoning output in chat responses                                                                          |
| `--mmproj <PATH>`                     | mmproj value          | Qwen3-VL mmproj GGUF for image inputs                                                                                        |
| `--image-min-tokens <N>`              | mmproj value          | Fewest output tokens per image                                                                                               |
| `--image-max-tokens <N>`              | mmproj value          | Most output tokens per image                                                                                                 |

### run and generate

`run` and `generate` take the same flags. `run` takes the model as a positional argument. `generate` takes `-m, --model` and `-p, --prompt`, both required.

| Flag                           | Default                                          | Meaning                                                                                         |
| ------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `-p`, `--prompt <TEXT>`        | unset for `run`, required for `generate`         | Prompt. With a prompt, the command generates once and exits. Without one, `run` opens the REPL. |
| `--chat-template <NAME\|FILE>` | model template, then built-in for the model type | Chat template. `run` and `chat` take this flag. `generate` does not.                            |
| `-v`, `--verbose`              | off                                              | Shows token IDs and timing. `run` and `chat` take this flag. `generate` does not.               |
| `--max-tokens <N>`             | `2048`                                           | Maximum tokens to generate                                                                      |
| `--temperature <F>`            | `0.7`                                            | Sampling temperature. `0` selects greedy decoding.                                              |
| `--top-p <F>`                  | `0.9`                                            | Nucleus sampling threshold                                                                      |
| `--top-k <N>`                  | `0`                                              | Top-k sampling. `0` disables it.                                                                |
| `--min-p <F>`                  | `0.05`                                           | Min-p threshold                                                                                 |
| `--repeat-penalty <F>`         | `1.1`                                            | Repetition penalty. `1.0` disables it.                                                          |
| `--repeat-last-n <N>`          | `64`                                             | Recent tokens the repetition penalty covers                                                     |
| `--gpu-layers <N>`             | `0`                                              | Layers on GPU. `-1` loads all layers on GPU. `0` selects automatically. CUDA path only.         |
| `--cpu`                        | off                                              | Runs on CPU                                                                                     |
| `--num-ctx <N>`                | `2048`                                           | Initial KV cache size. The cache grows as needed.                                               |
| `--paged-attention`            | off                                              | Paged KV cache. CUDA path only.                                                                 |
| `--graphs`                     | off                                              | CUDA graph capture for greedy decode. CUDA path only.                                           |
| `--mmproj <PATH>`              | mmproj value                                     | Qwen3-VL mmproj GGUF for image inputs                                                           |
| `--image-min-tokens <N>`       | mmproj value                                     | Fewest output tokens per image                                                                  |
| `--image-max-tokens <N>`       | mmproj value                                     | Most output tokens per image                                                                    |
| `--verbose-prompt`             | off                                              | Prints prompt tokens before generation. `generate` only.                                        |
| `--image <PATH>`               | none                                             | Image placed before the prompt. Needs `--mmproj`. Repeatable. `generate` only.                  |

Rules that apply to `run` and `generate`:

- Without `--cpu`, the command uses CUDA. A build without the `cuda` feature exits with an error that names `--cpu`.
- `--gpu-layers`, `--paged-attention`, and `--graphs` have no effect with `--cpu`.
- `--graphs` needs `--temperature 0`, `--repeat-penalty 1.0`, and no frequency or presence penalty. The default `--repeat-penalty` of `1.1` makes the run ineligible. An ineligible run logs a warning and decodes eagerly.
- `--image` needs `--mmproj` and `--prompt`.

### chat

| Flag                           | Default                                          | Meaning                               |
| ------------------------------ | ------------------------------------------------ | ------------------------------------- |
| `-s`, `--system <TEXT>`        | unset                                            | System prompt                         |
| `--chat-template <NAME\|FILE>` | model template, then built-in for the model type | Chat template                         |
| `--max-tokens <N>`             | `2048`                                           | Maximum tokens per turn               |
| `--temperature <F>`            | `0.7`                                            | Sampling temperature                  |
| `--top-p <F>`                  | `0.9`                                            | Nucleus sampling threshold            |
| `--num-ctx <N>`                | `4096`                                           | Context window size                   |
| `-v`, `--verbose`              | off                                              | Shows token IDs and timing            |
| `--mmproj <PATH>`              | mmproj value                                     | Qwen3-VL mmproj GGUF for image inputs |
| `--image-min-tokens <N>`       | mmproj value                                     | Fewest output tokens per image        |
| `--image-max-tokens <N>`       | mmproj value                                     | Most output tokens per image          |

`chat` has no `--cpu` flag. The device follows the build: CUDA with the `cuda` feature, CPU otherwise. The `/image <path>` command needs `--mmproj`. Other sampling keys take the `GenerationConfig` defaults.

### bench

| Flag                   | Default | Meaning                                                          |
| ---------------------- | ------- | ---------------------------------------------------------------- |
| `--num-ctx <N>`        | `2048`  | Context window size                                              |
| `--decode-tokens <N>`  | `128`   | Tokens generated per run                                         |
| `--runs <N>`           | `3`     | Measurement runs                                                 |
| `--json <PATH>`        | unset   | Writes results as JSON to this file                              |
| `--concurrency <N>...` | unset   | Concurrency levels to sweep, for example `--concurrency 1 2 4 8` |

`bench` uses fixed values that no flag sets. Prompts are 32, 128, and 512 tokens. It runs one warmup run before measuring. Decoding is greedy. Each concurrency level runs its requests one after another.

### list, info, pull, ps, convert, completions

| Command       | Flag or argument                | Default                            | Meaning                                                       |
| ------------- | ------------------------------- | ---------------------------------- | ------------------------------------------------------------- |
| `list`        | `-v`, `--verbose`               | off                                | Shows detailed information                                    |
| `info`        | `<MODEL>`                       | required                           | Model name or path                                            |
| `pull`        | `<REPO>`                        | required                           | Hugging Face repository ID                                    |
| `pull`        | `--file <NAME>`                 | unset                              | Fetches one file from the repository                          |
| `pull`        | `-o`, `--output <DIR>`          | `BLAZR_MODEL_DIR`, then `./models` | Output directory                                              |
| `ps`          | `--server <URL>`                | `http://localhost:8080`            | Server to query                                               |
| `convert`     | `<INPUT>`                       | required                           | Input model path                                              |
| `convert`     | `-o`, `--output <PATH>`         | auto-generated                     | Output path                                                   |
| `convert`     | `-f`, `--format <FORMAT>`       | `gguf`                             | Target format: `gguf`, `awq`, `gptq`, `fp8`, or `safetensors` |
| `convert`     | `-q`, `--quantization <METHOD>` | unset                              | Quantization method, for example `q4_k_m`                     |
| `convert`     | `-v`, `--verbose`               | off                                | Debug logs from `compressr`                                   |
| `completions` | `<SHELL>`                       | required                           | Prints shell completions to standard output                   |

`convert` runs the `compressr` binary found on `PATH`.

### swarm and disagg (distributed feature)

| Command  | Flag                   | Default             | Meaning                                              |
| -------- | ---------------------- | ------------------- | ---------------------------------------------------- |
| `swarm`  | `--role <ROLE>`        | `worker`            | `leader` or `worker`                                 |
| `swarm`  | `--token <TOKEN>`      | `BLAZR_SWARM_TOKEN` | Shared secret for swarm membership. Required.        |
| `swarm`  | `-m`, `--model <NAME>` | unset               | Model to serve. Leader only.                         |
| `swarm`  | `--leader <ADDR>`      | unset               | Leader address. Worker only.                         |
| `swarm`  | `--swarm-port <PORT>`  | `9090`              | Port for the nexar transport                         |
| `swarm`  | `--port <PORT>`        | `8080`              | HTTP API port. Leader only.                          |
| `swarm`  | `--mdns`               | off                 | Enables LAN discovery                                |
| `disagg` | `--role <ROLE>`        | `router`            | `router`, `prefill`, or `decode`                     |
| `disagg` | `-m`, `--model <NAME>` | unset               | Model. Required for `prefill` and `decode`.          |
| `disagg` | `--listen-addr <ADDR>` | `0.0.0.0:9190`      | Address this node listens on                         |
| `disagg` | `--router-addr <ADDR>` | unset               | Router address. Required for `prefill` and `decode`. |
| `disagg` | `--prefill <ADDR>...`  | unset               | Prefill worker addresses. Router only.               |
| `disagg` | `--decode <ADDR>...`   | unset               | Decode worker addresses. Router only.                |
| `disagg` | `--port <PORT>`        | `8080`              | HTTP API port. Router only.                          |

The narration commands `speak`, `transcribe`, and `sound-review` need the `audio` feature. Their flags live in `src/cli/narration/` and this page does not list them.

## Environment variables

| Variable              | Default                       | Read by                                                                                  | Meaning                                                                                                                                                               |
| --------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BLAZR_MODEL_DIR`     | `./models`                    | `serve`, `list`, `info`, `pull`, `swarm`, the server pull handler, disaggregated forward | Model directory                                                                                                                                                       |
| `BLAZR_API_KEY`       | unset                         | `serve --api-key`                                                                        | API key                                                                                                                                                               |
| `BLAZR_SWARM_TOKEN`   | unset                         | `swarm --token`                                                                          | Swarm shared secret                                                                                                                                                   |
| `BLAZR_VOICE_DIR`     | unset                         | `serve --voice-dir`                                                                      | Kokoro voice directory when the flag is unset                                                                                                                         |
| `BLAZR_TP_SIZE`       | `1`                           | `serve`, scheduler model load                                                            | Tensor parallel degree. `serve` returns an error above `1` in a build without `cuda` and `nccl`. The scheduler also returns an error above `1` without both features. |
| `BLAZR_TP_RANK`       | `0`                           | Scheduler model load, `cuda` and `nccl` builds                                           | Tensor parallel rank                                                                                                                                                  |
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
- A change, or the file's creation, reloads all keys of the file.
- No code path reads the reloaded keys. A reload changes no behavior in this source tree.

Restart `blazr serve` after any other change:

- Changes to `config.json` in the model directory. Blazr reads it when it loads the model.
- Changes to CLI flags.
- Changes to environment variables.

## Related

- [Architecture](architecture.md)

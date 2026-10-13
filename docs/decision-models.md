# System One decision models

blazr serves System One decision models on `POST /v1/systemone`. A decision model answers typed questions about one state. It returns a probability for each possible answer. It does not generate text, so a request has no prompt to write and no output to parse.

This page covers the supported families, how to run each one, and the limits. The request and response fields are in [api.md](api.md#system-one-decisions). The config keys and flags are in [configuration.md](configuration.md).

## Question types

| Type     | Asks                                 | Answer                                                          |
| -------- | ------------------------------------ | --------------------------------------------------------------- |
| `choice` | Which keyed option fits the state?   | The most likely key, its confidence, and a probability per key  |
| `noul`   | Is the statement true?               | The probability that it is true                                 |
| `score`  | Which level on the scale fits?       | The expected level, its confidence, the legend, and a probability per level |

The fields of each answer differ by family. [api.md](api.md#post-v1systemone) lists them.

## Families

| Family    | Models                       | Answers read from                                  | Weights                                                        | Notes                                                                 |
| --------- | ---------------------------- | -------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| `jebadiah` | Jebadiah                    | One label token per option                         | SafeTensors or GGUF                                            | Prompt goes inside the model's chat template                          |
| `decider`  | Mapika decider              | One label token per option                         | SafeTensors or GGUF                                            | Plain `Context:` and `Question:` layout by default                    |
| `kev`      | Kev                         | A pointer head over each option's closing token    | LoRA adapter and `head.pt`, plus a base model served separately | Served as a LoRA variant of its base                                  |
| `clef`     | Clef, tested as Clef-Flash  | A joint-schema head over every row of one record   | Full Qwen3.5 checkpoint plus `joint_head_config.json` and `joint_head.safetensors` | Text only. The backbone is large. See [Memory](#memory) |

A request for the schema-first layout is refused. Only the state-first layout is served.

### Memory

| Family | Memory notes                                                                                             |
| ------ | -------------------------------------------------------------------------------------------------------- |
| `kev`  | The base model loads once. The adapter loads unmerged on top of it.                                     |
| `clef` | The Clef-Flash text backbone is 9.0B parameters: 16.7 GiB in BF16 and 34 GiB in F32. It loads on one device. A 12 GB GPU does not hold it. On CPU, the machine needs that much free RAM. CPU runs are slow. |

blazr has no layer offloading for a server model. A backbone must fit one device.

## Files each family needs

| Family     | Required in the model directory                                                                                                  | Optional                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `jebadiah` | `tokenizer.json`, and a chat template (`chat_template.jinja` or `tokenizer_config.json`) for SafeTensors                         | `prompt_contract.json` or `jebadiah.json`. Without one, the model serves with a warning.   |
| `decider`  | `tokenizer.json` for SafeTensors. A chat template only for the `chat` layout.                                                    | `decider_config.json`. Without one, the defaults apply and the model serves with a warning. |
| `kev`      | `adapter_config.json`, `adapter_model.safetensors`, `head.pt`. A base model with `tokenizer.json`, found as described below.      | `blazr_calibration.json`                                                                   |
| `clef`     | `config.json`, `tokenizer.json`, the safetensors shards named by `model.safetensors.index.json`, `joint_head_config.json`, `joint_head.safetensors` | None                                                                                       |

A GGUF model reads its tokenizer and chat template from its header. It needs no `tokenizer.json`.

## Run a family

Set `BLAZR_MODEL_DIR` to the directory that holds your model directories. A request names a model by its directory name. The `--model` flag names the start model. Other models load on the first request that names them.

### jebadiah and decider

```bash
BLAZR_MODEL_DIR=/path/to/models blazr serve --model decider-4b --port 8080
```

Detection needs no flag. Requests that omit `model` use the start model.

For a GGUF file whose header does not name the family, pass the sidecar:

```bash
blazr serve --model /path/to/decider.gguf --decision-config /path/to/decider_config.json --port 8080
```

### kev

```bash
BLAZR_MODEL_DIR=/path/to/models blazr serve --model kev-4b --decision-base /path/to/base-model --port 8080
```

`--decision-base` takes a path, or a model name in the model directory. Without it, blazr reads the adapter's `base_model_name_or_path`. It looks for that base in the model directory and in the Hugging Face hub cache. It does not download a base. A missing base is a `404` that names `--decision-base`.

### clef

```bash
BLAZR_MODEL_DIR=/path/to/models blazr serve --model clef-flash --cpu --port 8080
```

`--cpu` runs the model on the CPU. Without it, blazr uses GPU 0 of the build. The checkpoint's `bfloat16` dtype applies unless you pass `--dtype`.

### Send a request

```bash
curl -s http://localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decider-4b",
    "state": {"ticket": {"subject": "Invoice does not match"}},
    "questions": {
      "route": {"type": "choice", "instructions": "Which team takes this?",
                "criteria": {"billing": "a charge or refund", "support": "a product question"}}
    }
  }'
```

Add `"calibration": "raw"` to the body to skip fitted temperatures. If `--api-key` is set, add the `Authorization: Bearer <api-key>` header. See [api.md](api.md#authentication).

## Model directories and family detection

blazr checks these in order. The first match names the family.

| Order | Check                                                                                                 | Family                        |
| ----- | ----------------------------------------------------------------------------------------------------- | ----------------------------- |
| 1     | A Kev directory: `head.pt`, `adapter_config.json` and `adapter_model.safetensors`, with no base weights | `kev`, whatever else is set |
| 2     | `--decision-family` or `inference.decision.family`                                                    | The named family              |
| 3     | The file name of `--decision-config` or `inference.decision.config`: `decider_config.json`, `prompt_contract.json`, `jebadiah.json`, or `joint_head_config.json` | The family of that name       |
| 4     | `joint_head_config.json` and `joint_head.safetensors` beside the weights                              | `clef`                        |
| 5     | `decider_config.json`, then `prompt_contract.json` or `jebadiah.json`, in the model directory or beside the `.gguf` | `decider` or `jebadiah` |
| 6     | A GGUF header whose `general.basename` is `decider` or `jebadiah`, and whose `general.tags` hold `system-one` or `decision-model` | The named family              |
| 7     | None of the above                                                                                     | Not a decision model: `400` |

The `--decision-family` flag and `--decision-config` apply to the start model when `--model` is set. Without `--model`, they apply to every model.

`GET /v1/models` reports `detected_by` for each model. See [Decision fields in GET /v1/models](#decision-fields-in-get-v1models).

## LoRA variants

Only `qwen35` models take LoRA adapters. Register an adapter on the start model with `--lora`. Each adapter serves as the variant `<model>:<name>`.

```bash
blazr serve --model decider-4b --lora /path/to/adapter=support --port 8080
```

Send `"model": "decider-4b:support"` or `"lora_adapter": "support"` in a request. An adapter detects its family from its base's files. A sidecar in the adapter directory takes precedence. A Kev model is always a variant of its base. Data-parallel start models refuse adapters. See [LoRA adapters](api.md#lora-adapters).

## Calibration

A model ships temperatures in its vendor sidecar. `blazr decision calibrate` fits one temperature per answer type (`choice`, `noul`, `score`) on your labelled data. It writes the fit to `blazr_calibration.json`.

```bash
blazr decision calibrate --model decider-4b --labels labels.jsonl --dry-run
blazr decision calibrate --model decider-4b --labels labels.jsonl
```

The first command prints the fit and writes nothing. The second writes `blazr_calibration.json` beside the model's sidecars.

- A type with fewer than `--min-rows` answers keeps the vendor temperature. The default is `50`.
- Each type in the file replaces the vendor temperature. Types the file does not set keep the vendor value.
- `blazr serve` reads the file at load. A running model reads it again only when it reloads.
- A file with the wrong `schema` or `family` stops the model from loading.
- `calibrate` never writes a vendor sidecar name.
- `calibrate` refuses `clef`. The Clef head returns a plain softmax and serves no temperature.

The label file holds one JSON object per line. The label format and the `calibration: "raw"` request field are in [api.md](api.md#fitted-temperatures).

## Prefix-state cache

Requests about one state often differ only in their questions. The decision worker keeps the model state after the shared prefix on the device. A later request about the same state runs only its questions.

| Family     | Cache                          |
| ---------- | ------------------------------ |
| `jebadiah` | On                             |
| `decider`  | On                             |
| `kev`      | On                             |
| `clef`     | Off. Its head reads the whole record. |

- Config: `inference.decision.prefix_cache_mb` or `--decision-prefix-cache-mb`. The default is `2048` MiB per model. `0` turns the cache off.
- Memory: kept states count against `inference.decision.lane_budget_mb`.
- Usage: `usage.cached_tokens` and `usage.cached_input_tokens` count the prefix tokens a hit saved. Kev responses keep Kev's own `usage` fields.
- Metrics, labelled by `model`: `blazr_decision_prefix_cache_hits_total`, `blazr_decision_prefix_cache_misses_total`, `blazr_decision_prefix_cache_evictions_total`, and the gauge `blazr_decision_prefix_cache_bytes`.
- `GET /v1/models` adds `decision.prefix_cache` while the worker runs.

Key, prefix, and cancel rules are in [api.md](api.md#prefix-state-cache).

## Limits

| Limit                          | Value                                                       |
| ------------------------------ | ----------------------------------------------------------- |
| `decider` choice options       | 255                                                         |
| `decider` score levels         | 2 to 10                                                     |
| `decider` state cap            | `max_state_tokens`, default 32768                           |
| `jebadiah` row tokens          | `max_tokens`, default 2048                                  |
| `kev` options or score levels  | 255                                                         |
| `kev` state                    | 65,536 tokens, delimiter included. Longer is `422`, unless `--decision-truncate-states` is set |
| `kev` row                      | 73,728 tokens                                               |
| `clef` record                  | 16,384 tokens. The state is cut to fit, and `warnings` says so. |

Request limits apply to every family. The defaults are `1024` rows, `4096` queued rows per model, and `1048576` tokens per request. Change them with the `inference.decision.*` keys or the `--decision-*` flags in [configuration.md](configuration.md#inference-section).

### Errors

| Status | `code`                 | Meaning                                                        |
| ------ | ---------------------- | -------------------------------------------------------------- |
| `400`  | `not_a_decision_model` | The model is not a decision model, or takes no LoRA adapter    |
| `404`  | `model_not_found`      | No model or LoRA adapter has that name                         |
| `413`  | `request_too_large`    | The request exceeds a row or token limit                       |
| `422`  | `invalid_request`      | Bad JSON, a refused question, a state over its cap, or a `clef` request with `images` or `videos` |
| `500`  | `internal_error`       | The model did not load, or the run failed                      |
| `503`  | `queue_full`           | The model's queue is full. Retry after the `retry-after` header |
| `503`  | `model_unavailable`    | The model unloaded or its worker stopped. Send the request again |

The `error.message` field describes the failure. See [api.md](api.md#post-v1systemone) for the error body.

## Decision fields in GET /v1/models

`GET /v1/models` adds a `decision` object to each decision model. blazr reads it from the sidecars and the GGUF header. It loads no weights.

| Field                                  | Families          | Meaning                                                    |
| -------------------------------------- | ----------------- | ---------------------------------------------------------- |
| `family`, `readout`, `detected_by`     | All               | The family, how it reads answers, and what named it        |
| `layout`, `max_options`                | All               | Prompt layout and the largest option count                 |
| `max_state_tokens`                     | `decider`, `kev`  | State cap, as listed under [Limits](#limits)               |
| `max_prompt_tokens`                    | `jebadiah`, `kev`, `clef` | Row cap (`jebadiah`, `kev`) or record cap (`clef`)  |
| `context_length`, `license`, `base_model` | All            | Context positions, and the license and base the model names |
| `prompt_contract`                      | `jebadiah`        | Whether the prompt contract checks pass                    |
| `calibration`                          | All               | Fitted temperatures and their source                       |
| `base_path`, `adapter_rank`, `temperature` | `kev`         | Base checkpoint, LoRA rank, and the fitted temperature     |
| `head_width`, `head_hidden_size`, `head_dtype` | `clef`    | Head width, backbone hidden size, and head dtype (`f32`)   |
| `prefix_cache`                         | All with a running worker | Cache state, as listed under [Prefix-state cache](#prefix-state-cache) |

```bash
curl -s http://localhost:8080/v1/models
```

## Tested and not claimed

| Family or area | Status |
| -------------- | ------ |
| `jebadiah`, `decider` | Route tests run the real 4B model on CPU in F32 and from GGUF on CPU. GGUF Q4_K_M on CUDA is tested. SafeTensors on CUDA and GGUF Q8_0 on CUDA are not. |
| `kev` | Fixture tests check rows, positions, and answers without weights. A route test runs a real adapter and base on CPU in F32. CUDA is not tested. |
| `clef` (Clef-Flash) | Fixture tests check records, spans, and the head on CPU. The head alone also matches its reference on CUDA. A route test runs the full model on CPU in BF16 when `BLAZR_CLEF_FULL=1` is set. The full model on CUDA is not tested. |
| Clef 27B | Not tested. Not claimed. |
| Prefix-state cache | Real-model route tests run with the cache on for `jebadiah`, `decider`, and `kev` on CPU. `clef` does not use it. |
| Quantized Clef backbone | Unconfirmed. None is tested. |
| Other families | Not served. Encoder families are not served. |
| Chat completions on a Kev name | Runs the base model with the adapter attached. The output is base-model text. Kev trains no generation head. |
| Images and video | Not served for `clef`. A `clef` request with `images` or `videos` is a `422`. |

Control-token text in a `clef` request stays text. The release reads it as control tokens, so the answers can differ from the release for such a request.

## Related

- [api.md](api.md#system-one-decisions): request, answer, and error fields
- [configuration.md](configuration.md): `inference.decision.*` keys and `--decision-*` flags
- [models.md](models.md): text model architectures and formats

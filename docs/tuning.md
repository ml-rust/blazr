# Performance tuning

blazr exposes tuning knobs as CLI flags and as `inference` keys. Model keys live in `config.json`. Global keys live in `~/.blazr/config.yaml`. This guide maps each knob to its effect, then maps symptoms to knobs.

## Measure first

Measure before you change a knob. Compare runs by tokens/s, TTFT, and ITL. Do not compare the wall-clock time of one request.

### Benchmark one model

Run `blazr bench` for single-model numbers.

| Flag              | Default | Effect                                                   |
| ----------------- | ------- | -------------------------------------------------------- |
| `--num-ctx`       | `2048`  | Sets the context window for the run.                     |
| `--decode-tokens` | `128`   | Sets the tokens generated per run.                       |
| `--runs`          | `3`     | Sets the measurement runs. One warmup run precedes them. |
| `--json`          | none    | Writes the results to the file path given.               |
| `--concurrency`   | none    | Runs a sweep at each listed concurrency level.           |

The sweep runs its requests one after another through a single executor. The sweep does not exercise the batching scheduler. Measure batching with a concurrent HTTP client against `blazr serve`.

### Watch the metrics

`blazr serve` exposes Prometheus metrics at `GET /metrics`.

| Metric                              | Type      | Use                                                 |
| ----------------------------------- | --------- | --------------------------------------------------- |
| `blazr_time_to_first_token_seconds` | histogram | TTFT per request.                                   |
| `blazr_inter_token_latency_seconds` | histogram | ITL between tokens.                                 |
| `blazr_tokens_per_second`           | histogram | Tokens per second per request.                      |
| `blazr_active_decode_slots`         | gauge     | Decode slots in use.                                |
| `blazr_requests_active`             | gauge     | Requests in progress.                               |
| `blazr_inflight_tokens`             | gauge     | Prompt and decode tokens in flight.                 |
| `blazr_tokens_prompted_total`       | counter   | Prompt tokens processed. Use `rate()` for tokens/s. |
| `blazr_tokens_generated_total`      | counter   | Generated tokens. Use `rate()` for tokens/s.        |
| `blazr_prefix_cache_hits_total`     | counter   | Prefix cache hits.                                  |
| `blazr_prefix_cache_misses_total`   | counter   | Prefix cache misses.                                |
| `blazr_queue_depth`                 | gauge     | Requests waiting for a batch slot or an HTTP slot. Rising values mean requests wait for capacity. |
| `blazr_slo_violations_total`        | counter   | Samples over a `server.slo` threshold. Labels `metric` and `percentile`. |

`blazr_token_budget_utilization_ratio` appears only when `server.max_inflight_tokens` is above `0`. `blazr_kv_cache_utilization_ratio` covers the batched scheduler only. `blazr_vram_used_bytes` updates on CUDA builds at each scrape while a model is loaded.

## How to read the knob tables

- A flag is a CLI argument. Flags apply to the command named in the table.
- A key is `inference.<name>` in the `config.json` of a model directory.
- Keys apply to SafeTensors models only. GGUF models take the defaults for every key.
- A SafeTensors `config.json` applies its `inference` block only when the file parses as a blazr config. Otherwise the model's own values fall back to defaults.
- The `inference` section of `config.yaml` overrides the matching key for every model.
- `blazr serve` has no flags for batching, cache, or context. It reads those keys from `config.json` and `config.yaml`.
- `blazr serve` reads the `server` section of `config.yaml`. Its `generation` values apply to requests that omit a value.

## Weights and quantization

| Knob              | Where set                                    | Default          | Effect                                                                                                                                                         |
| ----------------- | -------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compute dtype     | Checkpoint `torch_dtype`                     | `f16` when unset | Sets the SafeTensors dtype: `bfloat16` gives `bf16`, `float16` gives `f16`, `float32` gives `f32`. blazr overwrites `inference.dtype` with this value on load. |
| GGUF quantization | File name, or `blazr convert --quantization` | none             | Sets the block format of the weights, such as `q4_k_m` or `q8_0`. The file fixes the level.                                                                    |
| Convert target    | `blazr convert --format`                     | `gguf`           | Sets the output format. The command delegates to compressr.                                                                                                    |
| AWQ               | Checkpoint `config.json`, key `quant_method` | detected         | A checkpoint with `quant_method` set to `awq` loads as AWQ. No flag selects it.                                                                                |

## Batching

| Knob                     | Where set                                                                 | Default           | Effect                                                                                                                                                                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch size               | `inference.max_batch_size`                                                | `1`               | `1` serves one sequence at a time. Above `1`, batching starts. Non-GDN models need paged attention. GDN models use CUDA lanes.                                                                                                                                             |
| Paged attention          | `inference.paged_attention`, flag `--paged-attention` (`run`, `generate`) | `false`           | Enables the paged continuous batch for non-GDN models. Without it, a non-GDN model serves one sequence at a time, whatever `max_batch_size` says.                                                                                                                          |
| Decode lanes             | `inference.max_batch_size` for GDN models                                 | `max_batch_size`  | Sets the lane count for `qwen35` models. Lanes need a CUDA build. A CPU build serves these models one sequence at a time and logs a warning.                                                                                                                               |
| Token budget             | `inference.max_context_len`                                               | `4096` when unset | Caps the prompt and decode tokens in one scheduled batch. blazr sets the cap from `max_context_len`. No separate key exists.                                                                                                                                               |
| Chunked prefill          | `inference.prefill_chunk_size`                                            | `0` (off)         | Splits a long prompt into chunks of this many tokens. The chunks run between decode steps. The code comment says this reduces TTFT for concurrent requests. Chunking changes the prefill arithmetic in the last bits, so it does not match a one-shot prefill bit for bit. |
| Concurrent HTTP requests | `server.max_concurrent_requests`                                          | `16`              | Limits HTTP requests in flight. Extra requests wait for a slot. Set it in `config.yaml`.                                                                                                                                                                                                            |
| In-flight token cap      | `server.max_inflight_tokens`                                              | `0` (unlimited)   | Caps prompt and decode tokens in flight. An exceeded request returns `503`.                                                                                                                                                                                                             |
| Latency SLO              | `server.slo`                                                              | unset             | Logs warnings and increments `blazr_slo_violations_total` when a threshold breaks. While P95 TTFT or ITL is over target, batch growth stops at the running count, with a floor of one. See [SLO thresholds](configuration.md#slo-thresholds).                                                                                                                                  |

## Prefix cache

| Knob             | Where set                             | Default | Effect                                                                                                                                                                    |
| ---------------- | ------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefix cache     | `inference.prefix_cache`              | `false` | Reuses KV blocks for repeated prompt prefixes. Requires `inference.paged_attention`.                                                                                      |
| Cached blocks    | `inference.max_cached_blocks`         | `10000` | Caps the cached prefix blocks. The batched path uses the larger of this value and one quarter of the block pool.                                                          |
| GPU prefix cache | `inference.gpu_prefix_cache`          | `false` | Mirrors the cache lookup to VRAM with a CUDA kernel. Requires a CUDA build, `prefix_cache`, and `paged_attention`. Falls back to the CPU lookup when CUDA is unavailable. |
| RAM tier size    | `inference.gpu_prefix_cache_ram_tier` | `5000`  | Sets the entries in the warm RAM tier. Entries evicted from VRAM move here first. Read only when `gpu_prefix_cache` is `true`.                                            |

## KV cache and context

| Knob                   | Where set                                | Default    | Effect                                                                                                                                          |
| ---------------------- | ---------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Context length         | `inference.max_context_len`              | unset      | Caps the context. The batched paged path uses `4096` when unset. The executor's shared block pool uses `num_ctx` when unset.                    |
| Initial context        | `--num-ctx` (`run`, `generate`, `chat`, `bench`), `num_ctx` | `2048`     | Sets the initial KV cache allocation. The code comment says the cache grows up to the model maximum. `blazr serve` reads `num_ctx` from `config.yaml` and has no flag. |
| Block size             | `inference.block_size`                   | `16`       | Sets the tokens per KV block in the paged path.                                                                                                 |
| Pool blocks            | `inference.kv_pool_blocks`               | `0` (auto) | Sets the shared pool size for the batched paged path. Auto computes `max_context_len * max_batch_size / block_size`, with a minimum of `256`.   |
| Single-sequence blocks | `inference.num_blocks`                   | `0` (auto) | Sets the block count for the single-sequence paged cache.                                                                                       |
| Paged attention        | `inference.paged_attention`              | `false`    | Selects the paged KV path. Without it, blazr uses the non-paged KV path.                                                                        |

## Speculative decoding

| Knob           | Where set                                      | Default                        | Effect                                                |
| -------------- | ---------------------------------------------- | ------------------------------ | ----------------------------------------------------- |
| Draft model    | `inference.speculative.draft_model`            | required when the block is set | Sets the path or name of the draft model.             |
| Draft length   | `inference.speculative.num_speculative_tokens` | `5`                            | Sets the tokens drafted per iteration.                |
| Adaptive depth | `inference.speculative.adaptive_depth`         | `false`                        | Adjusts speculation depth from the acceptance rate. Passed to boostr `SpeculativeConfig`. |

Speculative decoding runs on the single-sequence path only. The batched paths do not read the block.

## CUDA graphs

| Knob          | Where set                           | Default | Effect                                                                                                                                                     |
| ------------- | ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph capture | flag `--graphs` (`run`, `generate`), `inference.graphs` | off     | Captures the greedy decode pass and replays it. Requires `--temperature 0` and no repeat, frequency, or presence penalty. Otherwise blazr decodes eagerly. |

The CUDA graph path needs a CUDA build. `inference.graphs` in `config.yaml` works like the flag for `run` and `generate`. `serve` does not use graph capture and logs a warning when the key is set.

## Parallelism

| Knob                             | Where set                                                                                        | Default         | Effect                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tensor parallel size             | env `BLAZR_TP_SIZE` (`serve`), `inference.tensor_parallel_size`                                                                    | `1`             | Shards the model across GPUs with NCCL all-reduce. Values above `1` require a build with `--features cuda,nccl`. |
| Tensor parallel rank             | env `BLAZR_TP_RANK`                                                                              | unset           | Refused. blazr runs every rank as a thread of one process. A set value is a load error.                          |
| Data parallel | `inference.data_parallel` or `--data-parallel` (`serve`) | `1` | Loads one copy of the start model per device, each with its own batching engine. Cannot combine with tensor parallel above `1`. See [Data-parallel replicas](deployment.md#data-parallel-replicas). |
| Swarm (pipeline)                 | `blazr swarm --role`, `--leader`, `--swarm-port`, `--port`, `--token` or env `BLAZR_SWARM_TOKEN` | `--role worker` | Assigns model layers to nodes with a pipeline schedule. Requires the `distributed` feature.                      |
| Disaggregated prefill and decode | `blazr disagg --role` (`router`, `prefill`, `decode`)                                            | `router`        | Runs prefill and decode on separate nodes. The router takes `--prefill` and `--decode` worker addresses.         |

## Layer offload to CPU

| Knob       | Where set                               | Default | Effect                                                                                      |
| ---------- | --------------------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| GPU layers | flag `--gpu-layers` (`run`, `generate`) | `0`     | Sets the layers on GPU. `-1` loads the whole model on GPU. `0` uses the offloading default. |

## LoRA adapters

A PEFT LoRA adapter applies at load time as a variant of a loaded `qwen35` base model, named `<model>:<name>`. The variant shares the base weights and attaches the adapter unmerged. It runs on its own executor, so each variant holds its own KV cache and costs KV memory. The base keeps serving as `<model>`. Adapters do not stack. Other model types refuse an adapter.

| Action             | Where set                                                                         | Effect                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Load at start      | flag `--lora <dir>[=<name>]` (`serve`, repeatable), key `inference.lora`          | Builds one variant of the start model per adapter. The name defaults to the adapter directory name. The flag replaces the key.               |
| Load at run time   | `POST /v1/lora` with a JSON body of `model`, `name`, and `path`                   | Registers and builds the variant `<model>:<name>`, loading `model` first when needed. A variant of the same name is replaced.                |
| Select per request | `model: "<model>:<name>"`, or `lora_adapter: "<name>"` with `model: "<model>"`     | Runs the request on that variant. Both fields must name the same adapter when both are set. An unknown adapter is a `404`, never the base.   |
| Default adapter    | key `generation.lora_adapter`                                                     | The adapter a request without `lora_adapter` selects. A `model` that names a variant ignores it.                                              |
| List               | `GET /v1/lora[?model=<model>]`, `GET /v1/models`                                  | Lists the registered variants with base, adapter path, rank and adapted module count.                                                        |
| Unload             | `DELETE /v1/lora/{name}?model=<model>`                                            | Unregisters the variant and drops its executor.                                                                                              |
| One-shot decode    | flag `--lora <dir>[=<name>]` (`run`, `generate`, once)                             | Decodes with the variant in place of the base.                                                                                                |

- Eviction and `keep_alive` drop a variant like any loaded model. A registered variant builds again from its base on its next request. Dropping a base drops its variants first. Only base models count against the loaded-model limit.
- A variant runs one sequence at a time, as any model other than the start model does. It gets no batching engine.
- With `inference.data_parallel` above `1`, the start model takes no adapter: `serve` refuses `--lora`, and `POST /v1/lora` returns `400`.
- A VoxCPM2 TTS bundle's `adapters/<one>.safetensors` is a separate path: the adapter folds into that bundle at load. See `--tts-model`.

## CPU threads

blazr has no flag or key for the CPU thread count.

## Symptom to knob

| Symptom                          | Check first                                                                          | Knobs                                                                                                                                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| High TTFT                        | `blazr_time_to_first_token_seconds`                                                  | `inference.prefill_chunk_size` for long prompts. `inference.prefix_cache` with `inference.paged_attention` for repeated prefixes. `inference.gpu_prefix_cache` on a CUDA build. `inference.max_batch_size` above `1` with `inference.paged_attention` when requests queue. |
| High ITL                         | `blazr_inter_token_latency_seconds`                                                  | `--graphs` on a CUDA build for greedy decode. `inference.speculative.num_speculative_tokens` on the single-sequence path. `inference.prefill_chunk_size` to interleave prompts with decode steps.                                                                          |
| Out of GPU memory                | Device memory from the GPU tools. `blazr_vram_used_bytes` also reports it on CUDA builds.                     | `inference.max_context_len`. `--num-ctx` (`run`, `generate`). `inference.max_batch_size` and `inference.kv_pool_blocks`, which size the pool. `--gpu-layers` (`run`, `generate`).                                                                 |
| Low throughput with many clients | `blazr_active_decode_slots`, `blazr_requests_active`, `blazr_tokens_generated_total` | `inference.max_batch_size` above `1` with `inference.paged_attention`. `inference.kv_pool_blocks` large enough for the batch. `inference.prefix_cache` for shared prefixes. GDN models need a CUDA build for lanes.                                                        |
| Long prompts block short ones    | `blazr_time_to_first_token_seconds`                                                  | `inference.prefill_chunk_size`. `inference.max_batch_size` above `1` with `inference.paged_attention`, so the scheduler runs requests together.                                                                                                                            |
| More than 16 concurrent requests | `blazr_requests_active`                                                              | `server.max_concurrent_requests` sets the limit, and its default is `16`. Run more `blazr serve` processes for more capacity.                                                                                                                                                                                     |

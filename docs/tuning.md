# Performance tuning

blazr exposes tuning knobs as CLI flags and as keys in the `inference` block of a model's `config.json`. This guide maps each knob to its effect, then maps symptoms to knobs.

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

These metrics are registered but no code path updates them in this build:

- `blazr_queue_depth`
- `blazr_token_budget_utilization_ratio`
- `blazr_kv_cache_utilization_ratio`
- `blazr_vram_used_bytes`

Use the device tools for GPU memory. Do not use these gauges for it.

## How to read the knob tables

- A flag is a CLI argument. Flags apply to the command named in the table.
- A key is `inference.<name>` in the `config.json` of a model directory.
- Keys apply to SafeTensors models only. GGUF models take the defaults for every key.
- A SafeTensors `config.json` applies its `inference` block only when the file parses as a blazr config. Otherwise every key takes its default.
- `blazr serve` has no flags for batching, cache, or context. It reads those keys from `config.json` only.
- `blazr serve` builds its server config from defaults. It reads no `server` block.

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
| Concurrent HTTP requests | `server.max_concurrent_requests`                                          | `16`              | Not read by `blazr serve`. The server applies the default `16`.                                                                                                                                                                                                            |
| In-flight token cap      | `server.max_inflight_tokens`                                              | `0` (unlimited)   | Not read by `blazr serve`. The server applies the default `0`.                                                                                                                                                                                                             |
| Latency SLO              | `server.slo`                                                              | unset             | Logs warnings and increments `blazr_slo_violations_total` when a threshold breaks. Does not change scheduling. Not read by `blazr serve`.                                                                                                                                  |

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
| Initial context        | `--num-ctx` (`run`, `generate`, `bench`) | `2048`     | Sets the initial KV cache allocation. The code comment says the cache grows up to the model maximum. `blazr serve` uses `2048` and has no flag. |
| Block size             | `inference.block_size`                   | `16`       | Sets the tokens per KV block in the paged path.                                                                                                 |
| Pool blocks            | `inference.kv_pool_blocks`               | `0` (auto) | Sets the shared pool size for the batched paged path. Auto computes `max_context_len * max_batch_size / block_size`, with a minimum of `256`.   |
| Single-sequence blocks | `inference.num_blocks`                   | `0` (auto) | Sets the block count for the single-sequence paged cache.                                                                                       |
| Paged attention        | `inference.paged_attention`              | `false`    | Selects the paged KV path. Without it, blazr uses the non-paged KV path.                                                                        |

## Speculative decoding

| Knob           | Where set                                      | Default                        | Effect                                                |
| -------------- | ---------------------------------------------- | ------------------------------ | ----------------------------------------------------- |
| Draft model    | `inference.speculative.draft_model`            | required when the block is set | Sets the path or name of the draft model.             |
| Draft length   | `inference.speculative.num_speculative_tokens` | `5`                            | Sets the tokens drafted per iteration.                |
| Adaptive depth | `inference.speculative.adaptive_depth`         | `false`                        | No effect. blazr parses the key and no code reads it. |

Speculative decoding runs on the single-sequence path only. The batched paths do not read the block.

## CUDA graphs

| Knob          | Where set                           | Default | Effect                                                                                                                                                     |
| ------------- | ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph capture | flag `--graphs` (`run`, `generate`) | off     | Captures the greedy decode pass and replays it. Requires `--temperature 0` and no repeat, frequency, or presence penalty. Otherwise blazr decodes eagerly. |

The CUDA graph path needs a CUDA build. The key `inference.graphs` has no reader, so set the flag instead.

## Parallelism

| Knob                             | Where set                                                                                        | Default         | Effect                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tensor parallel size             | env `BLAZR_TP_SIZE` (`serve`)                                                                    | `1`             | Shards the model across GPUs with NCCL all-reduce. Values above `1` require a build with `--features cuda,nccl`. |
| Tensor parallel rank             | env `BLAZR_TP_RANK`                                                                              | `0`             | Sets this process's rank within the tensor parallel group.                                                       |
| Swarm (pipeline)                 | `blazr swarm --role`, `--leader`, `--swarm-port`, `--port`, `--token` or env `BLAZR_SWARM_TOKEN` | `--role worker` | Assigns model layers to nodes with a pipeline schedule. Requires the `distributed` feature.                      |
| Disaggregated prefill and decode | `blazr disagg --role` (`router`, `prefill`, `decode`)                                            | `router`        | Runs prefill and decode on separate nodes. The router takes `--prefill` and `--decode` worker addresses.         |

## Layer offload to CPU

| Knob       | Where set                               | Default | Effect                                                                                      |
| ---------- | --------------------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| GPU layers | flag `--gpu-layers` (`run`, `generate`) | `0`     | Sets the layers on GPU. `-1` loads the whole model on GPU. `0` uses the offloading default. |

## LoRA adapters

| Action             | Where set                                                       | Effect                                                                                                  |
| ------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Load               | `POST /v1/lora` with a JSON body of `model`, `name`, and `path` | Registers an adapter with a model. The body fields come from `LoadLoraRequest` in `src/server/lora.rs`. |
| Select per request | Request field `lora_adapter` on a generation request            | Applies the named adapter to that request. A request without the field uses the base weights.           |
| List               | `GET /v1/lora?model=<model>`                                    | Lists the adapters loaded for a model.                                                                  |
| Unload             | `DELETE /v1/lora/{name}?model=<model>`                          | Removes an adapter from a model.                                                                        |

## CPU threads

blazr has no flag or key for the CPU thread count.

## Symptom to knob

| Symptom                          | Check first                                                                          | Knobs                                                                                                                                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| High TTFT                        | `blazr_time_to_first_token_seconds`                                                  | `inference.prefill_chunk_size` for long prompts. `inference.prefix_cache` with `inference.paged_attention` for repeated prefixes. `inference.gpu_prefix_cache` on a CUDA build. `inference.max_batch_size` above `1` with `inference.paged_attention` when requests queue. |
| High ITL                         | `blazr_inter_token_latency_seconds`                                                  | `--graphs` on a CUDA build for greedy decode. `inference.speculative.num_speculative_tokens` on the single-sequence path. `inference.prefill_chunk_size` to interleave prompts with decode steps.                                                                          |
| Out of GPU memory                | Device memory from the GPU tools. The VRAM gauge is not emitted.                     | `inference.max_context_len`. `--num-ctx` (`run`, `generate`). `inference.max_batch_size` and `inference.kv_pool_blocks`, which size the pool. `--gpu-layers` (`run`, `generate`). `inference.moe_offload`.                                                                 |
| Low throughput with many clients | `blazr_active_decode_slots`, `blazr_requests_active`, `blazr_tokens_generated_total` | `inference.max_batch_size` above `1` with `inference.paged_attention`. `inference.kv_pool_blocks` large enough for the batch. `inference.prefix_cache` for shared prefixes. GDN models need a CUDA build for lanes.                                                        |
| Long prompts block short ones    | `blazr_time_to_first_token_seconds`                                                  | `inference.prefill_chunk_size`. `inference.max_batch_size` above `1` with `inference.paged_attention`, so the scheduler runs requests together.                                                                                                                            |
| More than 16 concurrent requests | `blazr_requests_active`                                                              | No knob. `blazr serve` fixes the HTTP limit at `16`. Run more `blazr serve` processes.                                                                                                                                                                                     |

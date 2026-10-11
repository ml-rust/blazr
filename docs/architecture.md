# blazr architecture

blazr is an LLM inference server with a CLI and an HTTP API. boostr and numr do the tensor work.

## Layers

```text
+---------------------------------------------------------+
| blazr    CLI, HTTP server, config, model lifecycle,     |
|          request scheduling, sampling loop, streaming   |
+----------------------------+----------------------------+
                             |
+----------------------------v----------------------------+
| boostr   model architectures, SafeTensors and GGUF      |
|          loaders, quant kernels, KV and SSM state,      |
|          continuous batching, distributed primitives    |
+----------------------------+----------------------------+
                             |
+----------------------------v----------------------------+
| numr     Tensor<R>, Runtime, device backends            |
+---------------------------------------------------------+
```

blazr holds no tensor ops, model layers, or quant kernels. Those live in boostr. blazr calls boostr through its types and traits.

## Request path

1. The route table in `api_routes` (`src/server/routes.rs`) sends the request to a function, such as `chat_completions` in `src/server/chat.rs`.
2. The function calls `Scheduler::get_executor` (`src/engine/scheduler.rs`) through `state.route` (`src/server/handlers/route.rs`). The route returns the executor and scheduler for the request's model.
3. With a replica set, the router picks one replica. With a request scheduler set, `generate_via_scheduler` (`src/server/generation.rs`) submits to `RequestScheduler::submit`. Without one, the function calls the executor directly.
4. `BatchEngine::run` (`src/engine/batch_engine.rs`) takes batches from `RequestScheduler::schedule_batch` and runs prefill and decode steps.
5. `Executor::generate` (`src/engine/executor_generate.rs`) runs the forward pass through the boostr `LoadedModel`.
6. The sampler (`src/engine/sampling.rs`) picks each token. Grammar masks from `src/engine/grammar.rs` apply when the request sets a grammar.
7. Tokens return on a channel. `create_chat_stream` (`src/server/streaming.rs`) sends them as SSE. `generate_text` (`src/engine/generate_text.rs`) collects them into one response.

## Module map

| Module        | Job                                                                                         | Key files                                                                                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cli`         | Defines the `blazr` subcommands and runs them                                               | `commands.rs` (`Cli`, `Commands`), `serve.rs`, `serve_batching.rs`, `serve_replicas.rs`, `load_settings.rs`, `run.rs`, `chat.rs`, `swarm.rs`, `disaggregated.rs`, `narration/` (`audio`)                           |
| `config`      | Holds the config types and resolves each value from flag, env, file, and default            | `file.rs` (`FileConfig`, `~/.blazr/config.yaml`), `resolve.rs`, `resolve_serve.rs`, `resolve_cluster.rs`, `env.rs`, `overlay.rs`, `reload.rs`, `live.rs`, `blazr.rs`, `server.rs`, `generation.rs`, `inference.rs`, `vision.rs` |
| `engine`      | Schedules models and requests, runs the executor, samples tokens, and drives parallel modes | `scheduler.rs`, `executor*.rs`, `request_scheduler.rs`, `batch_engine.rs`, `lane_*.rs`, `sampling.rs`, `grammar*.rs`, `speculative.rs`, `tensor_parallel.rs`, `data_parallel.rs`, `cache_router.rs`, `latency_window.rs`, `slo_policy.rs`, `queue_depth.rs`, `keep_alive.rs`, `warmup.rs` |
| `loader`      | Detects the model format and source and loads weights                                       | `detect.rs`, `api.rs`, `gguf.rs`, `safetensors/`, `vision.rs`, `hub/` (`audio`)                                                                                                                                 |
| `model`       | Detects architectures and renders chat templates                                            | `detect.rs`, `config.rs`, `chat_template/`                                                                                                                                                                      |
| `server`      | Serves the HTTP API, API-key auth, and SSE streams                                          | `routes.rs`, `startup.rs`, `handlers/` (`route.rs`), `concurrency.rs`, `scheduled.rs`, `config_watch.rs`, `slo.rs`, `chat.rs`, `completions.rs`, `anthropic.rs`, `streaming.rs`, `management/`                    |
| `tokenizer`   | Loads the splintr BPE tokenizer for a vocabulary                                            | `splintr_tokenizer.rs`, `mod.rs`                                                                                                                                                                                |
| `distributed` | Re-exports boostr's distributed inference API under the `distributed` feature               | `mod.rs`                                                                                                                                                                                                        |
| `tls`         | Installs the rustls crypto provider and serves HTTPS                                        | `src/tls.rs`, `src/server/tls.rs`                                                                                                                                                                               |

## Config resolution

`~/.blazr/config.yaml` holds the user keys and the `server`, `inference`, and `generation` sections. `FileConfig` in `src/config/file.rs` parses the file. Each value takes the first source that sets it:

| Values                | Order                                                                    |
| --------------------- | ------------------------------------------------------------------------ |
| Most values           | Flag, then environment variable, then config file, then built-in default |
| `inference.*`         | Flag, then environment (`BLAZR_TP_SIZE` only), then config file, then model `config.json`, then built-in default |

- `src/config/resolve.rs` applies the order for each value. `resolve_serve.rs` and `resolve_cluster.rs` apply it to `blazr serve` and to `swarm` and `disagg`.
- `src/config/overlay.rs` holds the `inference` keys that the config file sets. They apply over every model's `config.json`.
- `src/config/reload.rs` lists the keys that apply on a hot reload. Every `generation` key and `server.reasoning_format` apply live. All other keys need a restart.
- `src/server/config_watch.rs` polls the file while `blazr serve` runs. A file that fails to parse keeps the running config. `src/config/live.rs` swaps in each parsed file as a whole.
- An unknown key logs a warning and has no effect.

## Model lifecycle

### Load

1. `Scheduler::get_executor` returns a cached executor. On a miss, it calls `load_model`.
2. `ensure_capacity` removes least recently used models until one more model fits.
3. `find_model_path` resolves the model name in the model directory.
4. The tensor-parallel size comes from `BLAZR_TP_SIZE`, then `inference.tensor_parallel_size`, then the model `config.json`, then `1`. `validate_tp_config` checks it. When the size is above 1, the scheduler loads a tensor-parallel shard with `loader::load_model_tp`. That path needs the `cuda` and `nccl` features.
5. Otherwise `detect_model_source` picks the format from a directory or a `.safetensors` or `.gguf` path.
6. `from_vocab_size` builds the tokenizer from the vocabulary size. `ChatTemplate::detect` picks the chat template.

For SafeTensors, `detect_architecture_from_loader` (`src/loader/safetensors/detect_arch.rs`) reads tensor names and shapes to identify the architecture.

### Keep and unload

- Loaded models stay in the scheduler map until removed.
- A request's `keep_alive` sets a per-model TTL. `parse_keep_alive` reads `"0"`, `"-1"`, `"5m"`, `"1h"`, `"30s"`, and bare seconds.
- The reaper task calls `evict_expired` every 10 seconds (`src/server/startup.rs`).
- `Scheduler::unload` removes a model by name.
- A keep-alive of `0` removes the model at once.

### Eviction rule

When a load needs room, the scheduler removes the least recently used model. The scheduler records the last access time on each `get_executor` hit. The default `max_loaded` is 1. No path in this tree calls `with_max_loaded`, so the scheduler holds one model at a time. Keep-alive expiry runs separately from this rule.

### Batch path selection

`batching_choice` (`src/cli/serve_batching.rs`) picks the scheduler that `blazr serve` runs behind the handlers.

| Condition                            | Path            | Scheduler                                         |
| ------------------------------------ | --------------- | ------------------------------------------------- |
| `inference.max_batch_size` is 0 or 1 | Single sequence | None. Handlers call the executor.                 |
| Model needs GDN state                | Lanes           | `LaneScheduler` and `LaneEngine` (`cuda` feature) |
| Other model with paged attention on  | Paged batch     | `RequestScheduler` and `BatchEngine`              |
| Other model with paged attention off | Single sequence | None                                              |

## State per model kind

The executor picks the state type by model capability. GDN state is checked first, then SSM state, then the KV cache.

| Model kind                               | State                                                                                                                 | Source                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Attention layers (Llama family, Mistral) | `LayeredKvCache` on the single-sequence path. `LayeredPagedKvCache` with `CpuBlockAllocator` on the paged batch path. | `src/engine/executor_generate.rs`, `src/cli/serve_batching.rs` |
| Mamba2                                   | `LayeredSsmState`, created for each generation call                                                                   | `src/engine/executor_generate.rs`                              |
| qwen35 (GDN layers)                      | `LayeredGdnState` for GDN layers, and a KV cache sized for the attention layers only                                  | `src/engine/executor_generate_qwen35.rs`                       |

## Distributed modes

### Tensor parallel

Tensor parallelism shards Llama-family SafeTensors weights across ranks. Set `BLAZR_TP_SIZE` and `BLAZR_TP_RANK` for each process. `init_tp` in `src/engine/tensor_parallel.rs` sets up the rank state and the NCCL communicator. `loader::load_model_tp` (`src/loader/api.rs`) loads the shard. The `cuda` and `nccl` features compile this path.

### Data parallel

`blazr serve` builds a `DataParallelGroup` (`src/engine/data_parallel.rs`) from the start model. `load_serving` in `src/cli/serve_replicas.rs` loads one copy per replica. The replica count comes from `--data-parallel`, then `inference.data_parallel`, then `1`.

- Replica `i` runs on CUDA device `device + i` on a CUDA build. On a CPU build, each replica is a CPU copy.
- Each replica has its own batching scheduler.
- `blazr serve` refuses a replica count of `0`.
- `blazr serve` refuses a replica count above `1` with no start model.
- `blazr serve` refuses a replica count above `1` with tensor parallelism above `1`.

The five generation handlers call `state.route` for each request. Without a replica set, `route` returns the loaded executor. With replicas, `CacheAwareRouter` (`src/engine/cache_router.rs`) picks the replica that served the longest known prompt prefix. A request with no known prefix goes to the least-loaded replica.

### Latency SLO and queue metrics

- `src/server/slo.rs` checks TTFT, ITL, and end-to-end latency against `server.slo`. A breach adds to `blazr_slo_violations_total{metric,percentile}`.
- `src/engine/slo_policy.rs` holds batch growth while the rolling P95 TTFT or ITL exceeds its target. Running sequences keep decoding.
- `src/engine/queue_depth.rs` feeds `blazr_queue_depth`. `src/server/concurrency.rs` limits HTTP requests with `server.max_concurrent_requests`. `/health` and `/metrics` skip that limit.

### Pipeline (swarm)

`blazr swarm` (`src/cli/swarm.rs`) starts or joins a swarm with a leader and workers. Its host, port, API key, model directory, and device resolve through the config file. Each worker runs one pipeline stage. Hidden states pass between stages over nexar transport. The activation wire format lives in `src/cli/swarm_forward.rs`. The command needs the `distributed` feature.

### Prefill and decode disaggregation

`blazr disagg` runs one of three roles: `router`, `prefill`, or `decode` (`src/cli/disaggregated.rs`). Prefill workers run prompt passes and send the KV cache. Decode workers receive it and run decode steps. The router accepts HTTP requests and dispatches them to the workers. The wire protocol lives in `crate::distributed::disaggregated`, which re-exports boostr. The command needs the `distributed` feature.

## Feature flags

There is no `cpu` feature. The CPU runtime is the default when `cuda` is off.

| Feature       | Default | Enables                                                                                                                                                           |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `f16`         | Yes     | `boostr/f16`                                                                                                                                                      |
| `cuda`        | No      | `boostr/cuda`, and `voxr-narrate/cuda` when `audio` is on                                                                                                         |
| `nccl`        | No      | `boostr/nccl`. Tensor-parallel loading also needs `cuda`.                                                                                                         |
| `distributed` | No      | `boostr/distributed`, the `nexar` dependency, and the `swarm` and `disagg` commands                                                                               |
| `audio`       | No      | `voxr` and `voxr-narrate`, Kokoro, VoxCPM2, VITS, Whisper, and CLAP support, the `/v1/audio/*` routes, and the `speak`, `transcribe`, and `sound-review` commands |

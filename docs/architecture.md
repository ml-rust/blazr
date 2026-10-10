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
2. The function calls `Scheduler::get_executor` (`src/engine/scheduler.rs`). That returns a loaded `Executor`, or loads the model on a miss.
3. With a request scheduler set, `generate_via_scheduler` (`src/server/generation.rs`) submits to `RequestScheduler::submit`. Without one, the function calls the executor directly.
4. `BatchEngine::run` (`src/engine/batch_engine.rs`) takes batches from `RequestScheduler::schedule_batch` and runs prefill and decode steps.
5. `Executor::generate` (`src/engine/executor_generate.rs`) runs the forward pass through the boostr `LoadedModel`.
6. The sampler (`src/engine/sampling.rs`) picks each token. Grammar masks from `src/engine/grammar.rs` apply when the request sets a grammar.
7. Tokens return on a channel. `create_chat_stream` (`src/server/streaming.rs`) sends them as SSE. `generate_text` (`src/engine/generate_text.rs`) collects them into one response.

## Module map

| Module        | Job                                                                                         | Key files                                                                                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cli`         | Defines the `blazr` subcommands and runs them                                               | `commands.rs` (`Cli`, `Commands`), `serve.rs`, `serve_batching.rs`, `run.rs`, `chat.rs`, `swarm.rs`, `disaggregated.rs`, `narration/` (`audio`)                                                                 |
| `config`      | Holds the config types for server, generation, inference, vision, and the user file         | `blazr.rs` (`BlazrConfig`), `server.rs` (`ServerConfig`), `generation.rs` (`GenerationConfig`), `inference.rs`, `user.rs` (`~/.blazr/config.yaml`), `vision.rs`                                                 |
| `engine`      | Schedules models and requests, runs the executor, samples tokens, and drives parallel modes | `scheduler.rs`, `executor*.rs`, `request_scheduler.rs`, `batch_engine.rs`, `lane_*.rs`, `sampling.rs`, `grammar*.rs`, `speculative.rs`, `tensor_parallel.rs`, `data_parallel.rs`, `moe_offload.rs`, `warmup.rs` |
| `loader`      | Detects the model format and source and loads weights                                       | `detect.rs`, `api.rs`, `gguf.rs`, `safetensors/`, `vision.rs`, `hub/` (`audio`)                                                                                                                                 |
| `model`       | Detects architectures and renders chat templates                                            | `detect.rs`, `config.rs`, `chat_template/`                                                                                                                                                                      |
| `server`      | Serves the HTTP API, API-key auth, and SSE streams                                          | `routes.rs`, `startup.rs`, `handlers/`, `chat.rs`, `completions.rs`, `anthropic.rs`, `streaming.rs`, `management/`                                                                                              |
| `tokenizer`   | Loads the splintr BPE tokenizer for a vocabulary                                            | `splintr_tokenizer.rs`, `mod.rs`                                                                                                                                                                                |
| `distributed` | Re-exports boostr's distributed inference API under the `distributed` feature               | `mod.rs`                                                                                                                                                                                                        |
| `tls`         | Installs the rustls crypto provider and serves HTTPS                                        | `src/tls.rs`, `src/server/tls.rs`                                                                                                                                                                               |

## Model lifecycle

### Load

1. `Scheduler::get_executor` returns a cached executor. On a miss, it calls `load_model`.
2. `ensure_capacity` removes least recently used models until one more model fits.
3. `find_model_path` resolves the model name in the model directory.
4. When `BLAZR_TP_SIZE` is above 1, the scheduler loads a tensor-parallel shard with `loader::load_model_tp`. That path needs the `cuda` and `nccl` features.
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

`DataParallelGroup` in `src/engine/data_parallel.rs` runs replicas of one model. It routes requests round-robin, or by prompt prefix with `CacheAwareRouter` in `src/engine/cache_router.rs`. No CLI or server path in this tree constructs the group.

### Pipeline (swarm)

`blazr swarm` (`src/cli/swarm.rs`) starts or joins a swarm with a leader and workers. Each worker runs one pipeline stage. Hidden states pass between stages over nexar transport. The activation wire format lives in `src/cli/swarm_forward.rs`. The command needs the `distributed` feature.

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

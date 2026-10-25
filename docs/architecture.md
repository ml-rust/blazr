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

The engine builds without the CLI and the HTTP server, as a library. The `cli` and `server` features add the two front ends. See [Feature flags](#feature-flags).

## Request path

1. The route table in `api_routes` (`src/server/routes.rs`) sends the request to a function, such as `chat_completions` in `src/server/chat_api/chat.rs`.
2. The function calls `Scheduler::get_executor` (`src/engine/lifecycle/scheduler.rs`) through `state.route` (`src/server/handlers/route.rs`). The route returns the executor and scheduler for the request's model.
3. With a replica set, the router picks one replica. With a request scheduler set, `generate_via_scheduler` (`src/server/serving/scheduled.rs`) or `stream_routed` (`src/server/serving/routed_stream.rs`) submits to `RequestScheduler::submit`. Without one, the function calls the executor directly.
4. `BatchEngine::run_until` (`src/engine/requests/batch_run.rs`) takes batches from `RequestScheduler::schedule_batch` and runs prefill and decode steps, on the `blazr-batch` thread.
5. Without a scheduler, the request's generation runs on its own `blazr-decode` thread (`src/engine/executors/blocking_stream.rs`). There, `Executor::generate` (`src/engine/executors/executor_generate.rs`) runs the forward pass through the boostr `LoadedModel`.
6. The sampler (`src/engine/decoding/sampling/`) picks each token. A constrained request carries a DFA that admission compiled through `GrammarCache` (`src/engine/decoding/grammar_cache.rs`). boostr's device mask applies it each step (`src/engine/executors/sampler_grammar.rs`).
7. Tokens return on a channel. `create_choices_stream` (`src/server/gen_core/chat_choices.rs`) and `create_completion_choices_stream` (`src/server/gen_core/completion_stream.rs`) send them as SSE, one choice after another. `generate_text` (`src/engine/executors/generate_text.rs`) collects them into one response.

## Module map

| Module        | Job                                                                                         | Key files                                                                                                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cli`         | Defines the `blazr` subcommands and runs them. Needs the `cli` feature.                     | `cmdline/` (`commands.rs`: `Cli`, `Commands`; `cluster_args.rs`; `load_settings.rs`), `serving/` (`serve.rs`, `serve_batching.rs`, `serve_replicas.rs`, `ps.rs`), `local/` (`run.rs`, `chat.rs`, `bench.rs`), `catalog/`, `cluster/` (`swarm.rs`, `swarm_scheduler.rs`, `swarm_stage.rs`, `disaggregated.rs`, `mesh.rs`), `narration/` (`audio`) |
| `config`      | Holds the config types and resolves each value from flag, env, file, and default            | `user_file/` (`file.rs`: `FileConfig`, `~/.blazr/config.yaml`; `reload.rs`, `live.rs`), `resolution/` (`resolve.rs`, `resolve_serve.rs`, `resolve_cluster.rs`, `env.rs`), `sections/` (`server.rs`, `generation.rs`, `inference.rs`, `overlay.rs`, `vision.rs`), `blazr.rs` |
| `engine`      | Schedules models and requests, runs the executor, samples tokens, and drives parallel modes | `lifecycle/` (`scheduler.rs`, `keep_alive.rs`), `executors/` (`executor*.rs`, `warmup.rs`, `pooling.rs`), `requests/` (`request_scheduler.rs`, `batch_engine.rs`, `latency_window.rs`, `slo_policy.rs`, `queue_depth.rs`), `lanes/` (`lane_*.rs`), `graphs/`, `decoding/` (`sampling/`, `grammar_cache.rs`), `parallel/` (`tensor_parallel.rs`, `data_parallel.rs`, `cache_router.rs`) |
| `loader`      | Detects the model format and source and loads weights                                       | `detect.rs`, `api.rs`, `formats/` (`gguf.rs`, `safetensors/`), `vision_load/`, `speech_load/` (`audio`), `hub/` (`hub`)                                                                                         |
| `model`       | Detects architectures and renders chat templates                                            | `detect.rs`, `config.rs`, `chat_template/`                                                                                                                                                                      |
| `server`      | Serves the HTTP API, API-key auth, and SSE streams. Needs the `server` feature.             | `routes.rs`, `serving/` (`startup.rs`, `concurrency.rs`, `scheduled.rs`, `config_watch.rs`), `handlers/` (`route.rs`), `chat_api/`, `text_api/`, `anthropic_api/`, `responses_api/`, `gen_core/` (`streaming.rs`, `completion_stream.rs`, `chat_choices.rs`), `media/`, `management/` |
| `tokenizer`   | Loads the splintr BPE tokenizer for a vocabulary                                            | `splintr_tokenizer.rs`, `mod.rs`                                                                                                                                                                                |
| `distributed` | Re-exports boostr's distributed inference API under the `distributed` feature               | `mod.rs`                                                                                                                                                                                                        |
| `runtime`     | Picks the runtime for the build: `cuda`, then `wgpu`, then CPU                              | `src/runtime.rs`, `src/config/device.rs`                                                                                                                                                                        |
| `telemetry`   | Holds metric names, metric recording, and latency SLO checks                                | `names.rs`, `record.rs`, `slo.rs`                                                                                                                                                                               |
| `tls`         | Installs the rustls crypto provider and serves HTTPS                                        | `src/tls.rs`, `src/server/tls.rs`                                                                                                                                                                               |

## Config resolution

`~/.blazr/config.yaml` holds the user keys and the `server`, `inference`, and `generation` sections. `FileConfig` in `src/config/user_file/file.rs` parses the file. Each value takes the first source that sets it:

| Values                | Order                                                                    |
| --------------------- | ------------------------------------------------------------------------ |
| Most values           | Flag, then environment variable, then config file, then built-in default |
| `inference.*`         | Flag, then environment (`BLAZR_TP_SIZE` only), then config file, then model `config.json`, then built-in default |

- `src/config/resolution/resolve.rs` applies the order for each value. `resolve_serve.rs` and `resolve_cluster.rs` apply it to `blazr serve` and to `swarm` and `disagg`.
- `src/config/sections/overlay.rs` holds the `inference` keys that the config file sets. They apply over every model's `config.json`.
- `src/config/user_file/reload.rs` lists the keys that apply on a hot reload. Every `generation` key and `server.reasoning_format` apply live. All other keys need a restart.
- `src/server/serving/config_watch.rs` polls the file while `blazr serve` runs. A file that fails to parse keeps the running config. `src/config/user_file/live.rs` swaps in each parsed file as a whole.
- An unknown key logs a warning and has no effect.

## Model lifecycle

### Load

1. `Scheduler::get_executor` returns a cached executor. On a miss, it calls `load_model`.
2. `ensure_capacity` removes least recently used models until one more model fits.
3. `find_model_path` resolves the model name in the model directory.
4. The tensor-parallel size comes from `BLAZR_TP_SIZE`, then `inference.tensor_parallel_size`, then the model `config.json`, then `1`. `validate_tp_config` checks it. When the size is above 1, the scheduler loads a tensor-parallel shard with `loader::load_model_tp`. That path needs the `cuda` and `nccl` features.
5. Otherwise `detect_model_source` picks the format from a directory or a `.safetensors` or `.gguf` path.
6. `from_vocab_size` builds the tokenizer from the vocabulary size. `ChatTemplate::detect` picks the chat template.

For SafeTensors, `detect_architecture_from_loader` (`src/loader/formats/safetensors/detect_arch.rs`) reads tensor names and shapes to identify the architecture.

### Keep and unload

- Loaded models stay in the scheduler map until removed.
- A request's `keep_alive` sets a per-model TTL. `parse_keep_alive` reads `"0"`, `"-1"`, `"5m"`, `"1h"`, `"30s"`, and bare seconds.
- The reaper task calls `evict_expired` every 10 seconds (`src/server/serving/startup.rs`).
- `Scheduler::unload` removes a model by name.
- A keep-alive of `0` removes the model at once.

### Eviction rule

When a load needs room, the scheduler removes the least recently used model. The scheduler records the last access time on each `get_executor` hit. The default `max_loaded` is 1. No path in this tree calls `with_max_loaded`, so the scheduler holds one model at a time. Keep-alive expiry runs separately from this rule.

### Batch path selection

`batching_choice` (`src/cli/serving/serve_batching.rs`) picks the scheduler that `blazr serve` runs behind the handlers.

| Condition                            | Path            | Scheduler                                         |
| ------------------------------------ | --------------- | ------------------------------------------------- |
| `inference.max_batch_size` is 0 or 1 | Single sequence | None. Handlers call the executor.                 |
| Model needs GDN state                | Lanes           | `LaneScheduler` and `LaneEngine` (`cuda` feature). `paged_attention = true` is an error at load. |
| Other model with paged attention on  | Paged batch     | `RequestScheduler` and `BatchEngine`              |
| Other model with paged attention off | Single sequence | None                                              |

## Threads

A forward pass blocks its thread. No forward pass runs on a Tokio worker, so `/health`, SSE writes and accepts keep running under load.

| Work                       | Thread                                                         | Source                                                                                  |
| -------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Single-sequence generation | One `blazr-decode` thread per request, streamed or not         | `src/engine/executors/blocking_stream.rs`, `decode_thread.rs`, `decode_output.rs`       |
| Paged continuous batch     | One `blazr-batch` thread for the server's life                 | `src/engine/requests/engine_thread.rs`, `batch_run.rs`                                  |
| `qwen35` lanes             | One `blazr-lanes` thread for the server's life (`cuda` feature) | `src/engine/requests/engine_thread.rs`, `src/engine/lanes/lane_engine.rs`               |
| Text and image embeddings  | The Tokio blocking pool                                        | `src/server/text_api/embeddings/handler.rs`                                             |
| Tensor-parallel followers  | One thread per rank above 0                                    | `src/engine/parallel/tp_group.rs`                                                       |

- Each decode and engine thread drives its work on a current-thread Tokio runtime with the timer on. Channels to the server's runtime work as they do between tasks.
- A decode thread builds its stream on that thread. The stream, its KV lease and its tensor-parallel guard never leave it.
- A decode thread sends each item on a bounded channel of 32. It stops within one token after the reader drops the channel.
- A reader that takes no item for `inference.decode_idle_timeout_secs` stops the generation, and the thread exits.
- A panic on a decode thread reaches the client as an error item. The server keeps running.
- The caller of a decode waits for the thread to exit, so its admission permit outlives the decode.
- `server.max_concurrent_requests` bounds the decode threads. The slot is a `WorkHold` (`src/engine/requests/work_hold.rs`). A decode thread holds it until it exits, and a batched request until its scheduler drops the request's state. A caller that leaves first keeps the slot taken until then.
- The scheduler `configure_batching` returns owns its engine thread. When the server state drops the scheduler, the thread gets its stop signal and is joined. The loop reads the signal between steps.
- An engine loop that panics, or ends before its stop signal, fails its scheduler through `FailRequests` (`src/engine/requests/engine_thread.rs`). `RequestRegistry::fail_all` sets the scheduler's `fault` and drops every request's state, so each `WorkHold` drops and its slot frees. Each request then gets one error item and its channel closes. A full channel waits up to five seconds for its reader. Every later submit is refused, and `GET /health` answers `503` with the fault.
- CUDA needs no setup per thread. cudarc makes its context current in every safe call. numr does the same before each raw driver call.
- A WebAssembly target starts no threads. There a generation runs on the caller, with no idle timeout.

## State per model kind

The executor picks the state type by model capability. GDN state is checked first, then SSM state, then the KV cache.

| Model kind                               | State                                                                                                                 | Source                                                                             |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Attention layers (Llama family, Mistral) | `LayeredKvCache` on the single-sequence path. `LayeredPagedKvCache` with `CpuBlockAllocator` on the paged batch path. | `src/engine/executors/executor_generate.rs`, `src/cli/serving/serve_batching.rs` |
| Mamba2                                   | `LayeredSsmState`, created for each generation call                                                                   | `src/engine/executors/executor_generate.rs`                                        |
| qwen35 (GDN layers)                      | `LayeredGdnState` for GDN layers, and a KV cache sized for the attention layers only                                  | `src/engine/executors/executor_generate_qwen35.rs`                                 |

## Distributed modes

### Tensor parallel

Tensor parallelism shards Llama-family SafeTensors weights across ranks. One process runs every rank. Rank r runs on GPU `device + r`. `init_tp` in `src/engine/parallel/tensor_parallel.rs` creates the NCCL group with `Comm::from_devices`, which shares one NCCL unique id. `TpGroup` in `src/engine/parallel/tp_group.rs` runs ranks 1 and up as threads. Each follower thread owns its shard and KV cache. Before each rank-0 forward, the group sends the step to every follower. The token ids reach the followers through an NCCL broadcast. `loader::load_model_tp` (`src/loader/api.rs`) loads each shard. A tensor-parallel model decodes with a contiguous KV cache, one sequence at a time. Paged attention, speculative decoding and the prefix caches are off for it. `BLAZR_TP_RANK` is refused: blazr shares no NCCL id across processes. The `cuda` and `nccl` features compile this path. No multi-GPU run has tested it.

### Data parallel

`blazr serve` builds a `DataParallelGroup` (`src/engine/parallel/data_parallel.rs`) from the start model. `load_serving` in `src/cli/serving/serve_replicas.rs` loads one copy per replica. The replica count comes from `--data-parallel`, then `inference.data_parallel`, then `1`.

- Replica `i` runs on CUDA device `device + i` on a CUDA build. On a CPU build, each replica is a CPU copy.
- Each replica has its own batching scheduler.
- `blazr serve` refuses a replica count of `0`.
- `blazr serve` refuses a replica count above `1` with no start model.
- `blazr serve` refuses a replica count above `1` with tensor parallelism above `1`.

The five generation handlers call `state.route` for each request. Without a replica set, `route` returns the loaded executor. With replicas, `CacheAwareRouter` (`src/engine/parallel/cache_router.rs`) picks the replica that served the longest known prompt prefix. A request with no known prefix goes to the least-loaded replica.

### Latency SLO and queue metrics

- `src/telemetry/slo.rs` checks TTFT, ITL, and end-to-end latency against `server.slo`. A breach adds to `blazr_slo_violations_total{metric,percentile}`.
- `src/engine/requests/slo_policy.rs` holds batch growth while the rolling P95 TTFT or ITL exceeds its target. Running sequences keep decoding.
- `src/engine/requests/queue_depth.rs` feeds `blazr_queue_depth`. `src/server/serving/concurrency.rs` limits HTTP requests with `server.max_concurrent_requests`. A streamed body holds its slot until it drops, and the work the request started holds it until that work ends. `/health` and `/metrics` skip that limit.

### Pipeline (swarm)

`blazr swarm` (`src/cli/cluster/swarm.rs`) starts the leader or joins a worker, one process per node. Its host, port, API key, model directory, and device resolve through the config file. The leader binds nexar's seed with boostr's `SwarmLeader::form_cluster` and holds rank 0. Workers join with `NexarClient::bootstrap_join` and take ranks in arrival order. `PipelineSchedule::by_rank` runs stage `r` on rank `r`, so a worker's previous and next stage follow from its rank. Each message is one frame on its own nexar tag, because nexar does not order two messages to one peer. The command needs the `distributed` feature.

Each rank serves one boostr `StageRunner`, which keeps one KV cache per request id over its layer range only. Each node loads its range with `loader::load_model_layers`, which reads only the tensors boostr's `LayerSpan::reads` names and builds a boostr `LlamaStage`. Every full-model path refuses a `LlamaStage`. The leader's front end is an `Executor::tokenizer_only`, as the disaggregated router's is, and its model scheduler is pinned. `LayerSpan::for_model` refuses a model a range cannot hold before any weight is read. `StageRunner::new` refuses a model with SSM or Gated DeltaNet state and a tensor-parallel shard, so blazr keeps no check of its own. boostr owns the request header, the activation format and the logits frame.

- `swarm_stage.rs`: `stage_config`, `load_stage_model`, `build_stage`, `leader_assignment`, and `LocalStage`, the thread that serves the leader's stage 0. Its frames and releases run in arrival order, so a release never overtakes a frame of its request.
- `swarm_scheduler.rs`: `SwarmScheduler`, a `GenerationScheduler` and `CancelRequest` with one task per request and a semaphore of `--max-requests` slots. A watch on nexar's failure reports marks the swarm down.
- `swarm_request.rs`: one request's task. It runs stage 0, sends the activation with `SwarmLeader::pipeline_forward`, and samples each logits row with `RowSampler` (`src/engine/executors/row_sampler.rs`), which goes through `Executor::sample_begin` and `sample_finish`. `RequestRegistry` renders the text and applies stops. Every exit releases the request on the leader's stage and with `SwarmLeader::release`. A `StageError` fails the request. Any other pipeline error sets the scheduler's `fault`, which `GET /health` reports as `503`.
- `swarm_serving.rs`: the `Serving::Single` replica whose scheduler is the swarm, beside a pinned model scheduler.

The leader stops in this order: the HTTP server, `SwarmScheduler::close`, which ends every request, then `SwarmLeader::shutdown`, which sends the shutdown notice and shuts down the nexar client.

### Cluster formation and shutdown

`src/cli/cluster/mesh.rs` builds each node's nexar `ClusterConfig` from `--listen-addr`, `--advertise-addr`, `--join-timeout` and the token. It turns nexar's formation errors into CLI errors that name the flag to fix. It refuses a seed node that no peer can dial. Every node ends with the nexar shutdown through boostr's `transport::shutdown_shared`, also after an error or a signal, so data a node sent reaches each peer before the connections close.

### Prefill and decode disaggregation

`blazr disagg` runs one of three roles: `router`, `prefill`, or `decode` (`src/cli/cluster/disaggregated.rs`), one process per node. The wire protocol lives in `crate::distributed::disaggregated`, which re-exports boostr. The command needs the `distributed` feature.

- Ranks (`src/cli/cluster/disaggregated_layout.rs`): boostr's `DisaggConfig::standard` puts the router on rank 0, prefill on `1..=prefill-count` and decode after them. A worker requests the rank of its role and `--index`.
- Formation (`src/cli/cluster/disaggregated_mesh.rs`): the router binds the seed on `--router-addr` and forms the cluster with `NexarClient::bootstrap_seed_with`. Each worker joins with `bootstrap_join`. When the HTTP server stops, `DisaggRouter::stop_workers` ends every worker's loop, and each node shuts down its nexar client.

- Router: loads the config, tokenizer and chat template of `--model`, and no weights (`Executor::tokenizer_only`). It serves the full HTTP API through the `blazr serve` startup path. Its model scheduler is pinned: it serves `--model` only and never loads, unloads or evicts a model.
- `DisaggScheduler` (`src/cli/cluster/disaggregated_scheduler.rs`): the router's one `GenerationScheduler`. It turns a request's `GenerationConfig` into boostr's `GenerationParams`, streams the tokens, applies text, stop sequences and finish reasons, and cancels both workers when the request handle drops.
- Prefill worker: runs the prompt, samples the first token and hands the KV cache to a decode worker.
- Decode worker: opens the cache once per request and samples each next token.
- Forward closures (`src/cli/cluster/disaggregated_forward.rs`): run the executor's contiguous-cache forward and return logits. The workers sample. Prefill and decode on two workers give the tokens of one process.
- Refused fields (`src/cli/cluster/disaggregated_params.rs`): the protocol carries no grammar or `response_format` constraint, mirostat, DRY, typical-p, dynamic temperature, `logit_bias` or `logprobs`. A request that sets one is a `400`.
- Refused models: GDN, SSM and tensor-parallel models fail at worker start. Their state does not cross the KV handoff.
- Endpoints with no generation path: embeddings return `501`. Image inputs and LoRA variants are not served.

## Feature flags

There is no `cpu` feature. The CPU runtime runs when neither `cuda` nor `wgpu` is on. A command also takes the CPU with `--cpu` or `device: cpu`.

| Feature       | Default | Enables                                                                                                                                                           |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `f16`         | Yes     | `boostr/f16`                                                                                                                                                      |
| `cli`         | Yes     | The `cli` module and the `blazr` binary: `clap`, `colored`, `rustyline`, `tracing-subscriber`, and `hub`                                                          |
| `server`      | Yes     | The `server` module: axum, tower-http, TLS, `reqwest`, the Prometheus exporter, and `hub`. `blazr serve` and `blazr ps` need it.                                 |
| `hub`         | No      | `hf-hub` and `indicatif`. `cli` and `server` enable it.                                                                                                           |
| `cuda`        | No      | `boostr/cuda`, and `voxr-narrate/cuda` when `audio` is on                                                                                                         |
| `wgpu`        | No      | `boostr/wgpu`: Metal, Vulkan, DX12, and WebGPU. `cuda` takes priority when both are on.                                                                           |
| `nccl`        | No      | `boostr/nccl`. Tensor-parallel loading also needs `cuda`.                                                                                                         |
| `distributed` | No      | `boostr/distributed`, the `nexar` dependency, and the `swarm` and `disagg` commands                                                                               |
| `audio`       | No      | Implies `server` and `cli`. Adds `voxr` and `voxr-narrate`, Kokoro, VoxCPM2, VITS, Whisper, and CLAP support, the `/v1/audio/*` routes, and the `speak`, `transcribe`, and `sound-review` commands |

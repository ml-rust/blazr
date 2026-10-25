# blazr deployment

This guide covers installing blazr, running `blazr serve`, and operating it behind TLS and a reverse proxy.

## Install

### Prebuilt archives

Each [GitHub release](https://github.com/ml-rust/blazr/releases) carries these archives. Desktop archives include the `audio` feature. The `-gpu` and macOS archives add the `wgpu` feature.

| Archive                      | Platform               | Build                               | Built on                                |
| ---------------------------- | ---------------------- | ----------------------------------- | --------------------------------------- |
| `blazr-linux-x64.tar.gz`     | Linux x86_64           | CPU, `audio`                        | `ubuntu-22.04`, glibc 2.35 or newer     |
| `blazr-linux-x64-gpu.tar.gz` | Linux x86_64           | Vulkan (`wgpu`), `audio`            | `ubuntu-22.04`, glibc 2.35 or newer     |
| `blazr-linux-arm64.tar.gz`   | Linux aarch64          | CPU, `audio`                        | `ubuntu-24.04-arm`, glibc 2.39 or newer |
| `blazr-macos-arm64.tar.gz`   | macOS on Apple silicon | Metal (`wgpu`), `audio`             | `macos-latest`                          |
| `blazr-windows-x64.zip`      | Windows x86_64         | CPU, `audio`                        | `windows-latest`                        |
| `blazr-windows-x64-gpu.zip`  | Windows x86_64         | DX12 (`wgpu`), `audio`              | `windows-latest`                        |
| `blazr-android-arm64.tar.gz` | Android arm64          | `libblazr.a` for the C ABI (`capi`) | Built with the `capi` feature           |
| `blazr-ios-arm64.tar.gz`     | iOS arm64              | `libblazr.a` for the C ABI (`capi`) | Built with the `capi` feature           |

Fetch the archive and `SHA256SUMS` from the latest release, then check the archive:

```bash
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/blazr-linux-x64.tar.gz
curl -LO https://github.com/ml-rust/blazr/releases/latest/download/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf blazr-linux-x64.tar.gz
./blazr --version
```

The check prints `blazr-linux-x64.tar.gz: OK`. Stop when it prints `FAILED`, and fetch the archive again.

Other platforms:

- macOS: run `shasum -a 256 --check --ignore-missing SHA256SUMS`.
- Windows: run `Get-FileHash -Algorithm SHA256 blazr-windows-x64.zip`, then compare the hash with the `blazr-windows-x64.zip` line in `SHA256SUMS`. Unpack the zip with `Expand-Archive`.

No prebuilt archive includes CUDA. Create the CUDA binary from source.

### Install with cargo

Installing from source or with `cargo install` requires Rust 1.95 or newer.

```bash
cargo install blazr --locked --features audio          # CPU
cargo install blazr --locked --features audio,cuda     # NVIDIA CUDA
```

The default features are `f16`, `cli`, and `server`. `cargo install` keeps them.

| Feature       | Adds                                                                            |
| ------------- | ------------------------------------------------------------------------------- |
| `audio`       | `/v1/audio/*` routes and the `speak`, `transcribe`, and `sound-review` commands. Implies `server` and `cli`. |
| `cuda`        | CUDA backend. Needs the CUDA toolkit 12.8 or newer, including 13.x.             |
| `wgpu`        | WebGPU backend over Metal, Vulkan, or DX12. Select it with `device: wgpu`.      |
| `distributed` | The `swarm` and `disagg` commands. Also needs `server`. See [Multi-node](#multi-node). |

Create the binary from a source checkout:

```bash
cargo build --release --features audio         # CPU
cargo build --release --features audio,cuda    # CUDA
cargo build --release --features wgpu          # WebGPU
```

The binary is `target/release/blazr`. Run `blazr --version` to list the built features, for example `<version> (audio, cuda)`.

## Run `blazr serve`

### Flags

| Flag                                       | Default                  | Purpose                                                                            |
| ------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------------- |
| `--model`, `-m`                            | `default_model`          | Model to load at startup. A name or a path.                                        |
| `--chat-template`                          | the model's own template | Built-in name or Jinja template file for every served model.                       |
| `--host`                                   | `server.host`, else `0.0.0.0` | Address to bind.                                                              |
| `--port`                                   | `server.port`, else `8080` | Port to bind.                                                                    |
| `--data-parallel`                          | `inference.data_parallel`, else `1` | Copies of the start model, one per device. See [Data-parallel replicas](#data-parallel-replicas). |
| `--api-key`                                | `BLAZR_API_KEY`          | API key. Flag wins over the variable.                                              |
| `--api-key-file`                           | none                     | File of API keys, one per line.                                                    |
| `--tls-cert`                               | `server.tls_cert`        | PEM certificate chain. Needs a key from `--tls-key` or `server.tls_key`.           |
| `--tls-key`                                | `server.tls_key`         | PEM private key. Needs a certificate from `--tls-cert` or `server.tls_cert`.       |
| `--reasoning-format`                       | `server.reasoning_format`, else `deepseek` | Default chat reasoning split: `deepseek` or `none`.                  |
| `--vision-model`                           | none                     | `NAME=PATH` vision embedder for image inputs to `/v1/embeddings`. Repeat for more. |
| `--mmproj`                                 | none                     | Qwen3-VL mmproj GGUF for image inputs to `qwen35` models.                          |
| `--image-min-tokens`, `--image-max-tokens` | the mmproj value         | Token range every image resizes into.                                              |
| `--asr-model`, `--tts-model`               | none                     | `NAME=PATH` Whisper or TTS bundle. Available in `audio` builds only.               |
| `--voice-dir`                              | `$BLAZR_VOICE_DIR`       | Kokoro voice directory. Available in `audio` builds only.                          |

The startup output prints the listen address, the model, and the backend: `Backend: CUDA`, `Backend: wgpu`, or `Backend: CPU`.

### Start a fixed model

Fetch the model before the first start. `blazr serve` does not fetch a model named by `--model`.

```bash
BLAZR_MODEL_DIR=/srv/blazr/models blazr pull Qwen/Qwen2.5-0.5B
BLAZR_MODEL_DIR=/srv/blazr/models blazr list
BLAZR_MODEL_DIR=/srv/blazr/models blazr serve --model <name> --host 127.0.0.1 --port 8080
```

`blazr pull` writes to the model directory when `--output` is absent. Replace `<name>` with a name that `blazr list` shows.

`--model` resolves in this order. Steps 2 to 5 use the model directory:

1. The value as a path.
2. `<model-dir>/<name>`.
3. `<model-dir>/<name>.safetensors`.
4. `<model-dir>/<name>.gguf`.
5. `<model-dir>/<name>/model.safetensors`.

The model directory comes from `BLAZR_MODEL_DIR`, then `model_dir` in `config.yaml`, then `./models`. The default is relative to the working directory of the process.

With `--model`, `blazr serve` loads and warms the model before it binds the port. A load error stops startup.

### Start on demand

Without `--model`, `blazr serve` loads `default_model` when that key is set. Otherwise it starts with no model loaded. Chat, embedding, tokenize, and template requests load the named model on first use.

`blazr serve` keeps one model loaded. A request for a second model evicts the least recently used model. No flag changes this limit.

Requests take an optional `keep_alive` string that sets how long the model stays loaded.

| `keep_alive`            | Effect                                                           |
| ----------------------- | ---------------------------------------------------------------- |
| `"0"`                   | Expires at once. The reaper removes the model within 10 seconds. |
| `"-1"`                  | Never expires.                                                   |
| `"30s"`, `"5m"`, `"1h"` | Expires after that duration.                                     |
| `"300"`                 | Expires after 300 seconds. A bare number counts seconds.         |

A model without `keep_alive` stays loaded until another model evicts it.

### Hugging Face cache and offline mode

The cache lookup checks these directories in order:

1. `$HF_HUB_CACHE`
2. `$HF_HOME/hub`
3. `$XDG_CACHE_HOME/huggingface/hub`
4. `~/.cache/huggingface/hub`

Set `HF_HUB_OFFLINE=1` on hosts without outbound network access. Any non-empty value turns offline mode on, except `0` and `false`, which match case-insensitively. Offline mode blocks the fetch that `blazr serve` makes for Kokoro's G2P files. A `--tts-model` Kokoro load then fails when the cache lacks those files.

### GPU and CPU

The build selects the backend. Priority is `cuda`, then `wgpu`, then the CPU. The `device` key in `config.yaml` selects the device index: `cuda:N` or `wgpu:N`.

- A `cuda` build runs on the CUDA device from `device`, or device 0 when the key is unset.
- A `wgpu` build runs on the adapter from `device`, or adapter 0 when the key is unset.
- A default build runs on the CPU. `blazr serve` has no `--cpu` flag.
- A `cuda` or `wgpu` build refuses `device: cpu` for `serve`. Remove the key or set a GPU index.
- A `wgpu` build exits at startup when no adapter opens at the index. The error names the adapter and the next action.

Check the startup output for `Backend: CUDA`, `Backend: wgpu`, or `Backend: CPU` after each start.

### Data-parallel replicas

Set `inference.data_parallel` in `config.yaml`, or pass `--data-parallel`, to load one copy of the start model per device.

- A CUDA build places copy `i` on CUDA device `device + i`.
- A wgpu build places copy `i` on wgpu adapter `device + i`.
- A CPU build loads one CPU copy for each replica.
- Each copy runs its own batching engine.
- The chat, completions, responses, Anthropic messages, and infill handlers route through the group.
- The router records token prefixes every 16 tokens, up to 1024 tokens.
- A request goes to the replica that holds its longest warm prefix. Otherwise it goes to the least loaded replica.
- A warm replica more than 4 requests above the least loaded replica gives up the request.
- Embeddings and rerank requests run on replica 0.
- `/health` reports the replica count in `data_parallel_replicas`.
- `blazr serve` refuses `0`, and a value above `1` with no start model. It also refuses a value above `1` with `tensor_parallel_size` above `1`. That error names both keys.
- LoRA variants are not replicated. Above `1`, `blazr serve` refuses `--lora` and `inference.lora`, and `POST /v1/lora` for the start model returns `400`.

## Deployment examples

### systemd

Create `/etc/systemd/system/blazr.service`:

```ini
[Unit]
Description=blazr inference server
After=network-online.target
Wants=network-online.target

[Service]
User=blazr
Group=blazr
Environment=BLAZR_MODEL_DIR=/srv/blazr/models
Environment=HF_HOME=/srv/blazr/hf
ExecStart=/usr/local/bin/blazr serve --model llama-3.2-1b --host 127.0.0.1 --port 8080 --api-key-file /etc/blazr/api-keys.txt
KillSignal=SIGTERM
TimeoutStopSec=60
Restart=on-failure
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/srv/blazr/hf

[Install]
WantedBy=multi-user.target
```

These examples are untested against a running deployment.

Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now blazr
journalctl -u blazr -f
```

Notes on the unit:

- The `blazr` user must read `/etc/blazr/api-keys.txt`. Run `chmod 600` on it.
- `ReadWritePaths` keeps the Hugging Face cache writable under `ProtectSystem=strict`.
- The unit binds loopback. The reverse proxy in the nginx example faces the network.
- `KillSignal=SIGTERM` and `TimeoutStopSec` give in-flight requests time to finish. See [Graceful shutdown](#graceful-shutdown).

### Docker (CPU)

Create `Dockerfile`:

```dockerfile
FROM ubuntu:24.04

ARG BLAZR_TAG
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fLO "https://github.com/ml-rust/blazr/releases/download/${BLAZR_TAG}/blazr-linux-x64.tar.gz" \
 && curl -fLO "https://github.com/ml-rust/blazr/releases/download/${BLAZR_TAG}/SHA256SUMS" \
 && sha256sum --check --ignore-missing SHA256SUMS \
 && tar -xzf blazr-linux-x64.tar.gz -C /usr/local/bin blazr \
 && rm blazr-linux-x64.tar.gz SHA256SUMS

ENV BLAZR_MODEL_DIR=/models \
    HF_HOME=/cache/huggingface
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS http://127.0.0.1:8080/health
ENTRYPOINT ["blazr"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
```

The base image has glibc 2.39, which the arm64 archive also needs. `BLAZR_TAG` has no default. Pass a release tag on every build. For an arm64 host, replace `blazr-linux-x64.tar.gz` with `blazr-linux-arm64.tar.gz`. The image runs as root. Add a `USER` line for a non-root run.

```bash
docker build --build-arg BLAZR_TAG=<tag> -t blazr:cpu .
docker run --rm -p 8080:8080 \
  -v /srv/blazr/models:/models:ro \
  -v blazr-hf-cache:/cache/huggingface \
  blazr:cpu serve --model llama-3.2-1b
```

### Docker (CUDA)

No prebuilt CUDA archive ships. Create a CUDA image from source:

- Start from a CUDA development image that matches a toolkit of 12.8 or newer.
- Create the binary inside that image with `cargo build --release --features audio,cuda`.
- Install the NVIDIA Container Toolkit on the host.
- Run the container with `docker run --gpus all`.

### nginx reverse proxy

Terminate TLS at nginx and proxy plain HTTP to `blazr` on loopback. This setup also gives the drain behavior described in [Graceful shutdown](#graceful-shutdown).

```nginx
limit_req_zone $binary_remote_addr zone=blazr:10m rate=10r/s;

server {
    listen 443 ssl;
    http2 on;
    server_name llm.example.com;

    ssl_certificate     /etc/nginx/tls/fullchain.pem;
    ssl_certificate_key /etc/nginx/tls/privkey.pem;

    client_max_body_size 10m;

    location / {
        limit_req zone=blazr burst=20 nodelay;

        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        chunked_transfer_encoding on;
    }
}
```

- `proxy_buffering off` forwards each SSE chunk when blazr sends it.
- `proxy_http_version 1.1` and `chunked_transfer_encoding on` keep the chunked SSE stream intact.
- `proxy_read_timeout` limits idle time between reads. It does not limit total request time.
- `client_max_body_size 10m` matches the blazr default body limit of 10 MB.
- `blazr` has no rate limiter. The `limit_req` directives provide one. Set `rate` and `burst` for your traffic.
- `limit_req_zone` must sit in the `http` context. Place it outside the `server` block, as shown.

## Security

### API keys

Pass keys in one of three ways:

| Source   | Form                    | Behavior                                                                     |
| -------- | ----------------------- | ---------------------------------------------------------------------------- |
| Flag     | `--api-key <KEY>`       | One key. The key shows in the process list. Prefer the variable or the file. |
| Variable | `BLAZR_API_KEY`         | Read when `--api-key` is absent.                                             |
| File     | `--api-key-file <PATH>` | One key per line. Blank lines and lines starting with `#` are ignored.       |

Keys from the flag or variable and the file combine. Without any key, the server accepts all requests.

Protected routes require the header `Authorization: Bearer <key>`. A missing or wrong key returns `401` with a JSON body and type `invalid_api_key`. Only `GET /health` and `GET /metrics` skip the check.

```bash
curl -H "Authorization: Bearer $BLAZR_API_KEY" http://127.0.0.1:8080/v1/models
```

### TLS

Set both the certificate and the key. Each comes from a flag, or from `server.tls_cert` and `server.tls_key` in `config.yaml`. Blazr rejects one without the other. Both files use PEM format.

- The listener uses rustls with its safe default protocol versions, TLS 1.2 and 1.3.
- The server does not request client certificates.
- A certificate file without a private key fails at startup with `No private key found in <path>`.

### CORS

CORS is enabled by default. With `server.cors_origins` empty, the server allows any origin, method, and header. Set `server.cors_origins` to a list to restrict browser origins. Set `server.cors_enabled` to `false` to turn CORS off.

### Limits

`blazr serve` reads these limits from the `server` section of `~/.blazr/config.yaml`.

| Limit                  | Default                  | Key                           | Exceeded response                     |
| ---------------------- | ------------------------ | ----------------------------- | ------------------------------------- |
| Request body           | 10 MB (`10485760` bytes) | `server.max_body_size`        | `413 Payload Too Large`               |
| Concurrent requests    | 16                       | `server.max_concurrent_requests` | Extra requests wait. No error. An open stream holds its slot. |
| Request timeout        | 300 seconds              | `server.request_timeout_secs` | `408 Request Timeout`                 |
| In-flight token budget | `0`, unlimited           | `server.max_inflight_tokens`  | `503` with `retry-after: 5`, when set |

Request logging is on by default through `server.request_logging`. Each request logs one line with the method, path, status, and latency.

## Operations

### Health

`GET /health` returns `200` with a JSON body. It needs no API key.

| Field           | Type   | Meaning                                                                             |
| --------------- | ------ | ----------------------------------------------------------------------------------- |
| `status`        | string | Always `ok` while the process serves.                                               |
| `version`       | string | blazr crate version.                                                                |
| `loaded_models` | array  | Names of loaded models.                                                             |
| `data_parallel_replicas` | integer | Copies of the start model that serve requests. `1` without data parallelism. |
| `gpu_memory`    | object | CUDA builds only: `used_bytes`, `free_bytes`, `total_bytes`, `used_gb`, `total_gb`. |

`/health` returns `200` when no model is loaded. Use `loaded_models` to check the model state.

### Metrics

`GET /metrics` returns Prometheus text format. It needs no API key.

| Name                                | Type      | Records                                                  |
| ----------------------------------- | --------- | -------------------------------------------------------- |
| `blazr_requests_total`              | counter   | HTTP requests, by `method` and `path`                    |
| `blazr_requests_active`             | gauge     | HTTP requests in flight                                  |
| `blazr_request_duration_seconds`    | histogram | HTTP request duration, by `method`, `path`, and `status` |
| `blazr_tokens_prompted_total`       | counter   | Prompt tokens processed                                  |
| `blazr_tokens_generated_total`      | counter   | Tokens generated                                         |
| `blazr_time_to_first_token_seconds` | histogram | Time to first token                                      |
| `blazr_tokens_per_second`           | histogram | Decode throughput per request                            |
| `blazr_inter_token_latency_seconds` | histogram | Time between consecutive tokens                          |
| `blazr_active_decode_slots`         | gauge     | Requests generating tokens                               |
| `blazr_inflight_tokens`             | gauge     | Estimated prompt and decode tokens in flight             |
| `blazr_models_loaded`               | gauge     | Loaded models, LoRA variants included                    |
| `blazr_scheduler_models_available`  | gauge     | Models in `BLAZR_MODEL_DIR`                              |
| `blazr_kv_cache_utilization_ratio`  | gauge     | KV cache blocks in use, batched scheduler only           |
| `blazr_prefix_cache_hits_total`     | counter   | Prefix cache hits                                        |
| `blazr_prefix_cache_misses_total`   | counter   | Prefix cache misses                                      |
| `blazr_vram_used_bytes`             | gauge     | GPU memory in use, CUDA builds only                      |
| `blazr_scheduler_loads_total`       | counter   | Model loads                                              |
| `blazr_scheduler_evictions_total`   | counter   | Model evictions. A base evicted with its LoRA variants counts once. |
| `blazr_lora_variant_builds_total`   | counter   | LoRA variants built over a loaded base: at registration, and again after an eviction or unload |
| `blazr_queue_depth`                 | gauge     | Requests waiting: queued sequences in the batching schedulers, plus requests waiting for a `server.max_concurrent_requests` slot |
| `blazr_slo_violations_total`        | counter   | Latency samples with a rolling percentile over its `server.slo` threshold. Labels: `metric` (`ttft`, `itl`, `e2e`) and `percentile` (`p50`, `p95`, `p99`) |
| `blazr_slo_batch_growth_held`       | gauge     | `1` while a P95 TTFT or ITL breach holds batch growth, else `0` |
| `blazr_data_parallel_replicas`      | gauge     | Data-parallel replicas of the start model                |
| `blazr_replica_active_requests`     | gauge     | Requests in flight on each replica. Label: `replica`.    |

`blazr_token_budget_utilization_ratio` appears only when `server.max_inflight_tokens` is above `0`. `blazr_vram_used_bytes` updates on CUDA builds at each scrape while a model is loaded.

The `path` label holds the raw request path. Each distinct path adds a time series.

### Graceful shutdown

On Unix, `SIGTERM` and `SIGINT` start shutdown.

- Plain HTTP: the server stops accepting connections and finishes in-flight requests.
- HTTPS (`--tls-cert` and `--tls-key`): the server stops accepting connections and does not wait for open connections. In-flight HTTPS requests can end early.

Terminate TLS at a reverse proxy when restarts must drain requests. See the nginx example.

### Overload and retry

`blazr serve` returns `503` for load only when `server.max_inflight_tokens` is above `0` and the budget is full.

When the budget is exceeded, the server returns:

- Status `503`.
- Header `retry-after: 5`.
- Body type `server_overloaded`.

Clients must wait the `retry-after` seconds before they retry. Other errors that clients handle:

| Status | Cause                        |
| ------ | ---------------------------- |
| `401`  | Missing or wrong API key     |
| `408`  | Request exceeded 300 seconds |
| `413`  | Request body exceeded 10 MB  |

### SLO thresholds

Set the thresholds under `server.slo` in `config.yaml`. A sample over its threshold logs a warning and adds 1 to `blazr_slo_violations_total`. While the P95 TTFT or P95 ITL is over its threshold, batching admits no sequences beyond the running count. An idle scheduler still admits one. See [SLO thresholds](configuration.md#slo-thresholds).

### Check running models

```bash
blazr ps --server http://127.0.0.1:8080
```

`blazr ps` reads `GET /api/ps` and prints the model names and sizes to standard error. It sends no API key. Against a server with API keys, it prints the `401` status as an error. It exits `0` when the server is unreachable or returns an error status.

## Multi-node

Multi-node commands exist only in a build with the `distributed` feature. The release archives and the default `cargo install` features do not include it.

```bash
cargo build --release --features distributed         # CPU
cargo build --release --features cuda,distributed    # CUDA
```

### Swarm

The `swarm` command runs one leader and `--workers` workers, one process per node. The leader binds nexar's seed and holds rank 0. Workers take ranks 1 to `--workers` in the order they join. Rank `r` runs pipeline stage `r`, a contiguous range of layers, and the leader runs stage 0.

Leader:

```bash
export BLAZR_SWARM_TOKEN=<shared-secret>
blazr swarm --role leader --model <name> --workers 2 --listen-addr <leader-ip>:9091 --swarm-port 9090 --port 8080
```

Worker, once per worker node:

```bash
export BLAZR_SWARM_TOKEN=<shared-secret>
blazr swarm --role worker --workers 2 --leader <leader-ip>:9090
```

The leader sends each worker its layer range and the model path, and starts the HTTP server once every worker loaded its model. The path is where the leader finds the model name: under its `model_dir`, or the name itself as a path. Every node must hold the model at that path. Each node reads only the tensors of its own layer range: stage 0 adds the token embedding, and the last stage adds the final norm and LM head. A node needs memory for its own layers, not the full model. SafeTensors, GPTQ, AWQ and GGUF checkpoints all load this way.

The leader serves the HTTP API of `blazr serve` for its model, and every generation runs through the pipeline. The leader runs stage 0, and the last worker returns the logits of each step. The leader samples every token, so grammars, `logprobs`, `logit_bias`, DRY, mirostat and the seed work as on one process. The leader's HTTP front end holds the model's config, tokenizer and chat template, and no weights beyond stage 0. Embeddings, images and LoRA variants need the whole model, so the leader answers them with `501`. A request for another model is a `404`.

- **Concurrency:** at most `--max-requests` requests run at once. Each stage holds one KV cache per request. A request past the cap waits for a slot.
- **Long prompts:** with `inference.prefill_chunk_size` above `0` (default `512`), a prompt goes through the pipeline in chunks of that many tokens. A client that leaves mid-prefill frees its request at the next chunk. `0` sends the whole prompt as one frame.
- **Errors:** a stage error ends only its request. A lost worker ends every request in flight and refuses every new one. `GET /health` then answers `503` with the reason. Restart every node to recover.
- **Models:** Mamba, hybrid and qwen35 models, and tensor-parallel shards, cannot run as stages. The leader fails at start with the reason, before it reads a weight and before any worker joins. A multimodal checkpoint runs its text model, and its vision and audio towers stay unread.

| Flag               | Default             | Required     | Purpose                                                                  |
| ------------------ | ------------------- | ------------ | ------------------------------------------------------------------------ |
| `--token`          | `BLAZR_SWARM_TOKEN` | Yes          | Shared secret. It is the nexar cluster token.                            |
| `--role`           | `worker`            | No           | `leader` or `worker`.                                                    |
| `--workers`        | `1`                 | No           | Number of worker nodes. Every node passes the same value. `0` runs the leader alone. |
| `--model`, `-m`    | `default_model`     | Leader only  | Model to serve.                                                          |
| `--leader`         | none                | Worker       | The leader's seed address, for example `192.168.1.10:9090`.              |
| `--swarm-port`     | `9090`              | No           | UDP port of the leader's seed, on the `--listen-addr` IP. Leader only.   |
| `--listen-addr`    | `0.0.0.0:9091`      | No           | This node's mesh listener. The port carries UDP and TCP.                 |
| `--advertise-addr` | `--listen-addr`     | See below    | The address peers dial to reach this node.                               |
| `--join-timeout`   | `600`               | No           | Seconds every node waits for the whole cluster to form.                  |
| `--port`           | `server.port`, else `8080` | No    | HTTP API port. Leader only.                                              |
| `--host`           | `server.host`, else `0.0.0.0` | No | HTTP API address. Leader only.                                           |
| `--api-key`        | `BLAZR_API_KEY`     | No           | API key for the leader's HTTP API.                                       |
| `--max-requests`   | `16`                | No           | Requests the swarm runs at once. Every node passes the same value.       |

The `--mdns` flag is removed. A worker needs `--leader`.

### Disaggregated prefill and decode

The `disagg` command runs three roles: `router`, `prefill`, and `decode`, one process per node. The router binds nexar's seed on `--router-addr` and holds rank 0. Each worker joins with the rank of its role and `--index`:

| Role      | `--index` | Rank                          |
| --------- | --------- | ----------------------------- |
| `router`  | none      | `0`                           |
| `prefill` | `i`       | `1 + i`                       |
| `decode`  | `i`       | `1 + --prefill-count + i`     |

Every node passes the same `--router-addr`, `--prefill-count` and `--decode-count`. The ranks do not depend on start order.

Router:

```bash
export NEXAR_CLUSTER_TOKEN=<shared-secret>
blazr disagg --role router --model <name> --router-addr <router-ip>:9190 --prefill-count 1 --decode-count 1 --port 8080
```

The router loads the config, tokenizer and chat template of `--model`, and no weights. Name the model the workers load. The router serves the full HTTP API of `blazr serve`: OpenAI completions and chat, streaming, stop sequences, usage, the Anthropic and Ollama routes, auth through `BLAZR_API_KEY`, and `/metrics`. It serves `--model` only. A request for another model is a `404`.

The workers sample with temperature, `top_k`, `top_p`, `min_p`, the repeat, frequency and presence penalties, the penalty window, and the seed. The router answers `400` to a request that sets any of these:

- `grammar`, `response_format` or a tool constraint
- `mirostat`
- `dry_multiplier`
- `typical_p`
- `dynatemp_range`
- `logit_bias`
- `logprobs`

Embeddings return `501`. Image inputs and LoRA variants are not served. GDN (`qwen35`), SSM and tensor-parallel models fail at worker start.

A client that leaves cancels its request on both workers. The prefill worker runs a prompt in chunks of `inference.prefill_chunk_size` tokens and stops at the next chunk once the cancel arrives.

Prefill worker:

```bash
export NEXAR_CLUSTER_TOKEN=<shared-secret>
blazr disagg --role prefill --model <name> --router-addr <router-ip>:9190 --prefill-count 1 --decode-count 1 --index 0
```

Decode worker:

```bash
export NEXAR_CLUSTER_TOKEN=<shared-secret>
blazr disagg --role decode --model <name> --router-addr <router-ip>:9190 --prefill-count 1 --decode-count 1 --index 0
```

A worker loads its model, then joins the cluster. The router starts the HTTP server once every worker joined.

| Flag               | Role     | Default               | Required for          | Purpose                                                       |
| ------------------ | -------- | --------------------- | --------------------- | ------------------------------------------------------------- |
| `--role`           | all      | `router`              | No                    | `router`, `prefill`, or `decode`. Other values fail.          |
| `--model`, `-m`    | all      | `default_model` (workers) | All roles. The router takes `--model` only. | Workers: model to load. Router: tokenizer and chat template source. |
| `--router-addr`    | all      | none                  | All roles             | The router's seed address. The router binds it, and each worker dials it. |
| `--prefill-count`  | all      | `1`                   | No                    | Number of prefill workers. The same on every node.            |
| `--decode-count`   | all      | `1`                   | No                    | Number of decode workers. The same on every node.             |
| `--index`          | workers  | `0`                   | Each worker past the first of its role | This worker's index in its role. The router refuses it. |
| `--cluster-token`  | all      | `NEXAR_CLUSTER_TOKEN` | Non-loopback addresses | Shared secret of the cluster.                                |
| `--listen-addr`    | all      | `0.0.0.0:9191`        | No                    | This node's mesh listener. The port carries UDP and TCP.      |
| `--advertise-addr` | all      | `--listen-addr`       | See below             | The address peers dial to reach this node.                    |
| `--join-timeout`   | all      | `600`                 | No                    | Seconds every node waits for the whole cluster to form.       |
| `--port`           | `router` | `server.port`, else `8080` | No               | HTTP API port.                                                |
| `--host`           | `router` | `server.host`, else `0.0.0.0` | No            | HTTP API address.                                             |

An `--index` at or past its role's count fails at start and names both flags. Two workers with one index fail with `rank N is not free`. The `--prefill` and `--decode` address lists are removed: the counts replace them.

Errors: a worker error ends only its request. A lost prefill or decode worker ends the requests it held and refuses every new one. `GET /health` on the router then answers `503` with the reason. Restart every node to recover.

### Multi-host setup

Both commands form one nexar cluster: one process per rank, joined through the seed. Set these on every host:

- **Seed address:** `swarm` workers pass the leader's `--listen-addr` IP and `--swarm-port` as `--leader`. `disagg` nodes pass the same `--router-addr`. The seed port carries UDP.
- **Token:** use one secret on every node. The bootstrap connection does not verify the seed's certificate, so the token is the only defense against a stranger who joins or poses as the seed. A seed, `--listen-addr` or `--advertise-addr` that is not loopback requires a token. Without one, the node fails at start and names the address and the flag. Pass the token through the variable: a flag shows in the process list.
- **Firewall:** open the seed port for UDP on the seed host. Open each node's `--listen-addr` port for UDP (QUIC) and TCP (the bulk sidecar).
- **NAT and containers:** set `--advertise-addr` to the address peers can reach. Publish the mesh port for both UDP and TCP. The seed node joins its own seed over loopback, so the seed cannot see its address. When both its seed and its `--listen-addr` bind `0.0.0.0`, it refuses to start without `--advertise-addr`. A worker with an unspecified IP advertises the IP the seed sees.
- **Shared hosts:** give each node on one host its own `--listen-addr` port.
- **Timeout:** `--join-timeout` bounds the whole formation on every node. `disagg` workers load their model before they join, so set it above the slowest model load.
- **nexar config:** set the same `NEXAR_*` variables on every node. A bulk sidecar mismatch (`NEXAR_ENABLE_TCP_BULK_SIDECAR`, `NEXAR_ENCRYPT_BULK_TRANSPORT`) fails formation.

A formation error names its cause and its fix: a token mismatch, a rank already taken, a world size that differs between nodes, or nodes that did not join in time.

### Shutdown

SIGINT or SIGTERM on the leader or router stops its HTTP server. It then sends every worker a shutdown notice. Each worker finishes the request in progress and leaves its loop. Every node then runs the nexar shutdown: it flushes the bulk sidecars and exchanges a done signal with each peer before it closes, so no sent data is lost. SIGINT or SIGTERM on a worker runs the same shutdown on that worker. A peer that does not answer within `NEXAR_BARRIER_TIMEOUT_SECS` (default `30`) is logged as a warning, and its connection closes anyway.

### Tensor parallelism

Set `BLAZR_TP_SIZE`, or `inference.tensor_parallel_size` in `config.yaml`, above `1` to split one model across GPUs. The variable wins over the key, and the key wins over the model's `config.json`. The default is `1`. Such a run needs a build with both `cuda` and `nccl`. `blazr serve` fails at startup in any other build. The loader checks that the head counts divide by the size before it reads weights. One blazr process runs every rank, each as a thread on GPU `device + rank`. Do not set `BLAZR_TP_RANK`: blazr refuses it, because it shares no NCCL id across processes. A tensor-parallel model decodes one sequence at a time with a contiguous KV cache. Paged attention, speculative decoding and the prefix caches are off for it. No multi-GPU run has tested this path.

## Environment variables

Blazr reads no `.env` file. Export these variables before the command runs.

| Variable                         | Read by                         | Purpose                                                |
| -------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `BLAZR_MODEL_DIR`                | `serve`, `pull`, `list`, `info` | Model directory. Then `model_dir`, then `./models`.    |
| `BLAZR_API_KEY`                  | `serve`, `swarm`                | API key, when `--api-key` is absent.                   |
| `BLAZR_VOICE_DIR`                | `serve`                         | Kokoro voice directory, when `--voice-dir` is absent.  |
| `BLAZR_SWARM_TOKEN`              | `swarm`                         | Swarm token, when `--token` is absent.                 |
| `NEXAR_CLUSTER_TOKEN`            | `disagg`                        | Cluster token, when `--cluster-token` is absent.       |
| `BLAZR_TP_SIZE`, `BLAZR_TP_RANK` | `serve`, model loader           | Tensor parallel size. A set rank is refused.           |
| `HF_HOME`, `HF_HUB_CACHE`        | `serve`, `pull`                 | Hugging Face cache location.                           |
| `HF_HUB_OFFLINE`                 | `serve`                         | Blocks Hugging Face fetches when set.                  |
| `RUST_LOG`                       | all commands                    | Log filter. Defaults to `blazr=info,tower_http=debug`. |

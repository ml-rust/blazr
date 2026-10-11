# blazr deployment

This guide covers installing blazr, running `blazr serve`, and operating it behind TLS and a reverse proxy.

## Install

### Prebuilt archives

Each [GitHub release](https://github.com/ml-rust/blazr/releases) carries one archive per platform. Every archive includes the `audio` feature and runs on the CPU only.

| Archive                    | Platform               | Built on                                |
| -------------------------- | ---------------------- | --------------------------------------- |
| `blazr-linux-x64.tar.gz`   | Linux x86_64           | `ubuntu-22.04`, glibc 2.35 or newer     |
| `blazr-linux-arm64.tar.gz` | Linux aarch64          | `ubuntu-24.04-arm`, glibc 2.39 or newer |
| `blazr-macos-arm64.tar.gz` | macOS on Apple silicon | `macos-latest`                          |
| `blazr-windows-x64.zip`    | Windows x86_64         | `windows-latest`                        |

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

| Feature       | Adds                                                                            |
| ------------- | ------------------------------------------------------------------------------- |
| `audio`       | `/v1/audio/*` routes and the `speak`, `transcribe`, and `sound-review` commands |
| `cuda`        | CUDA backend. Needs the CUDA toolkit 12.8 or newer, including 13.x.             |
| `distributed` | The `swarm` and `disagg` commands. See [Multi-node](#multi-node).               |

Create the binary from a source checkout:

```bash
cargo build --release --features audio         # CPU
cargo build --release --features audio,cuda    # CUDA
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

The startup output prints the listen address, the model, and `Backend: CUDA` or `Backend: CPU`.

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

The build selects the backend. The `device` key in `config.yaml` selects the CUDA index.

- A `cuda` build runs on the CUDA device from `device`, or device 0 when the key is unset.
- A default build runs on the CPU. `blazr serve` has no `--cpu` flag.
- A `cuda` build refuses `device: cpu` for `serve`. Remove the key or set a CUDA index.

Check the startup output for `Backend: CUDA` or `Backend: CPU` after each start.

### Data-parallel replicas

Set `inference.data_parallel` in `config.yaml`, or pass `--data-parallel`, to load one copy of the start model per device.

- A CUDA build places copy `i` on CUDA device `device + i`.
- A CPU build loads one CPU copy for each replica.
- Each copy runs its own batching engine.
- The chat, completions, responses, Anthropic messages, and infill handlers route through the group.
- The router records token prefixes every 16 tokens, up to 1024 tokens.
- A request goes to the replica that holds its longest warm prefix. Otherwise it goes to the least loaded replica.
- A warm replica more than 4 requests above the least loaded replica gives up the request.
- Embeddings and rerank requests run on replica 0.
- `/health` reports the replica count in `data_parallel_replicas`.
- `blazr serve` refuses `0`, and a value above `1` with no start model. It also refuses a value above `1` with `tensor_parallel_size` above `1`. That error names both keys.

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
| Concurrent requests    | 16                       | `server.max_concurrent_requests` | Extra requests wait. No error.     |
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
| `blazr_models_loaded`               | gauge     | Loaded models                                            |
| `blazr_scheduler_models_available`  | gauge     | Models in `BLAZR_MODEL_DIR`                              |
| `blazr_kv_cache_utilization_ratio`  | gauge     | KV cache blocks in use, batched scheduler only           |
| `blazr_prefix_cache_hits_total`     | counter   | Prefix cache hits                                        |
| `blazr_prefix_cache_misses_total`   | counter   | Prefix cache misses                                      |
| `blazr_vram_used_bytes`             | gauge     | GPU memory in use, CUDA builds only                      |
| `blazr_scheduler_loads_total`       | counter   | Model loads                                              |
| `blazr_scheduler_evictions_total`   | counter   | Model evictions                                          |
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

The `swarm` command runs one leader and any number of workers. Each node listens on `0.0.0.0:<swarm-port>`. Open `--swarm-port` between all nodes.

Leader:

```bash
BLAZR_SWARM_TOKEN=<shared-secret> blazr swarm --role leader --model <name> --swarm-port 9090 --port 8080
```

Worker, with a known leader address:

```bash
BLAZR_SWARM_TOKEN=<shared-secret> blazr swarm --role worker --leader <leader-host>:9090
```

Worker, with LAN discovery:

```bash
BLAZR_SWARM_TOKEN=<shared-secret> blazr swarm --role worker --mdns
```

| Flag            | Default             | Required                         | Purpose                                          |
| --------------- | ------------------- | -------------------------------- | ------------------------------------------------ |
| `--token`       | `BLAZR_SWARM_TOKEN` | Yes, all roles                   | Shared secret for swarm membership.              |
| `--role`        | `worker`            | No                               | `leader` or `worker`.                            |
| `--model`, `-m` | `default_model`     | Leader only                      | Model to serve.                                  |
| `--leader`      | none                | Worker, unless `--mdns` is set   | Leader address, for example `192.168.1.10:9090`. |
| `--mdns`        | off                 | Worker, unless `--leader` is set | Enable mDNS discovery on the LAN.                |
| `--swarm-port`  | `9090`              | No                               | Port for the swarm transport.                    |
| `--port`        | `server.port`, else `8080` | No                        | HTTP API port. Leader only.                      |
| `--host`        | `server.host`, else `0.0.0.0` | No                     | HTTP API address. Leader only.                   |
| `--api-key`     | `BLAZR_API_KEY`     | No                               | API key for the leader's HTTP API.               |

Pass the token through `BLAZR_SWARM_TOKEN`. A `--token` flag shows the token in the process list.

### Disaggregated prefill and decode

The `disagg` command runs three roles: `router`, `prefill`, and `decode`.

Router:

```bash
blazr disagg --role router --listen-addr 0.0.0.0:9190 --prefill <prefill-host>:9191 --decode <decode-host>:9192 --port 8080
```

Prefill worker:

```bash
blazr disagg --role prefill --model <name> --router-addr <router-host>:9190 --listen-addr 0.0.0.0:9191
```

Decode worker:

```bash
blazr disagg --role decode --model <name> --router-addr <router-host>:9190 --listen-addr 0.0.0.0:9192
```

| Flag            | Role     | Default        | Required for        | Purpose                                              |
| --------------- | -------- | -------------- | ------------------- | ---------------------------------------------------- |
| `--role`        | all      | `router`       | No                  | `router`, `prefill`, or `decode`. Other values fail. |
| `--model`, `-m` | all      | `default_model` | `prefill`, `decode` | Model to load.                                       |
| `--router-addr` | all      | none           | `prefill`, `decode` | Router address to connect to.                        |
| `--listen-addr` | all      | `0.0.0.0:9190` | No                  | Address the node listens on.                         |
| `--prefill`     | `router` | none           | No                  | One or more prefill worker addresses.                |
| `--decode`      | `router` | none           | No                  | One or more decode worker addresses.                 |
| `--port`        | `router` | `server.port`, else `8080` | No      | HTTP API port.                                       |
| `--host`        | `router` | `server.host`, else `0.0.0.0` | No   | HTTP API address.                                    |

All roles default to `0.0.0.0:9190`. Set a distinct `--listen-addr` on each node that shares a host. The router accepts empty `--prefill` and `--decode` lists. Pass at least one address of each kind.

### Tensor parallelism

Set `BLAZR_TP_SIZE`, or `inference.tensor_parallel_size` in `config.yaml`, above `1` to split one model across GPUs. The variable wins over the key, and the key wins over the model's `config.json`. The default is `1`. Such a run needs a build with both `cuda` and `nccl`. `blazr serve` fails at startup in any other build. The loader checks that the head counts divide by the size before it reads weights. Set `BLAZR_TP_RANK` on each process to its rank. The rank defaults to `0`.

## Environment variables

Blazr reads no `.env` file. Export these variables before the command runs.

| Variable                         | Read by                         | Purpose                                                |
| -------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `BLAZR_MODEL_DIR`                | `serve`, `pull`, `list`, `info` | Model directory. Then `model_dir`, then `./models`.    |
| `BLAZR_API_KEY`                  | `serve`, `swarm`                | API key, when `--api-key` is absent.                   |
| `BLAZR_VOICE_DIR`                | `serve`                         | Kokoro voice directory, when `--voice-dir` is absent.  |
| `BLAZR_SWARM_TOKEN`              | `swarm`                         | Swarm membership token.                                |
| `BLAZR_TP_SIZE`, `BLAZR_TP_RANK` | `serve`, model loader           | Tensor parallel size and rank.                         |
| `HF_HOME`, `HF_HUB_CACHE`        | `serve`, `pull`                 | Hugging Face cache location.                           |
| `HF_HUB_OFFLINE`                 | `serve`                         | Blocks Hugging Face fetches when set.                  |
| `RUST_LOG`                       | all commands                    | Log filter. Defaults to `blazr=info,tower_http=debug`. |

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
| `--model`, `-m`                            | none                     | Model to load at startup. A name or a path.                                        |
| `--chat-template`                          | the model's own template | Built-in name or Jinja template file for every served model.                       |
| `--host`                                   | `0.0.0.0`                | Address to bind.                                                                   |
| `--port`                                   | `8080`                   | Port to bind.                                                                      |
| `--api-key`                                | none                     | API key. Also read from `BLAZR_API_KEY`.                                           |
| `--api-key-file`                           | none                     | File of API keys, one per line.                                                    |
| `--tls-cert`                               | none                     | PEM certificate chain. Needs `--tls-key`.                                          |
| `--tls-key`                                | none                     | PEM private key. Needs `--tls-cert`.                                               |
| `--reasoning-format`                       | `deepseek`               | Default chat reasoning split: `deepseek` or `none`.                                |
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

`blazr pull` writes to `BLAZR_MODEL_DIR` when `--output` is absent. Replace `<name>` with a name that `blazr list` shows.

`--model` resolves in this order:

1. The value as a path.
2. `$BLAZR_MODEL_DIR/<name>`.
3. `$BLAZR_MODEL_DIR/<name>.safetensors`.
4. `$BLAZR_MODEL_DIR/<name>.gguf`.
5. `$BLAZR_MODEL_DIR/<name>/model.safetensors`.

`BLAZR_MODEL_DIR` defaults to `./models`, relative to the working directory of the process.

With `--model`, `blazr serve` loads and warms the model before it binds the port. A load error stops startup.

### Start on demand

Without `--model`, `blazr serve` starts with no model loaded. Chat, embedding, tokenize, and template requests load the named model on first use.

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

The build selects the backend. No flag selects it at run time.

- A `cuda` build runs on CUDA device 0. No flag selects another device.
- A default build runs on the CPU. `blazr serve` has no `--cpu` flag.

Check the startup output for `Backend: CUDA` or `Backend: CPU` after each start.

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

Pass both `--tls-cert` and `--tls-key`. The `serve` command rejects either flag alone. Both files use PEM format.

- The listener uses rustls with its safe default protocol versions, TLS 1.2 and 1.3.
- The server does not request client certificates.
- A certificate file without a private key fails at startup with `No private key found in <path>`.

### CORS

CORS is enabled by default. The server allows any origin, method, and header. No flag restricts the origins. Restrict browser origins at the reverse proxy, or change `src/cli/serve.rs`.

### Limits

`blazr serve` builds its server config from defaults. No flag or file changes these values.

| Limit                  | Default                  | Exceeded response                     |
| ---------------------- | ------------------------ | ------------------------------------- |
| Request body           | 10 MB (`10485760` bytes) | `413 Payload Too Large`               |
| Concurrent requests    | 16                       | Extra requests wait. No error.        |
| Request timeout        | 300 seconds              | `408 Request Timeout`                 |
| In-flight token budget | `0`, unlimited           | `503` with `retry-after: 5`, when set |

The config keys are `max_body_size`, `max_concurrent_requests`, `request_timeout_secs`, and `max_inflight_tokens` in `ServerConfig`. The `serve` command does not read them from `~/.blazr/config.yaml`. Changing them needs a code change in `src/cli/serve.rs`.

Request logging is on by default. Each request logs one line with the method, path, status, and latency.

## Operations

### Health

`GET /health` returns `200` with a JSON body. It needs no API key.

| Field           | Type   | Meaning                                                                             |
| --------------- | ------ | ----------------------------------------------------------------------------------- |
| `status`        | string | Always `ok` while the process serves.                                               |
| `version`       | string | blazr crate version.                                                                |
| `loaded_models` | array  | Names of loaded models.                                                             |
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

These names exist in the code but no code path records them under `blazr serve`:

- `blazr_queue_depth`
- `blazr_slo_violations_total`
- `blazr_token_budget_utilization_ratio`, which records only when `max_inflight_tokens` is above `0`

Do not create alerts on these names.

The `path` label holds the raw request path. Each distinct path adds a time series.

### Graceful shutdown

On Unix, `SIGTERM` and `SIGINT` start shutdown.

- Plain HTTP: the server stops accepting connections and finishes in-flight requests.
- HTTPS (`--tls-cert` and `--tls-key`): the server stops accepting connections and does not wait for open connections. In-flight HTTPS requests can end early.

Terminate TLS at a reverse proxy when restarts must drain requests. See the nginx example.

### Overload and retry

`blazr serve` returns no `503` for load. The in-flight token budget check runs only when `max_inflight_tokens` is above `0`, and `serve` never sets it.

When the budget applies and is exceeded, the server returns:

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

The `slo` key in `ServerConfig` has no effect. No code path builds the SLO checker, so no SLO warnings appear.

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
| `--model`, `-m` | none                | Leader only                      | Model to serve.                                  |
| `--leader`      | none                | Worker, unless `--mdns` is set   | Leader address, for example `192.168.1.10:9090`. |
| `--mdns`        | off                 | Worker, unless `--leader` is set | Enable mDNS discovery on the LAN.                |
| `--swarm-port`  | `9090`              | No                               | Port for the swarm transport.                    |
| `--port`        | `8080`              | No                               | HTTP API port. Leader only.                      |

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
| `--model`, `-m` | all      | none           | `prefill`, `decode` | Model to load.                                       |
| `--router-addr` | all      | none           | `prefill`, `decode` | Router address to connect to.                        |
| `--listen-addr` | all      | `0.0.0.0:9190` | No                  | Address the node listens on.                         |
| `--prefill`     | `router` | none           | No                  | One or more prefill worker addresses.                |
| `--decode`      | `router` | none           | No                  | One or more decode worker addresses.                 |
| `--port`        | `router` | `8080`         | No                  | HTTP API port.                                       |

All roles default to `0.0.0.0:9190`. Set a distinct `--listen-addr` on each node that shares a host. The router accepts empty `--prefill` and `--decode` lists. Pass at least one address of each kind.

### Tensor parallelism

Set `BLAZR_TP_SIZE` above `1` to split one model across GPUs. Such a run needs a build with both `cuda` and `nccl`. `blazr serve` fails at startup in any other build. Set `BLAZR_TP_RANK` on each process to its rank. The rank defaults to `0`.

## Environment variables

| Variable                         | Read by                         | Purpose                                                |
| -------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `BLAZR_MODEL_DIR`                | `serve`, `pull`, `list`, `info` | Model directory. Defaults to `./models`.               |
| `BLAZR_API_KEY`                  | `serve`                         | API key, when `--api-key` is absent.                   |
| `BLAZR_VOICE_DIR`                | `serve`                         | Kokoro voice directory, when `--voice-dir` is absent.  |
| `BLAZR_SWARM_TOKEN`              | `swarm`                         | Swarm membership token.                                |
| `BLAZR_TP_SIZE`, `BLAZR_TP_RANK` | `serve`, model loader           | Tensor parallel size and rank.                         |
| `HF_HOME`, `HF_HUB_CACHE`        | `serve`, `pull`                 | Hugging Face cache location.                           |
| `HF_HUB_OFFLINE`                 | `serve`                         | Blocks Hugging Face fetches when set.                  |
| `RUST_LOG`                       | all commands                    | Log filter. Defaults to `blazr=info,tower_http=debug`. |

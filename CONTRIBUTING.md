# Contributing to blazr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

Install these first:

| Need | For |
| --- | --- |
| Rust 1.95 or newer | Every build |
| `cargo-deny` | The dependency audit |
| CUDA toolkit 12.8 or newer, `nvcc` on `PATH` | `--features cuda` |

## Set up

```bash
cp .cargo/config.toml.example .cargo/config.toml
```

- Sibling checkouts: the file points numr, boostr, splintr, regexr, nexar, containr, voxr, and voxr-narrate at `../`. Without it, cargo resolves them from crates.io, and the build fails on APIs the published versions lack.
- Models: set `BOOSTR_MODELS_DIR` in that file to your checkpoint directory.
- Skipped tests: a test that needs a model skips with a notice when its variable is unset. Such a run reports success without testing the model.

## Build and test

Run these commands from the repository root. Each one is a command that a CI job in `.github/workflows/ci.yml` runs.

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features audio -- -D warnings
cargo test --features audio --no-fail-fast
cargo test --lib --no-default-features --no-fail-fast
cargo deny --exclude-dev check
```

CI also runs these jobs:

- MSRV: `cargo check --features audio` on the `rust-version` toolchain from `Cargo.toml`.
- Check: `cargo check --all-targets` on Linux, macOS, and Windows, plus `--features audio` on Linux.
- Target checks: `cargo check` for `wgpu`, `capi`, `wasm32-unknown-unknown`, Android, and iOS.

CI does not build the `cuda` feature. Build it locally when you change CUDA code:

```bash
cargo build --release --features cuda
```

## Conventions

- Put model architectures, tensor operations, and quantization kernels in [boostr](https://github.com/ml-rust/boostr). blazr delegates to boostr.
- Keep `mod.rs` files to `pub mod` and `pub use` lines only.
- Keep one concern per file. Keep each file under 500 lines.
- Do not call `.unwrap()` in library code. `src/main.rs` can use it.
- Do not copy tensors from GPU to CPU for computation.
- Return OpenAI-route errors as `{ error: { message, type, code } }`.
- Send OpenAI streams as SSE ending with `data: [DONE]`. Anthropic streams send named `event:` lines and have no `[DONE]`. Ollama streams send NDJSON.

## Pull requests

- Add tests for every behavior change.
- Update the matching page in `docs/` when you change a route, flag, config key, or model support.
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.

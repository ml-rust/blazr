# Contributing to blazr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and test

Run these commands from the repository root. They match the CI jobs in `.github/workflows/ci.yml`.

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features audio -- -D warnings
cargo test --features audio --no-fail-fast
cargo test --lib --no-default-features --no-fail-fast
```

Tests that need a model directory skip with a notice when their environment variable is unset.

CI runs `cargo check` for the `wgpu`, `capi`, `wasm32-unknown-unknown`, Android, and iOS targets. CI does not build the `cuda` feature. Build it locally when you change CUDA code:

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
- Send streaming responses as SSE and end them with `data: [DONE]`.

## Pull requests

- Add tests for every behavior change.
- Update the matching page in `docs/` when you change a route, flag, config key, or model support.
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.

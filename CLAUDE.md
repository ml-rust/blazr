# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

blazr is a blazing-fast inference server for oxidizr-trained models, supporting the custom hybrid architecture (Mamba2 + MLA + MoE) that standard inference tools like llama.cpp cannot handle.

## Build Commands

```bash
cargo build --release                 # CPU-only build
cargo build --release --features cuda # With CUDA support (requires CUDA 12.x)
cargo test                            # Run tests
cargo clippy --all-targets            # Lint
```

## CLI Usage

```bash
# Generate text
blazr generate --model path/to/checkpoint --prompt "Hello" --max-tokens 100

# Start server
blazr serve --model path/to/checkpoint --port 8080 --host 0.0.0.0

# Show model info
blazr info --model path/to/checkpoint
```

## API Endpoints

OpenAI-compatible REST API:

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion

## Architecture

### Hybrid Layer Types

The model supports two layer types, dispatched based on `config.mamba_layers`:

1. **Mamba2 layers** - State Space Model with:
   - Conv1D for causal convolution
   - SSM with A, B, C, D matrices
   - Chunk-based processing (prefill) or sequential (decode)

2. **MLA + MoE layers** - Attention with experts:
   - Multi-Head Latent Attention (compressed KV cache)
   - Fine-grained MoE with top-k routing
   - Optional shared expert (always active)

### State Management

- `Mamba2State` - SSM hidden state + conv state per Mamba2 layer
- `MlaCache` - Compressed KV latents per MLA layer

### Checkpoint Format

Expects oxidizr SafeTensors checkpoints:

```
checkpoint_dir/
├── model.safetensors    # Model weights
└── config.json          # Model configuration
```

## Key Dependencies

- `candle-core/nn/transformers` - Tensor operations (same as oxidizr)
- `splintr` - Tokenization (Llama 3 vocabulary)
- `axum` - HTTP server
- `safetensors` - Checkpoint loading

## Feature Flags

- `cuda` - Enable CUDA GPU acceleration

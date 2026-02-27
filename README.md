# blazr

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A blazing-fast inference server for hybrid neural architectures, supporting Mamba2 SSM, Multi-Head Latent Attention (MLA), Mixture of Experts (MoE), and standard transformers.

## Features

- **Auto-detection** - Automatically detects model architecture, format (HuggingFace vs oxidizr), and tokenizer vocabulary from checkpoint tensors
- **Hybrid Architecture Support** - Seamlessly handles mixed Mamba2 and attention layers in a single model
- **HuggingFace Compatible** - Loads standard HuggingFace Llama models (tested with llama3.2-1b) alongside custom oxidizr checkpoints
- **OpenAI-Compatible API** - Drop-in replacement with `/v1/completions` and `/v1/chat/completions` endpoints
- **High Performance** - Written in Rust using the Candle ML framework with optional CUDA acceleration
- **Multiple Tokenizers** - Supports cl100k_base, o200k_base, llama3, and deepseek_v3 vocabularies via splintr

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ml-rust/blazr.git
cd blazr

# Build (CPU-only)
cargo build --release

# Build with CUDA support (requires CUDA 12.x)
cargo build --release --features cuda
```

### Basic Usage

#### Generate Text

```bash
blazr generate \
  --model ./checkpoints/nano \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --vocab llama3
```

#### Start Server

```bash
blazr serve --model ./checkpoints/nano --port 8080
```

Then make API requests:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### Model Info

```bash
blazr info --model ./checkpoints/nano
```

## Supported Architectures

blazr auto-detects and supports:

- **Mamba2** - State Space Models with selective attention
- **MLA** - Multi-Head Latent Attention with compressed KV cache
- **MoE** - Mixture of Experts with top-k routing and optional shared expert
- **Standard Transformers** - GQA (Grouped Query Attention) with MLP layers

Models can mix and match these layer types freely.

### Auto-Detection

blazr automatically detects:

- **Architecture** - Identifies layer types (Mamba2, MLA, MoE, Transformer) from tensor name patterns
- **Model Format** - Distinguishes between oxidizr format (`layers.X.`) and HuggingFace format (`model.layers.X.`)
- **Tokenizer Vocabulary** - Infers vocabulary from `vocab_size` if `--vocab` is not specified:
  - ~100k tokens → `cl100k_base`
  - ~128k tokens → `llama3`
  - ~129k tokens → `deepseek_v3`
  - ~200k tokens → `o200k_base`

## Tokenizer

blazr uses [splintr](https://github.com/ml-rust/splintr) for high-performance BPE tokenization with pretrained vocabularies.

### Supported Vocabularies

| Vocabulary    | Description          | Vocab Size | Use Case                      |
| ------------- | -------------------- | ---------- | ----------------------------- |
| `cl100k_base` | GPT-4, GPT-3.5-turbo | ~100k      | OpenAI-compatible models      |
| `o200k_base`  | GPT-4o               | ~200k      | Extended multilingual support |
| `llama3`      | Meta Llama 3 family  | ~128k      | Llama 3.x models (default)    |
| `deepseek_v3` | DeepSeek V3/R1       | ~129k      | DeepSeek models               |

All vocabularies include 54 agent tokens for chat, reasoning, and tool-use applications.

### Custom Vocabularies

Custom vocabularies are not yet supported. If you need a custom vocabulary:

1. Train your model with one of the supported vocabularies above
2. Modify blazr's tokenizer module to load your `.tiktoken` file (base64-encoded tokens with ranks)

## Documentation

- [API Reference](docs/api.md) - Complete API endpoint documentation
- [Architecture](docs/architecture.md) - Technical details on hybrid model support
- [Configuration](docs/configuration.md) - Model configuration and tuning options

## CLI Commands

```bash
# Generate text from a prompt
blazr generate --model <path> --prompt "text" [OPTIONS]

# Start inference server
blazr serve --model <path> [--port 8080] [--host 0.0.0.0]

# Display model configuration
blazr info --model <path>

# Decode token IDs (debugging)
blazr decode --ids "123,456,789" --vocab llama3
```

### Options

**Generation:**

- `--model` - Model path (local directory or HuggingFace ID like `meta-llama/Llama-3.2-1B`)
- `--prompt` - Input text prompt
- `--max-tokens` - Maximum tokens to generate (default: 100)
- `--temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Nucleus sampling threshold (default: 0.9)
- `--top-k` - Top-k sampling (default: 40)
- `--vocab` - Tokenizer vocabulary (`llama3`, `cl100k_base`, `o200k_base`, `deepseek_v3`). Auto-detected if not specified.
- `--cpu` - Force CPU inference even if CUDA is available

**Server:**

- `--model` - Model path (local directory or HuggingFace ID)
- `--port` - Port to listen on (default: 8080)
- `--host` - Host to bind to (default: 0.0.0.0)
- `--cpu` - Force CPU inference even if CUDA is available

## Model Format

blazr loads models from SafeTensors checkpoints in two formats:

### oxidizr Format

```
checkpoint_dir/
├── model.safetensors    # Model weights
└── config.json          # Model configuration (optional)
```

Tensor naming: `embed_tokens`, `layers.X.mamba2`, `layers.X.self_attn`, `lm_head`

### HuggingFace Format

```
checkpoint_dir/
├── model.safetensors    # Model weights
└── config.json          # Standard HuggingFace config
```

Tensor naming: `model.embed_tokens`, `model.layers.X.self_attn`, `lm_head`

blazr automatically detects the format and architecture from tensor names. If `config.json` is missing or incomplete, all parameters are inferred from tensor shapes.

## Requirements

- Rust 1.70 or later
- (Optional) CUDA 12.x for GPU acceleration

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.

## Related Projects

- [oxidizr](https://github.com/ml-rust/oxidizr) - Training framework for hybrid Mamba2 + MLA + MoE architectures
- [splintr](https://github.com/ml-rust/splintr) - High-performance BPE tokenizer with Python bindings

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

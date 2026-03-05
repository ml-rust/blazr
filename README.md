# blazr

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Production-grade inference server for LLMs. Supports standard HuggingFace models (Llama, Mistral, Qwen, Phi, Gemma, DeepSeek) and custom hybrid architectures (Mamba2, MLA, MoE). Loads SafeTensors, AWQ, GPTQ, and GGUF formats.

## Features

- **Multi-Architecture** - Llama, Mistral, Mamba2, MLA+MoE, hybrid models, and more
- **Multi-Format** - SafeTensors (F16/BF16), AWQ INT4, GPTQ INT4, GGUF (23 quantization levels)
- **Auto-Detection** - Detects architecture, format, and tokenizer from checkpoint tensor names
- **OpenAI-Compatible API** - Drop-in replacement with `/v1/completions` and `/v1/chat/completions`
- **Streaming** - Server-Sent Events (SSE) for real-time token generation
- **HuggingFace Hub** - Pull models directly from HuggingFace
- **Production Features** - Rate limiting, request timeouts, graceful shutdown, CORS, error handling
- **CUDA Acceleration** - Optional GPU inference with optimized quantization kernels

## Quick Start

### Installation

```bash
# Build (CPU-only)
cargo build --release

# Build with CUDA support (requires CUDA 12.x)
cargo build --release --features cuda
```

### Generate Text

```bash
blazr run \
  --model meta-llama/Llama-3.2-1B \
  --prompt "Once upon a time" \
  --max-tokens 100
```

### Start Server

```bash
blazr serve --model meta-llama/Llama-3.2-1B --port 8080
```

```bash
# Text completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

### Other Commands

```bash
blazr info --model ./models/mistral-7b     # Show model architecture and config
blazr list                                  # List available local models
blazr pull Qwen/Qwen2.5-0.5B               # Download from HuggingFace
```

## Supported Models

### Tested

| Model                       | SafeTensors | AWQ | GPTQ | GGUF |
| --------------------------- | ----------- | --- | ---- | ---- |
| Llama 3.2 1B                | x           | x   | x    | x    |
| Mistral 7B                  | x           | x   | x    | x    |
| Mamba2 (oxidizr)            | x           | -   | -    |      |
| DeepSeek-V2/V3 (MLA+MoE)    | x           |     |      |      |
| Hybrid Mamba+Attn (oxidizr) | x           | -   | -    |      |

### Expected Compatible (route to Llama)

Qwen2/2.5, Phi-3/3.5/4, Gemma/Gemma2, StarCoder2, Yi, InternLM2, CodeLlama, Solar

### Planned

Mixtral (MoE), Falcon (ALiBi), Command-R, GPT-NeoX/Pythia, DBRX

## Model Formats

blazr auto-detects the format from checkpoint contents:

### SafeTensors (HuggingFace)

```
model_dir/
├── model.safetensors          # or sharded: model-00001-of-00002.safetensors
├── config.json                # HuggingFace model config
└── tokenizer_config.json      # Chat template (optional)
```

### SafeTensors (oxidizr)

```
checkpoint_dir/
├── model.safetensors
└── config.json                # oxidizr config (optional, inferred from tensors)
```

### GGUF

```
model.gguf                     # Single file: weights + tokenizer + config
```

Supports all 23 GGUF quantization levels (Q2_K through Q8_0, IQ series, TQ series). CPU has dedicated kernels for all formats. CUDA has optimized dp4a kernels for Q4_K, Q6_K, Q8_0 with generic fallback for the rest.

## API Reference

### Endpoints

| Method | Path                   | Description        |
| ------ | ---------------------- | ------------------ |
| GET    | `/health`              | Health check       |
| GET    | `/v1/models`           | List loaded models |
| POST   | `/v1/completions`      | Text completion    |
| POST   | `/v1/chat/completions` | Chat completion    |

### Request Parameters

```json
{
  "prompt": "text",
  "messages": [{ "role": "user", "content": "text" }],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop": ["\n\n"],
  "stream": false
}
```

### Response Format

Non-streaming responses follow the OpenAI format:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "choices": [
    {
      "text": "generated text",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  }
}
```

Streaming responses use SSE with `data: {...}` chunks and `data: [DONE]` sentinel.

## CLI Reference

```
blazr run       Generate text from a prompt
blazr serve     Start the inference server
blazr info      Display model architecture and configuration
blazr list      List available local models
blazr pull      Download a model from HuggingFace Hub
```

### Generation Options

| Flag            | Default  | Description                                     |
| --------------- | -------- | ----------------------------------------------- |
| `--model`       | required | Local path or HuggingFace model ID              |
| `--prompt`      | required | Input text                                      |
| `--max-tokens`  | 100      | Maximum tokens to generate                      |
| `--temperature` | 0.7      | Sampling temperature (0 = greedy)               |
| `--top-p`       | 0.9      | Nucleus sampling threshold                      |
| `--top-k`       | 40       | Top-k sampling                                  |
| `--vocab`       | auto     | Tokenizer vocabulary (auto-detected from model) |
| `--cpu`         | false    | Force CPU inference                             |

### Server Options

| Flag      | Default  | Description                        |
| --------- | -------- | ---------------------------------- |
| `--model` | required | Local path or HuggingFace model ID |
| `--port`  | 8080     | Port to listen on                  |
| `--host`  | 0.0.0.0  | Host to bind to                    |
| `--cpu`   | false    | Force CPU inference                |

## Tokenizer

Uses [splintr](https://github.com/ml-rust/splintr) for BPE tokenization. Vocabulary is auto-detected from model config or can be specified with `--vocab`.

| Vocabulary    | Models               | Vocab Size |
| ------------- | -------------------- | ---------- |
| `llama3`      | Llama 3.x, Mistral   | ~128k      |
| `cl100k_base` | GPT-4, GPT-3.5-turbo | ~100k      |
| `o200k_base`  | GPT-4o               | ~200k      |
| `deepseek_v3` | DeepSeek V3/R1       | ~129k      |

GGUF files include an embedded tokenizer which is extracted automatically.

## Architecture

blazr is a thin application layer built on [boostr](https://github.com/ml-rust/boostr) (ML framework) and [numr](https://github.com/ml-rust/numr) (numerical computing). All model architectures, tensor operations, and quantization kernels live in boostr. blazr provides the CLI, HTTP server, model loading, and request orchestration.

```
blazr (CLI + HTTP server + model lifecycle)
  |
boostr (model architectures + quant kernels + NN modules)
  |
numr (tensors + linalg + multi-backend: CPU/CUDA/WebGPU)
```

## Requirements

- Rust 1.70+
- (Optional) CUDA 12.x for GPU acceleration

## License

Apache-2.0 - see [LICENSE](LICENSE) for details.

## Related Projects

- [boostr](https://github.com/ml-rust/boostr) - ML framework (model architectures, quantization)
- [numr](https://github.com/ml-rust/numr) - Foundational numerical computing (tensors, linalg, multi-backend)
- [oxidizr](https://github.com/ml-rust/oxidizr) - Training framework for hybrid architectures
- [splintr](https://github.com/ml-rust/splintr) - High-performance BPE tokenizer
- [compressr](https://github.com/ml-rust/compressr) - Model conversion and compression

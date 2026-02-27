# Architecture

blazr is designed to efficiently run hybrid neural architectures that mix different layer types within a single model. This document explains how the inference engine works and what makes it unique.

## Overview

blazr supports four layer types that can be freely mixed:

1. **Mamba2** - State Space Models with selective state spaces
2. **MLA** - Multi-Head Latent Attention with compressed KV cache
3. **MoE** - Mixture of Experts with top-k routing
4. **Standard Transformers** - Traditional GQA + MLP layers

Most inference engines (llama.cpp, vLLM, etc.) only support standard transformers. blazr's architecture enables running oxidizr-trained hybrid models that combine these approaches.

## Auto-Detection

When loading a model, blazr automatically detects the architecture by inspecting tensor names in the SafeTensors checkpoint:

```rust
// Detects Mamba2 layers
if tensors.contains("model.layers.0.mamba.in_proj.weight") {
    layer_types.push(LayerType::Mamba2);
}

// Detects MLA layers
if tensors.contains("model.layers.0.attn.kv_a_proj_with_mqa.weight") {
    layer_types.push(LayerType::MlaWithMoe);
}

// Detects standard transformer layers
if tensors.contains("model.layers.0.attn.q_proj.weight") {
    layer_types.push(LayerType::StandardTransformer);
}
```

This means you can load a model without any configuration file - blazr will figure it out.

## Layer Types

### Mamba2 (State Space Model)

Mamba2 layers use a selective state space mechanism instead of attention:

**Components:**
- `in_proj` - Projects hidden states to internal dimension (typically 2x hidden size)
- `conv1d` - Causal 1D convolution for local context
- `x_proj`, `dt_proj`, `A_log`, `B`, `C`, `D` - SSM parameters
- `out_proj` - Projects back to hidden size
- `norm` - RMSNorm pre-normalization

**State:**
- `ssm_state`: `[batch, n_heads, head_dim, state_size]` - Recurrent hidden state
- `conv_state`: `[batch, d_inner, conv_kernel-1]` - Causal convolution buffer

**Processing:**
- **Prefill** - Processes sequences in chunks (default: 256 tokens) with parallel SSM
- **Decode** - Sequential processing, updating state at each step

**Advantages:**
- O(N) complexity (vs O(N²) for attention)
- Constant memory KV cache (state size independent of sequence length)
- Fast sequential generation

### MLA (Multi-Head Latent Attention)

MLA compresses key-value representations into a low-rank latent space:

**Components:**
- `kv_a_proj_with_mqa` - Projects to KV latent dimension
- `kv_b_proj` - Expands latents to full key/value
- `q_a_proj` - Projects to Q latent dimension (with RoPE)
- `q_b_proj` - Expands to full query
- `o_proj` - Output projection
- `norm` - RMSNorm pre-normalization

**Dimensions:**
```
hidden_size = 1024
kv_latent_dim = 256    # Compressed KV (4x compression)
q_latent_dim = 384     # Compressed Q
num_heads = 16
d_rope = 64            # RoPE applied to Q latents
```

**Cache:**
- Stores compressed latents instead of full K/V tensors
- Typical compression: 4-8x smaller than standard KV cache
- Expands on-the-fly during attention computation

**Advantages:**
- Reduced memory footprint for long contexts
- Lower bandwidth requirements
- Maintains full attention expressiveness

### MoE (Mixture of Experts)

Fine-grained MoE with top-k routing:

**Components:**
- `gate` - Router network (hidden_size → num_experts)
- `experts[i].gate_proj` - Expert input projection
- `experts[i].up_proj` - Expert up-projection
- `experts[i].down_proj` - Expert output projection
- `shared_expert.*` - Always-active shared expert (optional)

**Routing:**
```python
# Top-k selection (typically k=2)
scores = softmax(gate(x))
top_k_scores, top_k_indices = topk(scores, k)

# Compute weighted expert outputs
output = sum(score * expert(x) for score, expert in zip(top_k_scores, selected_experts))

# Add shared expert if enabled
if shared_expert:
    output += shared_expert(x)
```

**Advantages:**
- Sparse computation (only k experts active per token)
- High model capacity without proportional compute cost
- Shared expert provides universal knowledge

### Standard Transformer

Traditional attention + MLP:

**Components:**
- `attn.q_proj`, `attn.k_proj`, `attn.v_proj` - Query/Key/Value projections
- `attn.o_proj` - Output projection
- `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` - Feed-forward network
- GQA (Grouped Query Attention) support

**Cache:**
- Standard KV cache: `[batch, num_heads, seq_len, head_dim]`

## Hybrid Model Example

A typical hybrid architecture (e.g., 12-layer model):

```
Layer 0:  Mamba2
Layer 1:  Mamba2
Layer 2:  MLA + MoE
Layer 3:  Mamba2
Layer 4:  MLA + MoE
Layer 5:  Mamba2
Layer 6:  MLA + MoE
Layer 7:  Mamba2
Layer 8:  MLA + MoE
Layer 9:  Mamba2
Layer 10: MLA + MoE
Layer 11: MLA + MoE
```

**Why hybrid?**
- Mamba2 provides efficient local context modeling
- MLA handles long-range dependencies
- MoE adds specialized knowledge
- Together: better quality/compute tradeoff than pure approaches

## State Management

blazr maintains state across generation steps:

**InferenceState:**
```rust
struct InferenceState {
    mamba_states: Vec<Option<Mamba2State>>,  // Per-layer Mamba2 state
    mla_caches: Vec<Option<MlaCache>>,       // Per-layer MLA KV cache
    position: usize,                         // Current sequence position
}
```

**Prefill vs Decode:**

| Phase | Input | Mamba2 | MLA |
|-------|-------|--------|-----|
| Prefill | Full prompt | Chunked processing | Standard attention |
| Decode | Single token | Sequential update | Cached KV attention |

## Model Loading

blazr loads SafeTensors checkpoints:

```
checkpoint_dir/
├── model.safetensors
└── config.json (optional)
```

**Loading Process:**
1. Check for `config.json` (optional)
2. Auto-detect architecture from tensor names
3. Merge detected config with loaded config (if present)
4. Initialize model layers based on detected types
5. Load weights from SafeTensors

**Why SafeTensors?**
- Fast mmap-based loading
- Built-in safety (no code execution)
- Cross-framework compatibility
- Used by Hugging Face ecosystem

## Tokenization

blazr uses [splintr](https://github.com/farhan/splintr) for tokenization:

- BPE tokenizer compatible with Llama 3 vocabulary
- ~128k tokens
- PCRE2 regex backend with JIT compilation
- Handles special tokens for chat format

## Performance Considerations

**Memory:**
- Mamba2: O(1) state size (independent of sequence length)
- MLA: O(N) compressed cache (4-8x smaller than standard KV)
- MoE: Sparse activation (only k experts per token)

**Compute:**
- Mamba2: O(N) time complexity
- MLA: O(N²) but with smaller hidden dimensions
- MoE: O(k/n_experts) fraction of total MLP compute

**Batching:**
- Currently single-request inference
- Batch inference planned for future releases
- Mamba2 makes batching more complex (state-dependent)

## CUDA Support

When built with `--features cuda`:

- Auto-detects CUDA device 0
- Falls back to CPU if CUDA unavailable
- Same model code (device-agnostic tensors)
- Typical speedup: 10-50x vs CPU

**Requirements:**
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- CUDA 12.x
- cuDNN (bundled with Candle)

## Future Enhancements

Planned improvements:

- **Streaming** - Server-sent events for token-by-token generation
- **Batching** - Continuous batching for higher throughput
- **Quantization** - INT8/INT4 quantization for reduced memory
- **Flash Attention** - Faster attention kernels
- **Speculative Decoding** - Draft-verify for faster generation
- **Multiple Models** - Load and serve multiple models simultaneously

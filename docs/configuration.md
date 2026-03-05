# Configuration

This document describes model configuration options and inference settings for blazr.

## Model Configuration

blazr can load configuration from `config.json` in the checkpoint directory, or auto-detect it from tensor names.

### config.json Format

```json
{
  "hidden_size": 1024,
  "num_layers": 12,
  "vocab_size": 128256,
  "max_seq_len": 4096,
  "rms_norm_eps": 1e-5,

  "mamba2_num_heads": 16,
  "mamba2_head_dim": 64,
  "mamba2_state_size": 64,
  "mamba2_chunk_size": 256,
  "mamba2_expand": 2,
  "mamba2_conv_kernel": 4,
  "mamba2_n_groups": 1,

  "num_attention_heads": 16,
  "kv_latent_dim": 256,
  "q_latent_dim": 384,
  "d_rope": 64,

  "num_experts": 8,
  "experts_per_tok": 2,
  "shared_expert_enabled": true,
  "intermediate_size": 4096,

  "num_kv_heads": 4
}
```

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hidden_size` | int | Yes | Model hidden dimension (e.g., 1024, 2048, 4096) |
| `num_layers` | int | Yes | Total number of transformer layers |
| `vocab_size` | int | Yes | Vocabulary size (typically 128256 for Llama 3) |
| `max_seq_len` | int | No | Maximum sequence length (default: 4096) |
| `rms_norm_eps` | float | No | RMSNorm epsilon (default: 1e-5) |

### Mamba2 Parameters

Only required for layers using Mamba2:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mamba2_num_heads` | int | - | Number of SSM heads |
| `mamba2_head_dim` | int | - | Dimension per head |
| `mamba2_state_size` | int | - | SSM state dimension (N) |
| `mamba2_chunk_size` | int | 256 | Chunk size for prefill |
| `mamba2_expand` | int | 2 | Expansion factor for inner dimension |
| `mamba2_conv_kernel` | int | 4 | Conv1D kernel size |
| `mamba2_n_groups` | int | 1 | Number of SSM groups |

**Relationships:**
- `d_inner = hidden_size * mamba2_expand`
- Total SSM dimension: `mamba2_num_heads * mamba2_head_dim`

### MLA Parameters

Only required for layers using Multi-Head Latent Attention:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_attention_heads` | int | - | Number of attention heads |
| `kv_latent_dim` | int | - | Compressed KV dimension |
| `q_latent_dim` | int | - | Compressed Q dimension |
| `d_rope` | int | - | RoPE dimension (applied to Q latents) |

**Typical values:**
- `kv_latent_dim` ≈ hidden_size / 4 (4x compression)
- `q_latent_dim` ≈ hidden_size / 2.5
- `d_rope` = 64 or 128

### MoE Parameters

Only required for layers using Mixture of Experts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | int | - | Total number of experts |
| `experts_per_tok` | int | 2 | Number of experts activated per token |
| `shared_expert_enabled` | bool | false | Whether to use always-active shared expert |
| `intermediate_size` | int | - | Expert hidden dimension |

**Typical values:**
- `num_experts` = 8, 16, 32, 64
- `experts_per_tok` = 2 (standard)
- `intermediate_size` ≈ 4 * hidden_size

### Standard Transformer Parameters

Only required for standard transformer layers:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_kv_heads` | int | - | Number of KV heads for GQA |

For standard MHA (Multi-Head Attention), set `num_kv_heads = num_attention_heads`.

For GQA (Grouped Query Attention), use fewer KV heads (e.g., 4-8).

### Layer Types

You can manually specify layer types (optional):

```json
{
  "layer_types": [
    "Mamba2",
    "Mamba2",
    "MlaWithMoe",
    "Mamba2",
    "MlaWithMoe"
  ]
}
```

Or use the legacy `mamba_layers` format:

```json
{
  "mamba_layers": [0, 1, 3]
}
```

If neither is provided, blazr auto-detects from tensor names.

## Inference Settings

### Sampling Configuration

Control generation behavior via CLI flags or API parameters:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_tokens` | int | 1-∞ | 100 | Maximum tokens to generate |
| `temperature` | float | 0.0-2.0 | 0.7 | Sampling temperature (higher = more random) |
| `top_p` | float | 0.0-1.0 | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 1-∞ | 40 | Top-k sampling (consider only k most likely tokens) |

**Temperature Effects:**
- `0.0` - Greedy decoding (deterministic)
- `0.3-0.7` - Focused, coherent text
- `0.7-1.0` - Balanced creativity
- `1.0-2.0` - More random, creative

**Top-p (Nucleus) Sampling:**
- `0.9` - Keep tokens covering 90% of probability mass
- `0.95` - More conservative (common setting)
- `1.0` - Disable (use all tokens)

**Top-k Sampling:**
- `40` - Consider top 40 tokens
- `0` - Disable top-k filtering
- Combines with top-p (both filters applied)

### CLI Examples

```bash
# Conservative generation
blazr generate --model ./model \
  --prompt "Explain quantum computing" \
  --temperature 0.3 \
  --top-p 0.95 \
  --max-tokens 200

# Creative generation
blazr generate --model ./model \
  --prompt "Write a poem about" \
  --temperature 1.2 \
  --top-p 0.9 \
  --max-tokens 100

# Greedy (deterministic)
blazr generate --model ./model \
  --prompt "2 + 2 = " \
  --temperature 0.0 \
  --max-tokens 5
```

## Server Configuration

### Command-Line Flags

```bash
blazr serve \
  --model ./checkpoints/nano \
  --host 0.0.0.0 \
  --port 8080 \
  --cpu  # Force CPU (skip CUDA even if available)
```

### Environment Variables

Control logging with `RUST_LOG`:

```bash
# Info level (default)
RUST_LOG=blazr=info blazr serve --model ./model

# Debug level
RUST_LOG=blazr=debug,tower_http=debug blazr serve --model ./model

# Trace level (very verbose)
RUST_LOG=blazr=trace blazr serve --model ./model
```

## Device Selection

blazr automatically selects the best available device:

1. If `--cpu` flag: Use CPU
2. If CUDA available and enabled: Use CUDA device 0
3. Otherwise: Fall back to CPU

**Check device:**
```
[2024-12-03T10:52:30Z INFO  blazr] Using CUDA device 0
```

or

```
[2024-12-03T10:52:30Z INFO  blazr] Using CPU (CUDA feature not enabled)
```

## Performance Tuning

### Memory Usage

**Mamba2 state:**
```
memory = num_mamba_layers * batch_size * num_heads * head_dim * state_size * 4 bytes
```

Example: 6 Mamba layers, 16 heads, 64 head_dim, 64 state_size
```
6 * 1 * 16 * 64 * 64 * 4 = 1.5 MB (negligible)
```

**MLA cache:**
```
memory = num_mla_layers * batch_size * kv_latent_dim * seq_len * 4 bytes
```

Example: 6 MLA layers, 256 latent_dim, 4096 seq_len
```
6 * 1 * 256 * 4096 * 4 = 24 MB
```

**Model weights:**
```
memory ≈ num_params * 2 bytes (FP16)
```

Example: 1B parameter model
```
1B * 2 = 2 GB
```

### Chunk Size Tuning

For Mamba2 layers, adjust `mamba2_chunk_size`:

- Larger (512, 1024): Faster prefill, more memory
- Smaller (128, 256): Slower prefill, less memory
- Default (256): Good balance

### Batch Size

Currently, blazr only supports `batch_size = 1` (single-request inference).

Batch inference will be added in a future release.

## Validation

Check your configuration:

```bash
blazr info --model ./checkpoints/nano
```

Output:
```json
{
  "hidden_size": 1024,
  "num_layers": 12,
  "vocab_size": 128256,
  "layer_types": ["Mamba2", "Mamba2", "MlaWithMoe", ...],
  ...
}
```

Verify:
- All required parameters are present
- Layer-specific parameters match detected layer types
- Dimensions are consistent (e.g., `num_heads * head_dim` matches expected sizes)

## Troubleshooting

### Missing Configuration

**Error:**
```
called `Option::unwrap()` on a `None` value
```

**Cause:** Required parameter not set and couldn't be auto-detected.

**Solution:** Add to `config.json` or ensure tensor names follow expected patterns.

### Dimension Mismatch

**Error:**
```
Shape mismatch: expected [1024, 2048], got [1024, 4096]
```

**Cause:** Configuration doesn't match actual tensor shapes.

**Solution:** Verify `hidden_size`, `intermediate_size`, expansion factors.

### OOM (Out of Memory)

**Cause:** Model too large for available GPU memory.

**Solutions:**
- Use `--cpu` flag
- Reduce `max_seq_len` (if generating very long sequences)
- Close other GPU applications
- Use quantization (planned future feature)

### Slow Inference

**Cause:** Running on CPU or CUDA not properly enabled.

**Solutions:**
- Build with `--features cuda`
- Verify CUDA toolkit installed
- Check GPU is detected: Look for "Using CUDA device 0" in logs

# Model compatibility

blazr loads text, multimodal, embedding, and speech models from SafeTensors, AWQ, GPTQ, and GGUF files.

## Evidence levels

Each table cell names one level.

| Level               | Meaning                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `Tested`            | A repo test or the production checklist names a real checkpoint for this entry.       |
| `Supported in code` | The loader, detector, or architecture handles this entry. No test names a checkpoint. |
| `Routes to <arch>`  | Config maps this model type to architecture `<arch>`. No test covers the route.       |

A dash (`—`) marks a format the architecture lacks, or a format blazr does not load for it.

Tests that name a checkpoint through an environment variable skip when that variable is unset.

## Text architectures

### By format

| Architecture                                | `model_type` (GGUF `general.architecture`)             | SafeTensors         | AWQ                 | GPTQ                | GGUF                | Notes                                                                                                                                                                                     |
| ------------------------------------------- | ------------------------------------------------------ | ------------------- | ------------------- | ------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 2, 3, 3.1, 3.2                        | `llama` (`llama`, `llama2`, `llama3`)                  | `Tested`            | `Supported in code` | `Supported in code` | `Supported in code` | Llama 3.2 1B, 32 of 32 ids. Gated by `BLAZR_TEST_LLAMA_MODEL`.                                                                                                                            |
| Mistral 7B                                  | `mistral` (`mistral`)                                  | `Tested`            | `Tested`            | `Tested`            | `Tested`            | Named in the production checklist. CPU runs of GGUF Q2_K, Q4_K_M and Q8_0, AWQ and GPTQ give coherent text. No local HF weights exist for a reference.                                    |
| Yi, CodeLlama, Solar                        | `llama` (`llama`)                                      | `Supported in code` | `Supported in code` | `Supported in code` | `Supported in code` | Same path as Llama. No test names these models.                                                                                                                                           |
| Mamba2                                      | `mamba2` (`mamba`, `mamba2`)                           | `Supported in code` | —                   | —                   | `Supported in code` | No attention layers, so no AWQ or GPTQ.                                                                                                                                                   |
| Mamba3                                      | `mamba3` (`mamba3`)                                    | `Supported in code` | —                   | —                   | `Supported in code` | No attention layers, so no AWQ or GPTQ.                                                                                                                                                   |
| DeepSeek-V2, DeepSeek-V3 (MLA + MoE)        | `deepseek_v2`, `deepseek_v3` (`deepseek`, `deepseek2`) | `Supported in code` | `Supported in code` | `Supported in code` | `Supported in code` | Needs a generate test. GGUF `deepseek` maps to model type `deepseek`.                                                                                                                     |
| Hybrid Mamba + attention                    | `hybrid` (oxidizr layout)                              | `Supported in code` | —                   | —                   | —                   | No GGUF architecture name maps to `hybrid`. No AWQ or GPTQ path.                                                                                                                          |
| Qwen3.5 hybrid (Gated DeltaNet + attention) | `qwen35` (`qwen35`)                                    | `Supported in code` | —                   | —                   | `Tested`            | SafeTensors needs `model_type` `qwen35` in `config.json`. GGUF test: `tests/qwen35_generate.rs`, gated by `BOOSTR_BONSAI2_DIR`. No AWQ or GPTQ path found.                                |
| Qwen2, Qwen2.5                              | `qwen2` (`qwen2`)                                      | `Tested`            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | Qwen2.5-0.5B, 32 of 32 ids. `BLAZR_TEST_QWEN2_MODEL`.                                                                                                                                     |
| Qwen2-MoE                                   | `qwen2_moe`                                            | `Tested`            | `Routes to llama`   | `Routes to llama`   | —                   | Random-weight `hf-tiny-v2/tiny-random-Qwen2MoeForCausalLM`: 32 of 32 ids and logits. `BLAZR_TEST_QWEN2_MOE_MODEL`. GGUF architecture `qwen2moe` is not accepted.                          |
| Phi-2                                       | `phi`                                                  | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | —                   | GGUF architecture `phi` is not accepted.                                                                                                                                                  |
| Phi-3, Phi-3.5, Phi-4                       | `phi3` (`phi3`)                                        | `Tested`            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | Phi-3.5-mini-instruct, 32 of 32 ids. `BLAZR_TEST_PHI3_MODEL`. Phi-4 is untested.                                                                                                          |
| Gemma, Gemma 2                              | `gemma`, `gemma2` (`gemma`, `gemma2`)                  | `Tested`            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | Gemma 2 2B (`unsloth/gemma-2-2b` mirror), 32 of 32 ids. `BLAZR_TEST_GEMMA2_MODEL`. Gemma 1 shares the path and is untested.                                                               |
| StarCoder2                                  | `starcoder2` (`starcoder2`)                            | `Tested`            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | `TechxGenus/starcoder2-3b-instruct`, a StarCoder2-3B fine-tune, 32 of 32 ids. `BLAZR_TEST_STARCODER2_MODEL`.                                                                              |
| InternLM2                                   | `internlm2`                                            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | —                   | No name map for `attention.wqkv` or `feed_forward.*`, so the load fails.                                                                                                                  |
| Mixtral                                     | `mixtral`                                              | `Tested`            | `Routes to llama`   | `Routes to llama`   | —                   | `axolotl-ai-co/tiny-mixtral-30m`, a trained 30M Mixtral, 32 of 32 ids. `BLAZR_TEST_MIXTRAL_MODEL`. The transformers v5 fused `gate_up_proj` expert layout is not read.                    |
| Falcon                                      | `falcon` (`falcon`)                                    | `Tested`            | `Routes to llama`   | `Routes to llama`   | `Routes to llama`   | Falcon-RW-1B (`petals-team/falcon-rw-1b` SafeTensors copy): ALiBi, sequential block, 32 of 32 ids. `BLAZR_TEST_FALCON_MODEL`. The parallel Falcon-7B and Falcon-40B layouts are untested. |
| Command-R                                   | `cohere`, `command_r`                                  | `Tested`            | `Routes to llama`   | `Routes to llama`   | —                   | Random-weight `trl-internal-testing/tiny-CohereForCausalLM`: 32 of 32 ids and logits. `BLAZR_TEST_COHERE_MODEL`. `use_qk_norm: true` (Command-R+) is refused.                             |
| GPT-NeoX, Pythia                            | `gpt_neox`                                             | `Tested`            | `Routes to llama`   | `Routes to llama`   | —                   | Pythia-160M, 32 of 32 ids. `BLAZR_TEST_GPT_NEOX_MODEL`.                                                                                                                                   |
| DBRX                                        | `dbrx`                                                 | `Tested`            | `Routes to llama`   | `Routes to llama`   | —                   | Random-weight `trl-internal-testing/tiny-DbrxForCausalLM`: 32 of 32 ids and logits. `BLAZR_TEST_DBRX_MODEL`.                                                                              |

Any other model type that has an attention config loads through the Llama path. That route is untested.

The `Tested` SafeTensors cells with a `BLAZR_TEST_*_MODEL` variable run `tests/arch_reference.rs`. They load SafeTensors on the CPU in F32. Each test decodes 32 greedy tokens and compares the first 8 ids with `transformers`. All 32 ids match on every listed checkpoint. Random-weight checkpoints also compare the last prompt position's logits. AWQ, GPTQ and GGUF files of these architectures are untested.

Load paths:

- `boostr::format::restructure_hf_checkpoint` renames the tensors, splits fused QKV and gate-up projections, and stacks MoE experts.
- It also applies the Gemma norm offset, the query scale, and the partial or interleaved RoPE row order.
- The block config that blazr builds from `config.json` carries the rest: LayerNorm, MLP activation, residual layout, softcaps, `logit_scale`, `clip_qkv`, the ALiBi scale and sliding layers.
- Paged attention refuses ALiBi and softcapped models. It refuses a sliding window shorter than the sequence.
- CUDA graph mode and tensor parallelism run the Llama block only. They refuse every other block.

### Quantized weight formats

- AWQ and GPTQ read 4-bit weights only.
- The group size comes from `quant_config.json`, `quantize_config.json`, or the `quantization_config` block in `config.json`. The default is `128`.
- The AWQ and GPTQ loaders do not check the architecture. Their support follows the architecture row.

## GGUF quantization formats

The table lists every format in `boostr/src/quant/format.rs`.

- **Dedicated** names a per-format kernel. CPU dequantization uses a dedicated kernel for every format.
- **Generic** names the `dequant_generic_f32` kernel, which dispatches on format ID.

| Format  | GGML type ID | Block elements | Block bytes | CPU dequant | CUDA dequant |
| ------- | ------------ | -------------- | ----------- | ----------- | ------------ |
| Q4_0    | 2            | 32             | 18          | Dedicated   | Dedicated    |
| Q4_1    | 3            | 32             | 20          | Dedicated   | Generic      |
| Q5_0    | 6            | 32             | 22          | Dedicated   | Dedicated    |
| Q5_1    | 7            | 32             | 24          | Dedicated   | Generic      |
| Q8_0    | 8            | 32             | 34          | Dedicated   | Dedicated    |
| Q8_1    | 9            | 32             | 36          | Dedicated   | Generic      |
| Q2_K    | 10           | 256            | 84          | Dedicated   | Dedicated    |
| Q3_K    | 11           | 256            | 110         | Dedicated   | Dedicated    |
| Q4_K    | 12           | 256            | 144         | Dedicated   | Dedicated    |
| Q5_K    | 13           | 256            | 176         | Dedicated   | Dedicated    |
| Q6_K    | 14           | 256            | 210         | Dedicated   | Dedicated    |
| Q8_K    | 15           | 256            | 292         | Dedicated   | Generic      |
| IQ2_XXS | 16           | 256            | 66          | Dedicated   | Generic      |
| IQ2_XS  | 17           | 256            | 74          | Dedicated   | Dedicated    |
| IQ3_XXS | 18           | 256            | 98          | Dedicated   | Generic      |
| IQ1_S   | 19           | 256            | 50          | Dedicated   | Generic      |
| IQ4_NL  | 20           | 32             | 18          | Dedicated   | Dedicated    |
| IQ3_S   | 21           | 256            | 110         | Dedicated   | Dedicated    |
| IQ2_S   | 22           | 256            | 82          | Dedicated   | Generic      |
| IQ4_XS  | 23           | 256            | 136         | Dedicated   | Dedicated    |
| IQ1_M   | 29           | 256            | 56          | Dedicated   | Generic      |
| TQ1_0   | 34           | 256            | 54          | Dedicated   | Generic      |
| TQ2_0   | 35           | 256            | 66          | Dedicated   | Generic      |
| Q1_0    | 41           | 128            | 18          | Dedicated   | Generic      |
| Q2_0    | 42           | 64             | 18          | Dedicated   | Generic      |
| PQ2_0   | 142          | 128            | 34          | Dedicated   | Generic      |
| PTQ1_0  | 143          | 128            | 28          | Dedicated   | Generic      |

- `Q1_0`, `Q2_0`, `PQ2_0`, and `PTQ1_0` have no upstream ggml type ID. The IDs above are the values the files carry.
- Quantized matrix multiplication uses separate kernels. This table covers dequantization only.
- A GGUF tensor with a type outside this table fails to load.

## Multimodal, embedding, rerank, and speech

### Multimodal (image input)

| Model                                                   | Feature gate | Required files                                               | Evidence            | Notes                                                                                            |
| ------------------------------------------------------- | ------------ | ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------------------------------------------ |
| Qwen3-VL vision tower with a `qwen35` text model        | none         | Text GGUF and Qwen3-VL mmproj GGUF, passed with `--mmproj`   | `Tested`            | Test: `tests/qwen35_vision_generate.rs`, gated by `BOOSTR_BONSAI2_DIR`.                          |
| LLaVA-style CLIP or SigLIP encoder with an LLM backbone | none         | `config.json` with a `vision` block, and `model.safetensors` | `Supported in code` | Model type `llava` is inferred from `architectures`. Image serving for this path is unconfirmed. |
| Standalone SigLIP or CLIP encoder                       | none         | Directory with `config.json` and one `model.safetensors`     | `Supported in code` | Loaded by `serve --vision-model NAME=PATH`. Sharded checkpoints are not supported.               |

The `--mmproj` flag documents `qwen35` models only.

### Embeddings and rerank

| Endpoint                          | Model                 | Approach                                                            | Evidence            |
| --------------------------------- | --------------------- | ------------------------------------------------------------------- | ------------------- |
| `POST /v1/embeddings`             | Any loaded text model | Token embeddings with pooling `mean`, `cls`, `last`, or `none`      | `Supported in code` |
| `POST /rerank`, `POST /v1/rerank` | Any loaded text model | Cosine similarity between the pooled query and each pooled document | `Supported in code` |

- Rerank scores come from embedding similarity. Rerank does not run a cross-encoder.
- Dedicated encoder models, such as BERT, nomic-bert, and jina-bert, have loaders in boostr. blazr does not call those loaders.

### Speech

Speech endpoints and commands need the `audio` feature. A missing model downloads on first use into the Hugging Face hub layout.

| Model                                                   | Task                                                     | Required files                                                                                                                             | Evidence | Notes                                                                                             |
| ------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------- |
| Whisper (`openai/whisper-base`, `openai/whisper-small`) | ASR: `POST /v1/audio/transcriptions`, `blazr transcribe` | `config.json`, `tokenizer.json`, `model.safetensors`, `generation_config.json`                                                             | `Tested` | Test: `tests/audio_transcription_words.rs`, uses `whisper-tiny`. F16 weights cast to F32 at load. |
| Kokoro (`hexgrad/Kokoro-82M`)                           | TTS: `POST /v1/audio/speech`, `blazr speak`              | `config.json`, `vocab.json`, `model.safetensors` or `.pth`, optional `voices/`, `g2p/tagger/`, `g2p/oov/`                                  | `Tested` | Tests: `tests/narration/speak_kokoro.rs`, `tests/tts_cuda.rs`, gated by `KOKORO_DIR`. CPU or GPU. |
| VITS, MMS-TTS (`facebook/mms-tts-*`)                    | TTS                                                      | `config.json` with `model_type` `vits`, `vocab.json`, `tokenizer_config.json`, `model.safetensors`                                         | `Tested` | Tests: `speak_vits.rs`, `tests/tts_cuda.rs`, gated by `MMS_TTS_ZLM_DIR`. One voice. CPU or GPU.   |
| VoxCPM2 (`openbmb/VoxCPM2`)                             | TTS with voice cloning                                   | `config.json` with `dit_config`, `model.safetensors` or one `.tcf` or `.gguf`, AudioVAE, optional `tokenizer.json`, `voices/`, `adapters/` | `Tested` | Test: `tests/narration/speak_voxcpm.rs`, gated by `VOXCPM2_GGUF`. Default device on a CUDA build. |

- The Whisper vocabulary follows the audio model variant. Blazr does not select it by size.
- boostr contains a Silero VAD model. blazr does not load it.

### Models fetched on first use

A model the speech commands need, and do not find on disk, fetches once from the Hugging Face hub:

| Model                                   | Used by                                      |
| --------------------------------------- | -------------------------------------------- |
| `openai/whisper-base`                   | `transcribe` and `speak` for English         |
| `openai/whisper-small`                  | `transcribe` and `speak` for other languages |
| `hexgrad/Kokoro-82M`, `fs90/kokoro-g2p` | `speak` with Kokoro                          |
| `openbmb/VoxCPM2`                       | `speak` with VoxCPM2 (CUDA build with a GPU) |
| `laion/larger_clap_general`             | `sound-review`                               |

Fetched files land in hub layout in the model folder: `$MOTION_VIDEO_MODELS`, else the `models` entry of `~/.config/motion-video/config.json`, else `~/.cache/motion-video/models`. The search also reads `$BOOSTR_MODELS_DIR` and the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). `blazr serve` fetches only Kokoro's G2P models on first use, into the Hugging Face cache. Set `HF_HUB_OFFLINE=1` to forbid every fetch: a missing model then fails with its name and the `hf download` command that fetches it.

### Devices

On the CUDA build (`--features audio,cuda`) with an NVIDIA GPU, `speak` runs Kokoro, VITS, VoxCPM2 and Whisper on the first GPU. `transcribe` runs Whisper there, and `sound-review` runs CLAP there. `speak --device cpu` keeps every model of the run on the CPU. Every other build runs them all on the CPU. A Kokoro or VITS take follows `--seed`: one seed gives the same take on one device, and a different take on the CPU and the GPU.

`blazr serve` loads every `--asr-model` and `--tts-model` on the server's device: the configured CUDA device on the CUDA build, else the CPU.

`speak` picks VoxCPM2 by default only on the CUDA build with an NVIDIA GPU with 8 GB of memory. Elsewhere it picks Kokoro. `--engine voxcpm --device cpu` runs VoxCPM2 on the CPU, at a small fraction of GPU speed.

## Tokenizers

### Text checkpoints

Blazr runs a checkpoint with its own `tokenizer.json`. Without one, it selects a bundled vocabulary from the checkpoint's `vocab_size`.

| Vocabulary    | Base size | Models named in source         |
| ------------- | --------- | ------------------------------ |
| `cl100k_base` | 100277    | GPT-4, GPT-3.5-turbo           |
| `o200k_base`  | 200019    | GPT-4o                         |
| `llama3`      | 128256    | Llama 3.x                      |
| `deepseek_v3` | 128815    | DeepSeek V3, R1                |
| `mistral_v1`  | 32000     | Mistral 7B v0.1, v0.2, Mixtral |
| `mistral_v2`  | 32768     | Mistral 7B v0.3, Codestral     |
| `mistral_v3`  | 131072    | Mistral NeMo, Large 2, Pixtral |

- A checkpoint matches a vocabulary by its base size or its extended size. splintr adds agent tokens to the extended size.
- A size that matches no vocabulary fails. The error lists the known vocabularies. Blazr does not guess.
- Custom vocabularies fail with an error. Blazr does not read a `.tiktoken` file.
- A checkpoint with a `tokenizer.json` runs with it, and size matching does not run. A `tokenizer.json` that does not load is an error.
- The text path reads no `tokenizer.model`.
- The `tokenizer_vocab` field in `config` is not read by the load path.

### GGUF checkpoints

A GGUF file embeds its vocabulary. Blazr reads the vocabulary from the GGUF metadata with boostr's `extract_gguf_vocab`. It builds the tokenizer with splintr's `from_gguf_vocab`. No size matching runs for GGUF files.

## Chat templates

Blazr picks one template in this order:

1. The `--chat-template` value, when set. It takes a built-in name or a template file path. An unknown value fails.
2. The model's shipped template. For a GGUF file, `tokenizer.chat_template`, then the `tokenizer.chat_templates` list. For a checkpoint directory, `chat_template` in `tokenizer_config.json`. A list selects `default`, then the first entry.
3. The built-in format for the model type.

A template file renders with the model's own `bos_token` and `eos_token` when the model ships them.

Built-in format by model type:

| Model type                     | Built-in format |
| ------------------------------ | --------------- |
| `llama`                        | `llama3`        |
| `mistral`                      | `mistral`       |
| `qwen2`, `qwen2_moe`, `qwen35` | `chatml`        |
| `phi3`, `phi`                  | `phi3`          |
| `gemma`, `gemma2`              | `gemma`         |
| `deepseek_v2`, `deepseek_v3`   | `deepseek`      |
| Any other type                 | `generic`       |

- A GGUF file with model type `deepseek` falls to `generic` when it ships no template.
- Built-in names are case-insensitive. Accepted names: `llama3`, `llama`, `mistral`, `llama2`, `chatml`, `qwen`, `yi`, `phi3`, `phi`, `gemma`, `gemma2`, `deepseek`, `generic`, `raw`.

## Architecture detection

Blazr detects the format and architecture in this order.

- For a directory, blazr loads `model.safetensors`, then `pytorch_model.safetensors`, then the first `model-00001-of-*.safetensors` shard, then the first `*.gguf` file.
- Tensor names that start with `model.` select the HuggingFace layout. Other names select the oxidizr layout, with no prefix.
- The `embed_tokens.weight` tensor sets `vocab_size` and `hidden_size`.
- A missing `lm_head.weight` sets `tie_word_embeddings`.
- Each `layers.N.` index counts as one layer. Zero layers fail with `No layers detected in checkpoint`.
- Each layer takes the first matching type:
  - `mamba3.` gives Mamba3.
  - `mamba2.` gives Mamba2.
  - `self_attn.w_dkv` with `moe.` gives MLA with MoE.
  - `self_attn.w_dkv` with `mlp.` or alone gives MLA with MLP.
  - `self_attn.q_proj` gives a standard transformer.
- A `config.json` that parses replaces the detected values. Its `model_type` selects the route.
- Without `config.json`, blazr fixes the head dimension at `128`. It divides the `q_proj` and `k_proj` output sizes by it.
- Without `config.json`, the model type follows the layers. Mamba-only layers give `mamba2` or `mamba3`. Mixed or MLA layers give `hybrid`. Other layers give `llama`.
- AWQ is detected from `quant_config.json`, from `quantization_config` in `config.json`, or from a `.qweight` tensor. GPTQ is checked first. It uses `quantize_config.json`, `quant_config.json`, `quantization_config`, or a `.g_idx` tensor.
- A GGUF file selects its architecture from `general.architecture`. An unlisted value fails with `Unsupported GGUF architecture '<name>'`.

## WebGPU runtime

Build with `--features wgpu`. Select the adapter with `device: wgpu` or `device: wgpu:N`. WebGPU computes in F32 only. On this runtime, the loader decodes F16 and BF16 weights to F32 at load.

### Tested scope

| Model        | Format      | Check                                              | Result        |
| ------------ | ----------- | -------------------------------------------------- | ------------- |
| Llama 3.2 1B | SafeTensors | 32 greedy tokens, wgpu runtime against CPU runtime | Match exactly |
| Llama 3.2 1B | GGUF Q8_0   | 32 greedy tokens, wgpu runtime against CPU runtime | Match exactly |
| Llama 3.2 1B | GGUF Q4_K_M | 32 greedy tokens, wgpu runtime against CPU runtime | Match exactly |

- The host was an NVIDIA RTX 3060 through Vulkan.
- Per-layer hidden states stayed within 1e-5 relative of the CPU F32 reference. The test tolerance is 1e-3 relative.
- The test is `tests/wgpu_generate.rs`. It reads `BLAZR_TEST_LLAMA_MODEL` for SafeTensors, with `$BOOSTR_MODELS_DIR/llama-3.2-1b` as the fallback, and `BLAZR_TEST_WGPU_GGUF` for GGUF. Each test skips when its fixture is absent or no adapter opens.

### Not yet tested

- Other model families on wgpu.
- wgpu throughput.
- Metal, DX12, and the browser.

## Sources

- blazr: `src/loader/detect.rs`, `src/loader/formats/safetensors/`, `src/loader/formats/gguf_config.rs`, `src/loader/speech_load/tts/`, `src/loader/speech_load/whisper_loader.rs`, `src/model/chat_template/`, `src/tokenizer/`, `src/server/text_api/rerank.rs`, `src/server/text_api/embeddings.rs`.
- boostr: `src/format/hf_restructure.rs`, `src/model/config/block.rs`, `src/model/detection/`, `src/model/registry/load.rs`, `src/model/config/huggingface/model_type.rs`, `src/quant/format.rs`, `src/quant/cpu/dequant.rs`, `src/quant/cuda/dequant.rs`, `src/quant/cuda/kernels/dequant_generic.cu`.
- Tests: `tests/` in blazr. Gated tests name their checkpoints in the constants at the top of each file.
- Production checklist: `blazr_production_checklist.md`, sections 1 and 2.

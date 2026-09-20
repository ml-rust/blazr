//! Interactive generation command

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use boostr::CudaRuntime;

use super::run_repl::{interactive_repl, print_generation_stream};
use super::RuntimeArgs;
use crate::config::GenerationConfig;
use crate::engine::vision_tower::{VisionTower, IMAGE_MARKER_TEXT};
use crate::engine::Executor;
#[cfg(feature = "cuda")]
use crate::loader::OffloadingOptions;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::model::chat_template::ChatTemplate;
use crate::tokenizer::from_vocab_size;
use boostr::{CpuRuntime, DType, Runtime};

/// Run interactive generation. `images` are files placed before `prompt`,
/// one marker group each; they need `--mmproj`.
pub async fn run(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RuntimeArgs,
    images: Vec<PathBuf>,
) -> Result<()> {
    let images = read_images(&images)?;
    if runtime.cpu {
        run_cpu(model, prompt, chat_template, gen_config, runtime, images).await
    } else {
        #[cfg(feature = "cuda")]
        {
            run_cuda(model, prompt, chat_template, gen_config, runtime, images).await
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!(
                "CUDA support is not enabled. Rebuild with --features cuda, or use --cpu flag."
            )
        }
    }
}

#[cfg(feature = "cuda")]
async fn run_cuda(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RuntimeArgs,
    images: Vec<Vec<u8>>,
) -> Result<()> {
    let RuntimeArgs {
        gpu_layers,
        num_ctx,
        paged_attention,
        graphs,
        vision,
        ..
    } = runtime;
    // Initialize device
    let device = boostr::CudaDevice::new(0);

    tracing::info!("Loading model: {}", model);

    // Detect format to decide tokenizer strategy
    let source = detect_model_source(Path::new(&model))?;

    // Load model with appropriate tokenizer based on format
    let (loaded_model, mut config, tokenizer) = match source.format {
        ModelFormat::Gguf => {
            // GGUF: use embedded tokenizer (exact vocabulary from the file)
            let (m, c, tok) = loader::load_gguf_with_tokenizer::<CudaRuntime, _>(&model, &device)?;
            tracing::info!("Using GGUF-embedded tokenizer");
            (m, c, tok)
        }
        ModelFormat::SafeTensors => {
            if gpu_layers < 0 {
                let (m, c) = loader::load_model::<CudaRuntime, _>(&model, &device)?;
                let tok = from_vocab_size(c.vocab_size())?;
                (m, c, tok)
            } else {
                let options = if gpu_layers > 0 {
                    OffloadingOptions::default().gpu_layers(gpu_layers as usize)
                } else {
                    OffloadingOptions::default()
                };

                let (m, c, info) =
                    loader::load_model_with_offloading::<CudaRuntime, _>(&model, &device, options)?;

                tracing::info!(
                    "Model loaded: {} layers on GPU, {} on CPU ({:.2} GB on GPU)",
                    info.gpu_layers,
                    info.cpu_layers,
                    info.gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );

                let tok = from_vocab_size(c.vocab_size())?;
                (m, c, tok)
            }
        }
    };

    // Enable paged attention if requested
    if paged_attention {
        config.inference.paged_attention = true;
        tracing::info!(
            "Paged attention enabled (block_size={})",
            config.inference.block_size
        );
    }
    if graphs {
        config.inference.graphs = true;
        tracing::info!("CUDA graph mode enabled");
    }
    vision.into_overrides().apply(&mut config.inference);

    // Pre-load all CUDA PTX modules to avoid JIT compilation on first use
    {
        let client = boostr::CudaClient::new(device.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA client for preload: {}", e))?;
        boostr::preload_inference_modules(&client)
            .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {}", e))?;
        tracing::debug!("Pre-loaded all CUDA PTX modules");
    }

    // Create executor with num_ctx for KV cache initial capacity
    let template = ChatTemplate::detect(
        Path::new(&model),
        config.model_type(),
        chat_template.as_deref(),
    )?;
    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?
        .with_chat_template(template)
        .with_configured_vision_tower()?;

    // Warm up kernels to avoid first-run latency
    executor.warmup()?;
    tracing::info!("Model ready");

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        let prompt_tokens = image_prompt_ids(
            executor.tokenizer(),
            executor.vision_tower().map(|t| &**t),
            &prompt,
            images.len(),
        )?;
        // Graph replay reads only the argmax token, so it decodes greedily
        // and applies no penalties. Any other sampling config runs eagerly.
        let graph_eligible = gen_config.is_greedy() && !gen_config.has_penalties();
        if graphs && !graph_eligible {
            tracing::warn!(
                "CUDA graph mode needs temperature 0 and no repeat/frequency/presence \
                 penalty (got temperature={}, repeat_penalty={}); decoding eagerly",
                gen_config.temperature,
                gen_config.repeat_penalty
            );
        }
        if graphs && graph_eligible {
            if paged_attention && !images.is_empty() {
                return Err(anyhow!(
                    "--image needs graph mode without --paged-attention"
                ));
            }
            if paged_attention {
                print_generation_stream(
                    executor.generate_with_graphs_paged(&prompt_tokens, &gen_config),
                    "CUDA graph mode (paged)",
                )
                .await?;
            } else {
                print_generation_stream(
                    executor.generate_with_graphs_images(&prompt_tokens, &images, &gen_config),
                    "CUDA graph mode",
                )
                .await?;
            }
        } else {
            print_generation_stream(
                executor.generate_with_images(&prompt_tokens, &images, &gen_config),
                "",
            )
            .await?;
        }
        return Ok(());
    }
    if !images.is_empty() {
        return Err(anyhow!(
            "--image needs --prompt; the interactive session takes text only"
        ));
    }

    // Interactive REPL (graph mode uses standard generate in REPL for simplicity)
    interactive_repl(&executor, &gen_config, &model, graphs).await
}

async fn run_cpu(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RuntimeArgs,
    images: Vec<Vec<u8>>,
) -> Result<()> {
    let RuntimeArgs {
        num_ctx, vision, ..
    } = runtime;

    // Initialize CPU device
    let device = boostr::CpuDevice::new();

    tracing::info!("Loading model: {}", model);

    // Detect format and load appropriately
    let source = detect_model_source(Path::new(&model))?;

    let (loaded_model, mut config, tokenizer) = match source.format {
        ModelFormat::Gguf => {
            // Load GGUF with embedded tokenizer
            let loaded = loader::load_gguf_with_tokenizer::<CpuRuntime, _>(&model, &device)?;
            tracing::info!("Using GGUF-embedded tokenizer");
            loaded
        }
        ModelFormat::SafeTensors => {
            // Load SafeTensors with splintr tokenizer
            let (loaded_model, config) = loader::load_model::<CpuRuntime, _>(&model, &device)?;
            let tokenizer = from_vocab_size(config.vocab_size())?;
            (loaded_model, config, tokenizer)
        }
    };
    vision.into_overrides().apply(&mut config.inference);
    let template = ChatTemplate::detect(
        Path::new(&model),
        config.model_type(),
        chat_template.as_deref(),
    )?;
    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?
        .with_chat_template(template)
        .with_configured_vision_tower()?;

    // Warm up model (less important for CPU, but good for consistency)
    executor.warmup()?;
    tracing::info!("Model ready");

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        let prompt_tokens = image_prompt_ids(
            executor.tokenizer(),
            executor.vision_tower().map(|t| &**t),
            &prompt,
            images.len(),
        )?;
        print_generation_stream(
            executor.generate_with_images(&prompt_tokens, &images, &gen_config),
            "",
        )
        .await?;
        return Ok(());
    }
    if !images.is_empty() {
        return Err(anyhow!(
            "--image needs --prompt; the interactive session takes text only"
        ));
    }

    // Interactive REPL
    interactive_repl(&executor, &gen_config, &model, false).await
}

/// Read every `--image` file.
fn read_images(paths: &[PathBuf]) -> Result<Vec<Vec<u8>>> {
    paths
        .iter()
        .map(|path| {
            std::fs::read(path).map_err(|e| anyhow!("image {} did not read: {e}", path.display()))
        })
        .collect()
}

/// Encode `prompt` behind `n_images` marker groups. With no images this
/// is the plain prompt encode.
fn image_prompt_ids<R>(
    tokenizer: &splintr::AnyTokenizer,
    tower: Option<&VisionTower<R>>,
    prompt: &str,
    n_images: usize,
) -> Result<Vec<u32>>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::model::ModelClient<R>,
{
    if n_images == 0 {
        return Ok(tokenizer.encode(prompt));
    }
    let tower = tower.ok_or_else(|| anyhow!("--image needs --mmproj"))?;
    let text = format!("{}{prompt}", IMAGE_MARKER_TEXT.repeat(n_images));
    let ids = tokenizer.encode(&text);
    tower
        .check_marker_counts(&ids, n_images)
        .map_err(|message| anyhow!("{message}"))?;
    Ok(ids)
}

//! Interactive generation command

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use boostr::CudaRuntime;

use super::run_repl::{interactive_repl, print_generation_stream};
use super::RunSettings;
use crate::config::{DeviceChoice, GenerationConfig};
use crate::engine::vision_tower::{VisionTower, IMAGE_MARKER_TEXT};
use crate::engine::Executor;
#[cfg(feature = "cuda")]
use crate::loader::OffloadingOptions;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::model::chat_template::ChatTemplate;
use crate::tokenizer::checkpoint_tokenizer;
use boostr::{CpuRuntime, DType, Runtime};

/// Run interactive generation. `images` are files placed before `prompt`,
/// one marker group each; they need `--mmproj`.
pub async fn run(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RunSettings,
    images: Vec<PathBuf>,
) -> Result<()> {
    let images = read_images(&images)?;
    match runtime.device {
        DeviceChoice::Cpu => {
            let device = boostr::CpuDevice::new();
            run_plain::<CpuRuntime>(
                model,
                prompt,
                chat_template,
                gen_config,
                runtime,
                images,
                device,
            )
            .await
        }
        #[cfg(feature = "wgpu")]
        DeviceChoice::Wgpu(index) => {
            let device = crate::runtime::open_wgpu_device(index)?;
            run_plain::<boostr::runtime::wgpu::WgpuRuntime>(
                model,
                prompt,
                chat_template,
                gen_config,
                runtime,
                images,
                device,
            )
            .await
        }
        #[cfg(not(feature = "wgpu"))]
        DeviceChoice::Wgpu(_) => anyhow::bail!(
            "wgpu support is not enabled. Rebuild with --features wgpu, or pass --cpu, or set \
             device: cpu in ~/.blazr/config.yaml"
        ),
        #[cfg(feature = "cuda")]
        DeviceChoice::Cuda(index) => {
            run_cuda(
                model,
                prompt,
                chat_template,
                gen_config,
                runtime,
                index,
                images,
            )
            .await
        }
        #[cfg(not(feature = "cuda"))]
        DeviceChoice::Cuda(_) => anyhow::bail!(
            "CUDA support is not enabled. Rebuild with --features cuda, or pass --cpu, or set \
             device: cpu in ~/.blazr/config.yaml"
        ),
    }
}

#[cfg(feature = "cuda")]
async fn run_cuda(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RunSettings,
    cuda_index: usize,
    images: Vec<Vec<u8>>,
) -> Result<()> {
    let gpu_layers = runtime.gpu_layers;
    let num_ctx = runtime.load.num_ctx;
    // Initialize device
    let device = boostr::CudaDevice::new(cuda_index);

    tracing::info!("Loading model: {}", model);

    // Detect format to decide tokenizer strategy
    let source = detect_model_source(Path::new(&model))?;

    // Load model with appropriate tokenizer based on format
    let (loaded_model, mut config, tokenizer) = match source.format {
        ModelFormat::Gguf => {
            // GGUF: use embedded tokenizer (exact vocabulary from the file)
            let (m, c, tok) = loader::load_gguf_with_tokenizer_as::<CudaRuntime, _>(
                &model,
                &device,
                runtime.load.dtype.as_ref(),
            )?;
            tracing::info!("Using GGUF-embedded tokenizer");
            (m, c, tok)
        }
        ModelFormat::SafeTensors => {
            if gpu_layers < 0 {
                let (m, c) = loader::load_model_as::<CudaRuntime, _>(
                    &model,
                    &device,
                    runtime.load.dtype.as_ref(),
                )?;
                let tok = checkpoint_tokenizer(Path::new(&model), c.vocab_size())?;
                (m, c, tok)
            } else {
                let options = if gpu_layers > 0 {
                    OffloadingOptions::default().gpu_layers(gpu_layers as usize)
                } else {
                    OffloadingOptions::default()
                };

                let (m, c, info) = loader::load_model_with_offloading::<CudaRuntime, _>(
                    &model,
                    &device,
                    options,
                    runtime.load.dtype.as_ref(),
                )?;

                tracing::info!(
                    "Model loaded: {} layers on GPU, {} on CPU ({:.2} GB on GPU)",
                    info.gpu_layers,
                    info.cpu_layers,
                    info.gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );

                let tok = checkpoint_tokenizer(Path::new(&model), c.vocab_size())?;
                (m, c, tok)
            }
        }
    };

    // The user file's inference keys, then the flags.
    runtime.apply(&mut config.inference)?;
    let paged_attention = config.inference.paged_attention;
    let graphs = config.inference.graphs;
    if paged_attention {
        tracing::info!(
            "Paged attention enabled (block_size={})",
            config.inference.block_size
        );
    }
    if graphs {
        tracing::info!("CUDA graph mode enabled");
    }

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

/// Generation on a runtime with no paged or graph decode: the CPU or wgpu.
async fn run_plain<R>(
    model: String,
    prompt: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    runtime: RunSettings,
    images: Vec<Vec<u8>>,
    device: R::Device,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::ops::TensorOps<R>
        + boostr::ScalarOps<R>
        + boostr::ConvOps<R>
        + boostr::NormalizationOps<R>
        + boostr::UnaryOps<R>
        + boostr::ActivationOps<R>
        + boostr::BinaryOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + boostr::model::ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    let num_ctx = runtime.load.num_ctx;
    if runtime.paged_attention || runtime.graphs {
        tracing::warn!(
            "--paged-attention and --graphs apply on CUDA only. The {} run ignores them",
            R::name()
        );
    }

    tracing::info!("Loading model: {} on {}", model, R::name());

    // Detect format and load appropriately
    let source = detect_model_source(Path::new(&model))?;

    let (loaded_model, mut config, tokenizer) = match source.format {
        ModelFormat::Gguf => {
            // Load GGUF with embedded tokenizer
            let loaded = loader::load_gguf_with_tokenizer_as::<R, _>(
                &model,
                &device,
                runtime.load.dtype.as_ref(),
            )?;
            tracing::info!("Using GGUF-embedded tokenizer");
            loaded
        }
        ModelFormat::SafeTensors => {
            // Load SafeTensors with splintr tokenizer
            let (loaded_model, config) =
                loader::load_model_as::<R, _>(&model, &device, runtime.load.dtype.as_ref())?;
            let tokenizer = checkpoint_tokenizer(Path::new(&model), config.vocab_size())?;
            (loaded_model, config, tokenizer)
        }
    };
    // This path has no paged or graph decode, so only the load keys apply.
    runtime.load.apply(&mut config.inference)?;
    let template = ChatTemplate::detect(
        Path::new(&model),
        config.model_type(),
        chat_template.as_deref(),
    )?;
    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?
        .with_chat_template(template)
        .with_configured_vision_tower()?;

    // Warm up model: compiles GPU pipelines before the first token.
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

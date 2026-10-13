//! Interactive generation command

use std::path::PathBuf;

use anyhow::{anyhow, Result};

use super::run_load;
use super::run_repl::{interactive_repl, print_generation_stream};
use crate::cli::RunSettings;
use crate::config::{DeviceChoice, GenerationConfig};
use crate::engine::load::blocking::run_blocking;
use crate::engine::lora::run_adapter;
use crate::engine::vision_tower::{VisionTower, IMAGE_MARKER_TEXT};
use crate::engine::Executor;
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
    // Check `--lora` before the model loads.
    run_adapter(&runtime.lora)?;
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
    let (executor, paged_attention, graphs) = {
        let model = model.clone();
        run_blocking(move || {
            run_load::load_cuda(&model, cuda_index, &runtime, chat_template.as_deref())
        })
    }
    .await
    .map_err(|e| anyhow!("model {model}: the load stopped: {e}"))??;

    // Graph replay reads only the argmax token, so it decodes greedily and
    // applies no penalties. Any other sampling config runs eagerly. The
    // interactive session decodes eagerly.
    if let (Some(prompt), true) = (&prompt, graphs) {
        let graph_eligible = gen_config.is_greedy() && !gen_config.has_penalties();
        if !graph_eligible {
            tracing::warn!(
                "CUDA graph mode needs temperature 0 and no repeat/frequency/presence \
                 penalty (got temperature={}, repeat_penalty={}); decoding eagerly",
                gen_config.temperature,
                gen_config.repeat_penalty
            );
        } else {
            let prompt_tokens = image_prompt_ids(
                executor.tokenizer(),
                executor.vision_tower().map(|t| &**t),
                prompt,
                images.len(),
            )?;
            if paged_attention && !images.is_empty() {
                return Err(anyhow!(
                    "--image needs graph mode without --paged-attention"
                ));
            }
            return if paged_attention {
                print_generation_stream(
                    executor.generate_with_graphs_paged(&prompt_tokens, &gen_config),
                    "CUDA graph mode (paged)",
                )
                .await
            } else {
                print_generation_stream(
                    executor.generate_with_graphs_images(&prompt_tokens, &images, &gen_config),
                    "CUDA graph mode",
                )
                .await
            };
        }
    }
    prompt_or_session(&executor, &model, prompt, &images, &gen_config, graphs).await
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
    if runtime.paged_attention || runtime.graphs {
        tracing::warn!(
            "--paged-attention and --graphs apply on CUDA only. The {} run ignores them",
            R::name()
        );
    }

    let executor = {
        let model = model.clone();
        run_blocking(move || {
            run_load::load_plain::<R>(&model, device, &runtime, chat_template.as_deref())
        })
    }
    .await
    .map_err(|e| anyhow!("model {model}: the load stopped: {e}"))??;
    prompt_or_session(&executor, &model, prompt, &images, &gen_config, false).await
}

/// Generate once from `prompt` behind `images`, eagerly. With no prompt,
/// open the interactive session, which takes text only. `graphs` tells
/// the session that graph mode is on.
async fn prompt_or_session<R>(
    executor: &Executor<R>,
    model: &str,
    prompt: Option<String>,
    images: &[Vec<u8>],
    gen_config: &GenerationConfig,
    graphs: bool,
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
    let Some(prompt) = prompt else {
        if !images.is_empty() {
            return Err(anyhow!(
                "--image needs --prompt; the interactive session takes text only"
            ));
        }
        return interactive_repl(executor, gen_config, model, graphs).await;
    };
    let prompt_tokens = image_prompt_ids(
        executor.tokenizer(),
        executor.vision_tower().map(|t| &**t),
        &prompt,
        images.len(),
    )?;
    print_generation_stream(
        executor.generate_with_images(&prompt_tokens, images, gen_config),
        "",
    )
    .await
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

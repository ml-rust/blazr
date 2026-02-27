//! Interactive generation command

use std::io::{self, Write};

use anyhow::Result;
use futures::StreamExt;

#[cfg(feature = "cuda")]
use boostr::CudaRuntime;

use crate::config::GenerationConfig;
use crate::engine::Executor;
#[cfg(feature = "cuda")]
use crate::loader::OffloadingOptions;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::tokenizer::Tokenizer;
use boostr::CpuRuntime;

/// Run interactive generation
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model: String,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    _gpu_layers: i32,
    use_cpu: bool,
    num_ctx: usize,
) -> Result<()> {
    if use_cpu {
        run_cpu(model, prompt, max_tokens, temperature, top_p, num_ctx).await
    } else {
        #[cfg(feature = "cuda")]
        {
            run_cuda(
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                _gpu_layers,
                num_ctx,
            )
            .await
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
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    gpu_layers: i32,
    num_ctx: usize,
) -> Result<()> {
    // Initialize device
    let device = boostr::CudaDevice::new();

    tracing::info!("Loading model: {}", model);

    // Load model with optional GPU layer limiting
    let (loaded_model, config) = if gpu_layers < 0 {
        // -1 means load all layers to GPU (may OOM)
        loader::load_model::<CudaRuntime, _>(&model, &device)?
    } else {
        // 0 means auto-detect, >0 means specific layer count
        let options = if gpu_layers > 0 {
            OffloadingOptions::default().gpu_layers(gpu_layers as usize)
        } else {
            OffloadingOptions::default()
        };

        let (model, config, info) =
            loader::load_model_with_offloading::<CudaRuntime, _>(&model, &device, options)?;

        tracing::info!(
            "Model loaded: {} layers on GPU, {} on CPU ({:.2} GB on GPU)",
            info.gpu_layers,
            info.cpu_layers,
            info.gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        (model, config)
    };

    // Create tokenizer
    let tokenizer = Tokenizer::from_vocab_size(config.vocab_size())?;

    // Create executor with num_ctx for KV cache initial capacity
    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?;

    // Warm up kernels to avoid first-run latency (~90ms savings on first generation)
    executor.warmup()?;
    tracing::info!("Model ready");

    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        ..Default::default()
    };

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        generate_response(&executor, &prompt, &gen_config).await?;
        return Ok(());
    }

    // Interactive loop
    tracing::info!("Starting interactive session.");
    println!("Model: {}", model);
    println!("Type your prompt and press Enter. Type 'exit' or Ctrl+C to quit.\n");
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            break;
        }

        generate_response(&executor, input, &gen_config).await?;
        println!();
    }

    Ok(())
}

async fn run_cpu(
    model: String,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    num_ctx: usize,
) -> Result<()> {
    use std::path::Path;

    // Initialize CPU device
    let device = boostr::CpuDevice::new();

    tracing::info!("Loading model: {}", model);

    // Detect format and load appropriately
    let source = detect_model_source(Path::new(&model))?;

    let executor = match source.format {
        ModelFormat::Gguf => {
            // Load GGUF with embedded tokenizer
            let (loaded_model, config, tokenizer) =
                loader::load_gguf_with_tokenizer::<CpuRuntime, _>(&model, &device)?;
            tracing::info!("Using GGUF-embedded tokenizer (SentencePiece)");
            Executor::new(loaded_model, config, tokenizer, device, num_ctx)?
        }
        ModelFormat::SafeTensors => {
            // Load SafeTensors with splintr tokenizer
            let (loaded_model, config) = loader::load_model::<CpuRuntime, _>(&model, &device)?;
            let tokenizer = Tokenizer::from_vocab_size(config.vocab_size())?;
            Executor::new(loaded_model, config, tokenizer, device, num_ctx)?
        }
    };

    // Warm up model (less important for CPU, but good for consistency)
    executor.warmup()?;
    tracing::info!("Model ready");

    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        ..Default::default()
    };

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        generate_response_cpu(&executor, &prompt, &gen_config).await?;
        return Ok(());
    }

    // Interactive loop
    tracing::info!("Starting interactive session.");
    println!("Model: {}", model);
    println!("Type your prompt and press Enter. Type 'exit' or Ctrl+C to quit.\n");
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            break;
        }

        generate_response_cpu(&executor, input, &gen_config).await?;
        println!();
    }

    Ok(())
}

#[cfg(feature = "cuda")]
async fn generate_response(
    executor: &Executor<CudaRuntime>,
    prompt: &str,
    gen_config: &GenerationConfig,
) -> Result<()> {
    println!();

    let start = std::time::Instant::now();
    let mut token_count = 0usize;

    let stream = executor.generate(prompt, gen_config);
    let mut stream = std::pin::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token.text);
                io::stdout().flush()?;
                token_count += 1;
            }
            Err(e) => {
                eprintln!("\nError during generation: {}", e);
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    let tok_per_sec = token_count as f64 / elapsed.as_secs_f64();

    println!();
    tracing::info!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        token_count,
        elapsed.as_secs_f64(),
        tok_per_sec
    );
    Ok(())
}

async fn generate_response_cpu(
    executor: &Executor<CpuRuntime>,
    prompt: &str,
    gen_config: &GenerationConfig,
) -> Result<()> {
    println!();

    let start = std::time::Instant::now();
    let mut token_count = 0usize;

    let stream = executor.generate(prompt, gen_config);
    let mut stream = std::pin::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token.text);
                io::stdout().flush()?;
                token_count += 1;
            }
            Err(e) => {
                eprintln!("\nError during generation: {}", e);
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    let tok_per_sec = token_count as f64 / elapsed.as_secs_f64();

    println!();
    tracing::info!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        token_count,
        elapsed.as_secs_f64(),
        tok_per_sec
    );
    Ok(())
}

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
    gen_config: GenerationConfig,
    _gpu_layers: i32,
    use_cpu: bool,
    num_ctx: usize,
    paged_attention: bool,
    graphs: bool,
) -> Result<()> {
    if use_cpu {
        run_cpu(model, prompt, gen_config, num_ctx).await
    } else {
        #[cfg(feature = "cuda")]
        {
            run_cuda(
                model,
                prompt,
                gen_config,
                _gpu_layers,
                num_ctx,
                paged_attention,
                graphs,
            )
            .await
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (graphs, paged_attention);
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
    gen_config: GenerationConfig,
    gpu_layers: i32,
    num_ctx: usize,
    paged_attention: bool,
    graphs: bool,
) -> Result<()> {
    // Initialize device
    let device = boostr::CudaDevice::new(0);

    tracing::info!("Loading model: {}", model);

    // Detect format to decide tokenizer strategy
    let source = detect_model_source(std::path::Path::new(&model))?;

    // Load model with appropriate tokenizer based on format
    let (loaded_model, mut config, tokenizer): (_, _, BoxedTokenizer) = match source.format {
        ModelFormat::Gguf => {
            // GGUF: use embedded tokenizer (exact vocabulary from the file)
            let (m, c, tok) = loader::load_gguf_with_tokenizer::<CudaRuntime, _>(&model, &device)?;
            tracing::info!("Using GGUF-embedded tokenizer (SentencePiece)");
            (m, c, Box::new(tok))
        }
        ModelFormat::SafeTensors => {
            if gpu_layers < 0 {
                let (m, c) = loader::load_model::<CudaRuntime, _>(&model, &device)?;
                let tok = Tokenizer::from_vocab_size(c.vocab_size())?;
                (m, c, Box::new(tok))
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

                let tok = Tokenizer::from_vocab_size(c.vocab_size())?;
                (m, c, Box::new(tok))
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

    // Pre-load all CUDA PTX modules to avoid JIT compilation on first use
    {
        let client = boostr::CudaClient::new(device.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA client for preload: {}", e))?;
        boostr::preload_inference_modules(&client)
            .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {}", e))?;
        tracing::debug!("Pre-loaded all CUDA PTX modules");
    }

    // Create executor with num_ctx for KV cache initial capacity
    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?;

    // Warm up kernels to avoid first-run latency
    executor.warmup()?;
    tracing::info!("Model ready");

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        if graphs && gen_config.is_greedy() {
            print_generation_stream(
                executor.generate_with_graphs(&prompt, &gen_config),
                "CUDA graph mode",
            )
            .await?;
        } else {
            print_generation_stream(executor.generate(&prompt, &gen_config), "").await?;
        }
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

        if graphs && gen_config.is_greedy() {
            print_generation_stream(
                executor.generate_with_graphs(input, &gen_config),
                "CUDA graph mode",
            )
            .await?;
        } else {
            print_generation_stream(executor.generate(input, &gen_config), "").await?;
        }
        println!();
    }

    Ok(())
}

async fn run_cpu(
    model: String,
    prompt: Option<String>,
    gen_config: GenerationConfig,
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

    // If prompt provided, generate once
    if let Some(prompt) = prompt {
        print_generation_stream(executor.generate(&prompt, &gen_config), "").await?;
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

        print_generation_stream(executor.generate(input, &gen_config), "").await?;
        println!();
    }

    Ok(())
}

/// Print tokens from a generation stream, showing timing stats when done.
async fn print_generation_stream(
    stream: impl futures::Stream<Item = Result<crate::engine::GeneratedToken>>,
    mode: &str,
) -> Result<()> {
    println!();

    let start = std::time::Instant::now();
    let mut token_count = 0usize;
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
    if mode.is_empty() {
        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s)",
            token_count,
            elapsed.as_secs_f64(),
            tok_per_sec
        );
    } else {
        tracing::info!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s) [{}]",
            token_count,
            elapsed.as_secs_f64(),
            tok_per_sec,
            mode
        );
    }
    Ok(())
}

//! Interactive chat command with multi-turn conversation support

use std::io::{self, Write};

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;

use crate::chat_template::{ChatMessage, ChatTemplate};
use crate::config::GenerationConfig;
use crate::engine::Executor;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::tokenizer::{BoxedTokenizer, Tokenizer};

#[cfg(feature = "cuda")]
type ChatRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ChatRuntime = boostr::CpuRuntime;

/// Run interactive multi-turn chat
#[allow(clippy::too_many_arguments)]
pub async fn chat(
    model: String,
    system: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    num_ctx: usize,
    verbose: bool,
) -> Result<()> {
    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        verbose_prompt: verbose,
        ..Default::default()
    };

    #[cfg(feature = "cuda")]
    let executor = load_cuda(&model, num_ctx)?;
    #[cfg(not(feature = "cuda"))]
    let executor = load_cpu(&model, num_ctx)?;

    // Detect chat template
    let model_path = std::path::Path::new(&model);
    let chat_template = ChatTemplate::detect(model_path, executor.config().model_type());

    // Warmup with spinner
    let spinner = super::util::spinner("Warming up model...");
    executor.warmup()?;
    spinner.finish_and_clear();

    eprintln!(
        "{} - Interactive multi-turn conversation",
        "blazr chat".bold().cyan()
    );
    eprintln!("Model: {}", model.bold());
    eprintln!("Template: {:?}", chat_template);
    eprintln!();
    eprintln!(
        "Commands: {} {} {}",
        "/clear".dimmed(),
        "/system <msg>".dimmed(),
        "/exit".dimmed()
    );
    eprintln!();

    let mut history: Vec<ChatMessage> = Vec::new();

    // Add system message if provided
    if let Some(ref sys) = system {
        history.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
        });
    }

    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle slash commands
        if input.starts_with('/') {
            match input {
                "/exit" | "/quit" => break,
                "/clear" => {
                    history.clear();
                    if let Some(ref sys) = system {
                        history.push(ChatMessage {
                            role: "system".to_string(),
                            content: sys.clone(),
                        });
                    }
                    println!("Conversation cleared.");
                    continue;
                }
                s if s.starts_with("/system ") => {
                    let new_system = s[8..].to_string();
                    // Replace or add system message
                    if let Some(msg) = history.iter_mut().find(|m| m.role == "system") {
                        msg.content = new_system.clone();
                    } else {
                        history.insert(
                            0,
                            ChatMessage {
                                role: "system".to_string(),
                                content: new_system.clone(),
                            },
                        );
                    }
                    println!("System prompt updated: {}", new_system);
                    continue;
                }
                _ => {
                    println!("Unknown command. Use /clear, /system <msg>, or /exit");
                    continue;
                }
            }
        }

        // Add user message to history
        history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Format with chat template
        let prompt = chat_template.apply(&history);

        // Stream generation
        let start = std::time::Instant::now();
        let stream = executor.generate(&prompt, &gen_config);
        let mut stream = std::pin::pin!(stream);
        let mut response_text = String::new();
        let mut token_count = 0usize;

        println!();
        while let Some(result) = stream.next().await {
            match result {
                Ok(token) => {
                    print!("{}", token.text);
                    io::stdout().flush()?;
                    response_text.push_str(&token.text);
                    token_count += 1;
                }
                Err(e) => {
                    eprintln!("\nError: {}", e);
                    break;
                }
            }
        }

        let elapsed = start.elapsed();
        let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
            token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!();
        eprintln!(
            "\n{}",
            format!(
                "[{} tokens, {:.1} tok/s, {:.2}s]",
                token_count,
                tok_per_sec,
                elapsed.as_secs_f64()
            )
            .dimmed()
        );

        // Add assistant response to history
        if !response_text.is_empty() {
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
            });
        }
    }

    Ok(())
}

/// Load model from any format, detecting source and dispatching to the correct loader.
macro_rules! load_model_from_source {
    ($model:expr, $device:expr, $num_ctx:expr) => {{
        let source = detect_model_source(std::path::Path::new($model))?;
        let (loaded_model, config, tokenizer): (_, _, BoxedTokenizer) = match source.format {
            ModelFormat::Gguf => {
                let (m, c, tok) =
                    loader::load_gguf_with_tokenizer::<ChatRuntime, _>($model, &$device)?;
                (m, c, Box::new(tok))
            }
            ModelFormat::SafeTensors => {
                let (m, c) = loader::load_model::<ChatRuntime, _>($model, &$device)?;
                let tok = Tokenizer::from_vocab_size(c.vocab_size())?;
                (m, c, Box::new(tok))
            }
        };
        Executor::new(loaded_model, config, tokenizer, $device, $num_ctx)
    }};
}

#[cfg(feature = "cuda")]
fn load_cuda(model: &str, num_ctx: usize) -> Result<Executor<ChatRuntime>> {
    let device = boostr::CudaDevice::new(0);

    // Pre-load CUDA modules
    let client = boostr::CudaClient::new(device.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA client: {}", e))?;
    boostr::preload_inference_modules(&client)
        .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {}", e))?;

    load_model_from_source!(model, device, num_ctx)
}

fn load_cpu(model: &str, num_ctx: usize) -> Result<Executor<ChatRuntime>> {
    #[cfg(feature = "cuda")]
    {
        let _ = (model, num_ctx);
        unreachable!("load_cpu should not be called when cuda feature is enabled");
    }
    #[cfg(not(feature = "cuda"))]
    {
        let device = boostr::CpuDevice::new();
        load_model_from_source!(model, device, num_ctx)
    }
}

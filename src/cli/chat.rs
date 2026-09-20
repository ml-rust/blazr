//! Interactive chat command with multi-turn conversation support

use std::io::{self, Write};

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;

use crate::config::{GenerationConfig, VisionOverrides};
use crate::engine::vision_tower::{IMAGE_MARKER_TEXT, MARKER_NAMES};
use crate::engine::Executor;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::model::chat_template::{
    encode_chat_prompt_with_markers, ChatMessage, ChatTemplate, RenderOptions,
};
use crate::tokenizer::from_vocab_size;

#[cfg(feature = "cuda")]
type ChatRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ChatRuntime = boostr::CpuRuntime;

/// Run interactive multi-turn chat
#[allow(clippy::too_many_arguments)]
pub async fn chat(
    model: String,
    system: Option<String>,
    chat_template: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    num_ctx: usize,
    verbose: bool,
    vision: VisionOverrides,
) -> Result<()> {
    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        verbose_prompt: verbose,
        ..Default::default()
    };

    #[cfg(feature = "cuda")]
    let executor = load_cuda(&model, num_ctx, &vision)?;
    #[cfg(not(feature = "cuda"))]
    let executor = load_cpu(&model, num_ctx, &vision)?;

    // Detect chat template
    let model_path = std::path::Path::new(&model);
    let template = ChatTemplate::detect(
        model_path,
        executor.config().model_type(),
        chat_template.as_deref(),
    )?;
    let executor = executor.with_chat_template(template);

    // Warmup with spinner
    let spinner = super::util::spinner("Warming up model...");
    executor.warmup()?;
    spinner.finish_and_clear();

    eprintln!(
        "{} - Interactive multi-turn conversation",
        "blazr chat".bold().cyan()
    );
    eprintln!("Model: {}", model.bold());
    eprintln!("Template: {}", executor.chat_template().describe());
    eprintln!();
    eprintln!(
        "Commands: {} {} {} {}",
        "/clear".dimmed(),
        "/system <msg>".dimmed(),
        "/image <path>".dimmed(),
        "/exit".dimmed()
    );
    eprintln!("Use arrow keys for history navigation.\n");

    let mut history: Vec<ChatMessage> = Vec::new();
    // Every image of the conversation, in turn order; the prompt holds one
    // marker group per entry. `pending` waits for the next user turn.
    let mut session_images: Vec<Vec<u8>> = Vec::new();
    let mut pending_images: Vec<Vec<u8>> = Vec::new();
    let server_markers: splintr::FxHashSet<String> =
        MARKER_NAMES.iter().map(|s| s.to_string()).collect();
    let no_markers = splintr::FxHashSet::default();

    // Add system message if provided
    if let Some(ref sys) = system {
        history.push(ChatMessage::new("system", sys.clone()));
    }

    // Set up rustyline
    let history_path = dirs::data_dir()
        .map(|d| d.join("blazr").join("chat_history.txt"))
        .unwrap_or_else(|| std::path::PathBuf::from(".blazr_chat_history"));

    if let Some(parent) = history_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut rl = rustyline::DefaultEditor::new()?;
    let _ = rl.load_history(&history_path);

    loop {
        let readline = rl.readline(">>> ");
        let input = match readline {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => break,
            Err(e) => {
                eprintln!("Input error: {}", e);
                break;
            }
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        rl.add_history_entry(&input)?;

        // Handle slash commands
        if input.starts_with('/') {
            match input.as_str() {
                "/exit" | "/quit" => break,
                "/clear" => {
                    history.clear();
                    session_images.clear();
                    pending_images.clear();
                    if let Some(ref sys) = system {
                        history.push(ChatMessage::new("system", sys.clone()));
                    }
                    eprintln!("Conversation cleared.");
                    continue;
                }
                s if s.starts_with("/image ") => {
                    if executor.vision_tower().is_none() {
                        eprintln!("Error: /image needs --mmproj");
                        continue;
                    }
                    let path = s[7..].trim();
                    match std::fs::read(path) {
                        Ok(bytes) => {
                            pending_images.push(bytes);
                            eprintln!("Image attached to the next turn: {path}");
                        }
                        Err(e) => eprintln!("Error: image {path} did not read: {e}"),
                    }
                    continue;
                }
                s if s.starts_with("/system ") => {
                    let new_system = s[8..].to_string();
                    if let Some(msg) = history.iter_mut().find(|m| m.role == "system") {
                        msg.content.clone_from(&new_system);
                    } else {
                        history.insert(0, ChatMessage::new("system", new_system.clone()));
                    }
                    eprintln!("System prompt updated: {}", new_system);
                    continue;
                }
                _ => {
                    eprintln!(
                        "Unknown command. Use /clear, /system <msg>, /image <path>, or /exit"
                    );
                    continue;
                }
            }
        }

        // Add user message to history, its pending images marked in front.
        let turn_images = pending_images.len();
        let content = if turn_images == 0 {
            input
        } else {
            let markers = vec![IMAGE_MARKER_TEXT; turn_images].join("\n");
            format!("{markers}\n{input}")
        };
        history.push(ChatMessage::new("user", content));
        session_images.append(&mut pending_images);

        // Render and encode under the template's own special-token rule,
        // plus the image markers this session wrote.
        let markers = if session_images.is_empty() {
            &no_markers
        } else {
            &server_markers
        };
        let encoded = encode_chat_prompt_with_markers(
            executor.chat_template(),
            executor.tokenizer(),
            &history,
            "",
            &RenderOptions::default(),
            markers,
        )
        .map_err(|e| e.to_string())
        .and_then(|(_prompt, ids)| match executor.vision_tower() {
            Some(tower) if !session_images.is_empty() => tower
                .check_marker_counts(&ids, session_images.len())
                .map(|()| ids),
            _ => Ok(ids),
        });
        let prompt_tokens = match encoded {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("Error: {}", e);
                history.pop();
                session_images.truncate(session_images.len() - turn_images);
                continue;
            }
        };

        // Stream generation
        let start = std::time::Instant::now();
        let stream = executor.generate_with_images(&prompt_tokens, &session_images, &gen_config);
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
            history.push(ChatMessage::new("assistant", response_text));
        }
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}

/// Load model from any format, detecting source and dispatching to the correct loader.
macro_rules! load_model_from_source {
    ($model:expr, $device:expr, $num_ctx:expr, $vision:expr) => {{
        let source = detect_model_source(std::path::Path::new($model))?;
        let (loaded_model, mut config, tokenizer) = match source.format {
            ModelFormat::Gguf => {
                let (m, c, tok) =
                    loader::load_gguf_with_tokenizer::<ChatRuntime, _>($model, &$device)?;
                (m, c, tok)
            }
            ModelFormat::SafeTensors => {
                let (m, c) = loader::load_model::<ChatRuntime, _>($model, &$device)?;
                let tok = from_vocab_size(c.vocab_size())?;
                (m, c, tok)
            }
        };
        $vision.apply(&mut config.inference);
        Executor::new(loaded_model, config, tokenizer, $device, $num_ctx)?
            .with_configured_vision_tower()
    }};
}

#[cfg(feature = "cuda")]
fn load_cuda(
    model: &str,
    num_ctx: usize,
    vision: &VisionOverrides,
) -> Result<Executor<ChatRuntime>> {
    let device = boostr::CudaDevice::new(0);

    // Pre-load CUDA modules
    let client = boostr::CudaClient::new(device.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA client: {}", e))?;
    boostr::preload_inference_modules(&client)
        .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {}", e))?;

    load_model_from_source!(model, device, num_ctx, vision)
}

// Gated to mirror `load_cuda` above: each backend gets its own function
// instead of an `unreachable!()` branch for the other, so no path panics
// on a case the compiler can already rule out.
#[cfg(not(feature = "cuda"))]
fn load_cpu(
    model: &str,
    num_ctx: usize,
    vision: &VisionOverrides,
) -> Result<Executor<ChatRuntime>> {
    let device = boostr::CpuDevice::new();
    load_model_from_source!(model, device, num_ctx, vision)
}

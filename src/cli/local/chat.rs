//! Interactive chat command with multi-turn conversation support

use std::io::{self, Write};
use std::path::Path;

use anyhow::{anyhow, Result};
use colored::Colorize;
use futures::StreamExt;

use crate::cli::LoadSettings;
use crate::config::GenerationConfig;
use crate::engine::load::blocking::run_blocking;
use crate::engine::load::parts::{load_model_parts, start_executor};
use crate::engine::vision_tower::{IMAGE_MARKER_TEXT, MARKER_NAMES};
use crate::engine::Executor;
use crate::model::chat_template::{encode_chat_prompt_with_markers, ChatMessage, RenderOptions};

use crate::runtime::ServerRuntime as ChatRuntime;

/// Run interactive multi-turn chat. `gen_config` is resolved from the
/// flags over the user config file. `gpu_index` comes from `device`.
pub async fn chat(
    model: String,
    system: Option<String>,
    chat_template: Option<String>,
    gen_config: GenerationConfig,
    gpu_index: usize,
    load: LoadSettings,
) -> Result<()> {
    let executor = {
        let model = model.clone();
        run_blocking(move || load_executor(&model, gpu_index, &load, chat_template.as_deref()))
    }
    .await
    .map_err(|e| anyhow!("model {model}: the load stopped: {e}"))??;

    // Warmup with spinner
    let spinner = crate::cli::util::spinner("Warming up model...");
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

/// Load `model` on GPU `gpu_index` of the build's runtime, or the CPU,
/// with the chat template `chat_template` names, else the model's own.
fn load_executor(
    model: &str,
    gpu_index: usize,
    load: &LoadSettings,
    chat_template: Option<&str>,
) -> Result<Executor<ChatRuntime>> {
    let device = crate::runtime::open_device(gpu_index)?;
    #[cfg(feature = "cuda")]
    crate::runtime::preload_cuda_modules(&device)?;

    let path = Path::new(model);
    let (loaded_model, mut config, tokenizer) =
        load_model_parts::<ChatRuntime>(path, &device, load.dtype.as_ref(), None)?;
    load.apply(&mut config.inference)?;
    start_executor(
        path,
        (loaded_model, config, tokenizer),
        device,
        load.num_ctx,
        chat_template,
    )
}

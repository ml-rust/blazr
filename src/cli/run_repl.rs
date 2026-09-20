//! The `run` REPL and the token printer both prompt paths share.

use std::io::{self, Write};

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;

use crate::config::GenerationConfig;
use crate::engine::Executor;
use boostr::{DType, Runtime};

/// Interactive multi-turn REPL with arrow key history and slash commands
pub(super) async fn interactive_repl<R>(
    executor: &Executor<R>,
    gen_config: &GenerationConfig,
    model_name: &str,
    _graphs: bool,
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
        + boostr::quant::traits::DequantOps<R>,
{
    eprintln!("{} - Interactive generation", "blazr run".bold().cyan());
    eprintln!("Model: {}", model_name.bold());
    eprintln!();
    eprintln!(
        "Commands: {} {} {}",
        "/clear".dimmed(),
        "/history".dimmed(),
        "/exit".dimmed(),
    );
    eprintln!("Use arrow keys for history navigation.\n");

    let history_path = dirs::data_dir()
        .map(|d| d.join("blazr").join("run_history.txt"))
        .unwrap_or_else(|| std::path::PathBuf::from(".blazr_history"));

    // Ensure parent directory exists
    if let Some(parent) = history_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut rl = rustyline::DefaultEditor::new()?;
    let _ = rl.load_history(&history_path);

    let mut turn_count = 0u32;

    loop {
        let readline = rl.readline("> ");
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

        // Slash commands
        if input.starts_with('/') {
            match input.as_str() {
                "/exit" | "/quit" => break,
                "/clear" => {
                    turn_count = 0;
                    eprintln!("Session cleared.");
                    continue;
                }
                "/history" => {
                    eprintln!("{} turns in this session", turn_count);
                    continue;
                }
                _ => {
                    eprintln!("Unknown command. Use /clear, /history, or /exit");
                    continue;
                }
            }
        }

        turn_count += 1;
        let prompt_tokens = executor.tokenizer().encode(&input);
        print_generation_stream(executor.generate(&prompt_tokens, gen_config), "").await?;
        println!();
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}

/// Print tokens from a generation stream, showing timing stats when done.
pub(super) async fn print_generation_stream(
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
    let stats = if mode.is_empty() {
        format!(
            "[{} tokens, {:.1} tok/s, {:.2}s]",
            token_count,
            tok_per_sec,
            elapsed.as_secs_f64()
        )
    } else {
        format!(
            "[{} tokens, {:.1} tok/s, {:.2}s, {}]",
            token_count,
            tok_per_sec,
            elapsed.as_secs_f64(),
            mode
        )
    };
    eprintln!("{}", stats.dimmed());
    Ok(())
}

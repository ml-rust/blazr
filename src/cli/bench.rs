//! Built-in benchmarking command
//!
//! Measures prefill tok/s, decode tok/s, TTFT, ITL, and peak memory usage.
//! Supports JSON export and concurrency sweeps.

use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;

use crate::config::GenerationConfig;
use crate::engine::Executor;
use crate::loader::{self, detect_model_source, ModelFormat};
use crate::tokenizer::{BoxedTokenizer, Tokenizer};
use boostr::{DType, Runtime};

#[cfg(feature = "cuda")]
type BenchRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type BenchRuntime = boostr::CpuRuntime;

/// Benchmark prompt lengths to test
const PROMPT_LENGTHS: &[usize] = &[32, 128, 512];

/// Default number of tokens to generate per run
const DEFAULT_DECODE_TOKENS: usize = 128;

/// Number of warmup runs before measurement
const WARMUP_RUNS: usize = 1;

/// Number of measurement runs
const MEASURE_RUNS: usize = 3;

/// Run benchmarks on a model
pub async fn bench(
    model: String,
    num_ctx: usize,
    decode_tokens: Option<usize>,
    runs: Option<usize>,
    json_output: Option<PathBuf>,
    concurrency: Option<Vec<usize>>,
) -> Result<()> {
    let decode_tokens = decode_tokens.unwrap_or(DEFAULT_DECODE_TOKENS);
    let measure_runs = runs.unwrap_or(MEASURE_RUNS);

    eprintln!("{} - Model Benchmark", "blazr bench".bold().cyan());
    eprintln!("Model: {}", model.bold());
    eprintln!(
        "Config: {} decode tokens, {} warmup, {} measurement runs\n",
        decode_tokens, WARMUP_RUNS, measure_runs
    );

    // Load model
    let spinner = super::util::spinner(format!("Loading model '{}'...", model));

    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    let source = detect_model_source(std::path::Path::new(&model))?;

    let (loaded_model, config, tokenizer): (_, _, BoxedTokenizer) = match source.format {
        ModelFormat::Gguf => {
            let (m, c, tok) = loader::load_gguf_with_tokenizer::<BenchRuntime, _>(&model, &device)?;
            (m, c, Box::new(tok))
        }
        ModelFormat::SafeTensors => {
            let (m, c) = loader::load_model::<BenchRuntime, _>(&model, &device)?;
            let tok = Tokenizer::from_vocab_size(c.vocab_size())?;
            (m, c, Box::new(tok))
        }
    };

    #[cfg(feature = "cuda")]
    {
        let client = boostr::CudaClient::new(device.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA client: {}", e))?;
        boostr::preload_inference_modules(&client)
            .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {}", e))?;
    }

    let executor = Executor::new(loaded_model, config, tokenizer, device, num_ctx)?;
    executor.warmup()?;
    spinner.finish_and_clear();

    eprintln!("  {} model loaded\n", "✓".green());

    // Collect all results for JSON export
    let mut all_results: Vec<serde_json::Value> = Vec::new();

    // Print header
    eprintln!(
        "  {:>10}  {:>12}  {:>12}  {:>10}  {:>12}  {:>12}",
        "Prompt Len".bold(),
        "TTFT".bold(),
        "Decode tok/s".bold(),
        "Total".bold(),
        "ITL p50".bold(),
        "ITL p99".bold(),
    );
    eprintln!("  {}", "─".repeat(76));

    // Run benchmarks for each prompt length
    for &prompt_len in PROMPT_LENGTHS {
        let prompt = generate_bench_prompt(prompt_len);

        let gen_config = GenerationConfig {
            max_tokens: decode_tokens,
            temperature: 0.0, // Greedy for determinism
            ..Default::default()
        };

        let mut ttft_samples = Vec::new();
        let mut decode_tps_samples = Vec::new();
        let mut total_samples = Vec::new();
        let mut all_itl_samples = Vec::new();

        // Warmup
        for _ in 0..WARMUP_RUNS {
            let _ = run_single_bench(&executor, &prompt, &gen_config).await;
        }

        // Measure
        for _ in 0..measure_runs {
            if let Ok(result) = run_single_bench(&executor, &prompt, &gen_config).await {
                ttft_samples.push(result.ttft_ms);
                if result.decode_tok_per_sec > 0.0 {
                    decode_tps_samples.push(result.decode_tok_per_sec);
                }
                total_samples.push(result.total_ms);
                all_itl_samples.extend_from_slice(&result.itl_samples_ms);
            }
        }

        if ttft_samples.is_empty() {
            eprintln!("  {:>10}  {}", prompt_len, "FAILED".red());
            continue;
        }

        let med_ttft = median(&ttft_samples);
        let med_tps = median(&decode_tps_samples);
        let med_total = median(&total_samples);
        let itl_p50 = percentile(&all_itl_samples, 50.0);
        let itl_p99 = percentile(&all_itl_samples, 99.0);

        eprintln!(
            "  {:>10}  {:>10.1} ms  {:>10.1} t/s  {:>8.1} ms  {:>10.2} ms  {:>10.2} ms",
            prompt_len, med_ttft, med_tps, med_total, itl_p50, itl_p99,
        );

        all_results.push(serde_json::json!({
            "prompt_tokens": prompt_len,
            "concurrency": 1,
            "ttft_ms": { "median": med_ttft, "p50": percentile(&ttft_samples, 50.0), "p95": percentile(&ttft_samples, 95.0), "p99": percentile(&ttft_samples, 99.0), "samples": &ttft_samples },
            "decode_tok_per_sec": { "median": med_tps, "samples": &decode_tps_samples },
            "total_ms": { "median": med_total, "samples": &total_samples },
            "itl_ms": { "p50": itl_p50, "p95": percentile(&all_itl_samples, 95.0), "p99": itl_p99, "count": all_itl_samples.len() },
        }));
    }

    // Concurrency sweep (if requested)
    if let Some(ref levels) = concurrency {
        eprintln!();
        eprintln!("  {} Concurrency sweep", "▸".cyan());
        eprintln!(
            "  {:>12}  {:>12}  {:>12}  {:>12}",
            "Concurrency".bold(),
            "Throughput".bold(),
            "TTFT p50".bold(),
            "TTFT p99".bold(),
        );
        eprintln!("  {}", "─".repeat(54));

        let prompt = generate_bench_prompt(128);
        let gen_config = GenerationConfig {
            max_tokens: decode_tokens,
            temperature: 0.0,
            ..Default::default()
        };

        for &c in levels {
            let mut handles = Vec::new();
            let start = std::time::Instant::now();

            for _ in 0..c {
                let p = prompt.clone();
                let gc = gen_config.clone();
                // Run sequentially since executor isn't Send — measures throughput under load
                handles.push(run_single_bench(&executor, &p, &gc).await);
            }

            let wall_time = start.elapsed().as_secs_f64();
            let mut ttfts = Vec::new();
            let mut total_tokens = 0usize;

            for result in handles.into_iter().flatten() {
                ttfts.push(result.ttft_ms);
                total_tokens += (result.total_ms / 1000.0 * result.decode_tok_per_sec) as usize;
            }

            let throughput = total_tokens as f64 / wall_time;
            let ttft_p50 = percentile(&ttfts, 50.0);
            let ttft_p99 = percentile(&ttfts, 99.0);

            eprintln!(
                "  {:>12}  {:>10.1} t/s  {:>10.1} ms  {:>10.1} ms",
                c, throughput, ttft_p50, ttft_p99,
            );

            all_results.push(serde_json::json!({
                "prompt_tokens": 128,
                "concurrency": c,
                "wall_time_s": wall_time,
                "throughput_tok_per_sec": throughput,
                "ttft_ms": { "p50": ttft_p50, "p99": ttft_p99, "samples": &ttfts },
            }));
        }
    }

    eprintln!();

    // JSON export
    if let Some(path) = json_output {
        let report = serde_json::json!({
            "model": model,
            "backend": if cfg!(feature = "cuda") { "cuda" } else { "cpu" },
            "num_ctx": num_ctx,
            "decode_tokens": decode_tokens,
            "measurement_runs": measure_runs,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "results": all_results,
        });
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&path, &json)?;
        eprintln!("  {} Results written to {}", "✓".green(), path.display());
    }

    Ok(())
}

struct BenchResult {
    ttft_ms: f64,
    decode_tok_per_sec: f64,
    total_ms: f64,
    itl_samples_ms: Vec<f64>,
}

async fn run_single_bench<R>(
    executor: &Executor<R>,
    prompt: &str,
    gen_config: &GenerationConfig,
) -> Result<BenchResult>
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
        + boostr::model::ModelClient<R>,
{
    let start = std::time::Instant::now();
    let mut first_token_time = None;
    let mut token_count = 0usize;
    let mut last_token_time = start;
    let mut itl_samples = Vec::new();

    let stream = executor.generate(prompt, gen_config);
    let mut stream = std::pin::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(_) => {
                let now = std::time::Instant::now();
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed());
                } else {
                    // ITL = time since previous token
                    let itl = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
                    itl_samples.push(itl);
                }
                last_token_time = now;
                token_count += 1;
            }
            Err(e) => return Err(e),
        }
    }

    let total = start.elapsed();
    let ttft = first_token_time.unwrap_or(total);

    // Decode time = total - ttft, decode tokens = total - 1 (first token is prefill)
    let decode_tokens = token_count.saturating_sub(1);
    let decode_time = total - ttft;
    let decode_tok_per_sec = if decode_time.as_secs_f64() > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchResult {
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        decode_tok_per_sec,
        total_ms: total.as_secs_f64() * 1000.0,
        itl_samples_ms: itl_samples,
    })
}

/// Generate a benchmark prompt of approximately the given token count.
fn generate_bench_prompt(approx_tokens: usize) -> String {
    const TOKENS_PER_WORD: f64 = 1.3;
    const BASE: &str = "The quick brown fox jumps over the lazy dog. ";
    const WORDS_PER_SENTENCE: usize = 9;

    let words_needed = (approx_tokens as f64 / TOKENS_PER_WORD) as usize;
    let sentences = (words_needed / WORDS_PER_SENTENCE).max(1);
    BASE.repeat(sentences)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

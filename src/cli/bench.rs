//! Built-in benchmarking command
//!
//! Measures prefill tok/s, decode tok/s, TTFT, and peak memory usage.

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

    // Print header
    eprintln!(
        "  {:>10}  {:>12}  {:>12}  {:>10}",
        "Prompt Len".bold(),
        "TTFT".bold(),
        "Decode tok/s".bold(),
        "Total".bold(),
    );
    eprintln!("  {}", "─".repeat(50));

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
            }
        }

        if ttft_samples.is_empty() {
            eprintln!("  {:>10}  {}", prompt_len, "FAILED".red());
            continue;
        }

        let avg_ttft = median(&ttft_samples);
        let avg_tps = median(&decode_tps_samples);
        let avg_total = median(&total_samples);

        eprintln!(
            "  {:>10}  {:>10.1} ms  {:>10.1} t/s  {:>8.1} ms",
            prompt_len, avg_ttft, avg_tps, avg_total,
        );
    }

    eprintln!();
    Ok(())
}

struct BenchResult {
    ttft_ms: f64,
    decode_tok_per_sec: f64,
    total_ms: f64,
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
        + boostr::model::ModelClient<R>,
{
    let start = std::time::Instant::now();
    let mut first_token_time = None;
    let mut token_count = 0usize;

    let stream = executor.generate(prompt, gen_config);
    let mut stream = std::pin::pin!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(_) => {
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed());
                }
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
    })
}

/// Generate a benchmark prompt of approximately the given token count.
/// Uses repetitive text that tokenizes predictably.
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

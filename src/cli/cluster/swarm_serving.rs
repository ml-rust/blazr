//! The swarm leader's HTTP serving: every generation route runs through the
//! pipeline stages.

use std::sync::Arc;

use super::swarm_scheduler::SwarmScheduler;
use crate::engine::{Executor, GenerationScheduler};
use crate::runtime::ServerRuntime;
use crate::server::handlers::{ServeReplica, Serving};

/// The serving of a leader whose start model is `executor`, with every
/// generation of it on `swarm`.
///
/// The model scheduler beside it must be pinned to `executor`, so no
/// request loads, unloads or replaces the model the swarm serves.
pub(crate) fn swarm_serving(
    executor: Arc<Executor<ServerRuntime>>,
    swarm: Arc<SwarmScheduler>,
) -> Serving {
    let scheduler: Arc<dyn GenerationScheduler> = swarm;
    Serving::Single(ServeReplica {
        executor,
        scheduler: Some(scheduler),
        device: "swarm".to_string(),
    })
}

/// Output parity with one process, on a seeded CPU model split in three.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::{anyhow, Result};
    use boostr::CpuDevice;
    use serde_json::json;

    use super::super::disaggregated_test_rig::{app, ok, post};
    use super::super::swarm_test_rig::{get, Probe, Swarm, LAYERS, SEED, WAIT};
    use super::*;
    use crate::config::GenerationConfig;
    use crate::engine::batch_test_rig::alone_tokens;
    use crate::engine::test_support::seeded_full_vocab_layers;
    use crate::engine::{GeneratedToken, Scheduler};
    use crate::server::handlers::test_state;

    const MODEL: &str = "m";

    fn executor() -> Result<Executor<ServerRuntime>> {
        seeded_full_vocab_layers(SEED, LAYERS)
    }

    async fn swarm_of(executor: Executor<ServerRuntime>) -> Result<Swarm> {
        Swarm::start(Arc::new(executor), vec![Probe::default(); 3]).await
    }

    async fn tokens_of(
        scheduler: &dyn GenerationScheduler,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<Vec<GeneratedToken>> {
        let mut handle = scheduler.submit(prompt.to_vec(), cfg.clone())?;
        let mut tokens = Vec::new();
        loop {
            let item = tokio::time::timeout(WAIT, handle.token_rx.recv())
                .await?
                .ok_or_else(|| anyhow!("the stream closed before a finish reason"))?;
            let token = item?;
            let done = token.finish_reason.is_some();
            tokens.push(token);
            if done {
                return Ok(tokens);
            }
        }
    }

    fn view(tokens: &[GeneratedToken]) -> Vec<(u32, String, Option<crate::engine::FinishReason>)> {
        tokens
            .iter()
            .map(|t| (t.token_id, t.text.clone(), t.finish_reason))
            .collect()
    }

    fn greedy(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            ..GenerationConfig::greedy()
        }
    }

    fn seeded(seed: u64) -> GenerationConfig {
        GenerationConfig {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
            seed: Some(seed),
            max_tokens: 12,
            ..GenerationConfig::default()
        }
    }

    /// The key check: three stages on three ranks produce the ids, text and
    /// finish reason of one uninterrupted run, greedy, seeded, and with
    /// logprobs and a logit bias the leader's sampler applies.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn swarm_output_equals_the_single_process_output() -> Result<()> {
        let swarm = swarm_of(executor()?).await?;
        let single = executor()?;
        let prompt = single.tokenizer().encode("The quick brown fox");
        let shaped = GenerationConfig {
            logprobs: true,
            top_logprobs: 3,
            logit_bias: [(prompt[0], 2.0)].into_iter().collect(),
            ..greedy(10)
        };
        for cfg in [greedy(12), seeded(5), shaped] {
            let want = alone_tokens(&single, &prompt, &cfg).await?;
            let got = tokens_of(swarm.scheduler.as_ref(), &prompt, &cfg).await?;
            assert!(!want.is_empty());
            assert_eq!(view(&got), view(&want), "temperature {}", cfg.temperature);
            for (g, w) in got.iter().zip(&want) {
                match (g.logprob, w.logprob) {
                    (Some(g), Some(w)) => assert!((g - w).abs() < 1e-3, "logprob {g} vs {w}"),
                    (None, None) => {}
                    other => return Err(anyhow!("logprobs differ: {other:?}")),
                }
                let tops = |t: &GeneratedToken| {
                    t.top_logprobs
                        .as_ref()
                        .map(|v| v.iter().map(|l| l.token_id).collect::<Vec<_>>())
                };
                assert_eq!(tops(g), tops(w));
            }
        }
        swarm.shutdown().await
    }

    /// A prompt longer than the prefill chunk goes as a prefill frame and
    /// decode frames, and decodes the ids one prefill gives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_chunked_prefill_decodes_the_one_process_ids() -> Result<()> {
        let mut chunked = executor()?;
        chunked.config.inference.prefill_chunk_size = 2;
        let swarm = swarm_of(chunked).await?;
        let single = executor()?;
        let prompt = single.tokenizer().encode("Once upon a time there was");
        assert!(prompt.len() > 2);
        let want = alone_tokens(&single, &prompt, &greedy(8)).await?;
        let got = tokens_of(swarm.scheduler.as_ref(), &prompt, &greedy(8)).await?;
        let ids = |t: &[GeneratedToken]| t.iter().map(|t| t.token_id).collect::<Vec<_>>();
        assert_eq!(ids(&got), ids(&want));
        swarm.shutdown().await
    }

    /// Requests in the pipeline at once keep their own caches and tokens.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_do_not_cross() -> Result<()> {
        let swarm = swarm_of(executor()?).await?;
        let single = executor()?;
        let prompts: Vec<Vec<u32>> = ["Hello there", "A long time ago", "Rust is"]
            .iter()
            .map(|t| single.tokenizer().encode(t))
            .collect();
        let cfg = greedy(10);
        let runs = prompts
            .iter()
            .map(|p| tokens_of(swarm.scheduler.as_ref(), p, &cfg));
        let got = futures::future::try_join_all(runs).await?;
        for (prompt, got) in prompts.iter().zip(got) {
            let want = alone_tokens(&single, prompt, &cfg).await?;
            assert_eq!(view(&got), view(&want));
        }
        swarm.shutdown().await
    }

    /// The text, last finish reason and completion tokens of an SSE stream.
    fn read_stream(raw: &str) -> Result<(String, String, serde_json::Value)> {
        let (mut text, mut finish, mut usage) = (String::new(), String::new(), json!(null));
        for data in raw.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if data == "[DONE]" {
                return Ok((text, finish, usage));
            }
            let chunk: serde_json::Value = serde_json::from_str(data)?;
            if let Some(choice) = chunk["choices"].get(0) {
                text.push_str(choice["delta"]["text"].as_str().unwrap_or(""));
                if let Some(f) = choice["finish_reason"].as_str() {
                    finish = f.to_string();
                }
            }
            if !chunk["usage"].is_null() {
                usage = chunk["usage"].clone();
            }
        }
        Err(anyhow!("the stream has no [DONE]: {raw}"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_http_routes_serve_through_the_swarm_as_one_process_serves() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(MODEL))?;
        std::fs::write(dir.path().join(MODEL).join("config.json"), "{}")?;
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), CpuDevice::new());
        let models = models.pinned();
        let served = models
            .register_base(MODEL, dir.path().join(MODEL), executor()?)
            .await;
        let swarm = Swarm::start(Arc::clone(&served), vec![Probe::default(); 3]).await?;
        let serving = swarm_serving(served, Arc::clone(&swarm.scheduler));
        let routed = app(Arc::new(models), serving);

        let plain_state = test_state(dir.path());
        plain_state
            .scheduler
            .load_base(MODEL, |_| executor())
            .await?;
        let plain = crate::server::api_routes().with_state(Arc::new(plain_state));

        let body = json!({
            "model": MODEL,
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0,
        });
        let want = ok(&plain, "/v1/completions", &body).await?;
        let got = ok(&routed, "/v1/completions", &body).await?;
        let choice = |v: &serde_json::Value, f: &str| v["choices"][0][f].clone();
        assert_eq!(choice(&got, "text"), choice(&want, "text"));
        assert_eq!(
            choice(&got, "finish_reason"),
            choice(&want, "finish_reason")
        );
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            assert_eq!(got["usage"][field], want["usage"][field], "{field}");
        }

        let mut stream = body.clone();
        stream["stream"] = json!(true);
        stream["stream_options"] = json!({"include_usage": true});
        let (status, raw) = post(&routed, "/v1/completions", &stream).await?;
        assert_eq!(status, 200, "{raw}");
        let (text, finish, usage) = read_stream(&raw)?;
        assert_eq!(json!(text), choice(&want, "text"));
        assert_eq!(json!(finish), choice(&want, "finish_reason"));
        assert_eq!(
            usage["completion_tokens"],
            want["usage"]["completion_tokens"]
        );

        let chat = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 6,
            "temperature": 0,
        });
        let want = ok(&plain, "/v1/chat/completions", &chat).await?;
        let got = ok(&routed, "/v1/chat/completions", &chat).await?;
        assert_eq!(
            got["choices"][0]["message"]["content"],
            want["choices"][0]["message"]["content"]
        );

        let mut other = body.clone();
        other["model"] = json!("not-served");
        let (status, raw) = post(&routed, "/v1/completions", &other).await?;
        assert_eq!(status, 404, "the pinned leader loads no other model: {raw}");

        let health = get(&routed, "/health").await?;
        assert_eq!(health.0, 200, "{}", health.1);
        drop(routed);
        swarm.shutdown().await
    }
}

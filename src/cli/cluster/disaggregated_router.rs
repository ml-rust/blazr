//! The disaggregated router's HTTP front end: the tokenizer-only executor
//! and the serving that routes every generation through the workers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

use super::disaggregated_scheduler::DisaggScheduler;
use crate::config::DEFAULT_NUM_CTX;
use crate::distributed::disaggregated::DisaggRouter;
use crate::engine::{Executor, GenerationScheduler, Scheduler};
use crate::loader::load_model_meta;
use crate::model::chat_template::ChatTemplate;
use crate::runtime::{ServerDevice, ServerRuntime};
use crate::server::handlers::{ServeReplica, Serving};

/// The executor the router serves the model at `path` with: its config,
/// tokenizer and chat template, and no weights.
///
/// # Errors
///
/// A path that holds no model, or a config, tokenizer or chat template that
/// does not load.
pub(crate) fn router_executor(
    path: &Path,
    device: ServerDevice,
) -> Result<Executor<ServerRuntime>> {
    let (config, tokenizer) = load_model_meta::<ServerRuntime>(path)?;
    let template = ChatTemplate::detect(path, config.model_type(), None)?;
    Ok(
        Executor::tokenizer_only(config, tokenizer, device, DEFAULT_NUM_CTX)?
            .with_chat_template(template),
    )
}

/// The model scheduler and serving of a router that serves `executor` as
/// `name` and runs its generation through `router`.
///
/// `models` becomes the model scheduler, pinned: it serves `name` only and
/// never loads a model, so the router process holds no weights. Every
/// generation route takes the one replica, whose scheduler is a
/// [`DisaggScheduler`].
pub(crate) async fn router_serving(
    router: Arc<DisaggRouter>,
    name: &str,
    path: PathBuf,
    executor: Executor<ServerRuntime>,
    models: Scheduler<ServerRuntime>,
) -> (Arc<Scheduler<ServerRuntime>>, Serving) {
    let scheduler = Arc::new(models.pinned());
    let executor = scheduler.register_base(name, path, executor).await;
    let disagg: Arc<dyn GenerationScheduler> = DisaggScheduler::new(
        router,
        Arc::clone(&executor.tokenizer),
        executor.eos_ids(),
        executor.config().max_seq_len(),
    );
    let replica = ServeReplica {
        executor,
        scheduler: Some(disagg),
        device: "disaggregated".to_string(),
    };
    (scheduler, Serving::Single(replica))
}

/// The real forward closures on a seeded CPU model, against one process.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::anyhow;
    use axum::body::Body;
    use axum::http::Request;
    use boostr::CpuDevice;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::super::disaggregated_test_rig::Cluster;
    use super::*;
    use crate::config::GenerationConfig;
    use crate::engine::batch_test_rig::alone_tokens;
    use crate::engine::test_support::seeded_full_vocab_executor;
    use crate::server::handlers::{test_state, AppState};

    const SEED: u64 = 0x0d15;
    const MODEL: &str = "m";

    /// A router over a cluster of two seeded workers, with the executor
    /// the router serves.
    async fn disagg() -> Result<(Cluster, Executor<ServerRuntime>)> {
        let cluster = Cluster::real(
            seeded_full_vocab_executor(SEED)?,
            seeded_full_vocab_executor(SEED)?,
        )
        .await?;
        let model = seeded_full_vocab_executor(SEED)?;
        let executor = Executor::tokenizer_only(
            model.config().clone(),
            crate::tokenizer::from_pretrained("llama3")?,
            CpuDevice::new(),
            model.config().max_seq_len(),
        )?;
        Ok((cluster, executor))
    }

    async fn tokens_of(
        scheduler: &dyn GenerationScheduler,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<Vec<crate::engine::GeneratedToken>> {
        let mut handle = scheduler.submit(prompt.to_vec(), cfg.clone())?;
        let mut tokens = Vec::new();
        while let Some(item) = handle.token_rx.recv().await {
            let token = item?;
            let done = token.finish_reason.is_some();
            tokens.push(token);
            if done {
                break;
            }
        }
        Ok(tokens)
    }

    /// The key check: prefill on one worker and decode on another produce
    /// the ids, text and finish reason of one uninterrupted run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disaggregated_output_equals_the_single_process_output() -> Result<()> {
        let (cluster, executor) = disagg().await?;
        let single = seeded_full_vocab_executor(SEED)?;
        let disagg = DisaggScheduler::new(
            Arc::clone(&cluster.router),
            Arc::clone(&executor.tokenizer),
            executor.eos_ids(),
            executor.config().max_seq_len(),
        );
        let prompt = single.tokenizer().encode("The quick brown fox");
        let greedy = GenerationConfig {
            max_tokens: 12,
            ..GenerationConfig::greedy()
        };
        let seeded = GenerationConfig {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
            seed: Some(5),
            max_tokens: 12,
            ..GenerationConfig::default()
        };
        for cfg in [greedy, seeded] {
            let want = alone_tokens(&single, &prompt, &cfg).await?;
            let got = tokens_of(disagg.as_ref(), &prompt, &cfg).await?;
            let view = |t: &[crate::engine::GeneratedToken]| {
                t.iter()
                    .map(|t| (t.token_id, t.text.clone(), t.finish_reason))
                    .collect::<Vec<_>>()
            };
            assert!(!want.is_empty());
            assert_eq!(view(&got), view(&want), "temperature {}", cfg.temperature);
        }
        Ok(())
    }

    /// A server over `serving`, as `blazr serve` builds its state.
    fn app(scheduler: Arc<Scheduler<ServerRuntime>>, serving: Serving) -> axum::Router {
        let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let state = AppState::new(scheduler, metrics).with_serving(serving);
        crate::server::api_routes().with_state(Arc::new(state))
    }

    async fn post(app: &axum::Router, path: &str, body: &Value) -> Result<(u16, String)> {
        let request = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))?;
        let response = app.clone().oneshot(request).await?;
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        Ok((status, String::from_utf8(bytes.to_vec())?))
    }

    async fn ok(app: &axum::Router, path: &str, body: &Value) -> Result<Value> {
        let (status, raw) = post(app, path, body).await?;
        if status != 200 {
            return Err(anyhow!("{path} answered {status}: {raw}"));
        }
        Ok(serde_json::from_str(&raw)?)
    }

    /// The text, last finish reason and usage of an SSE completion stream.
    fn read_stream(raw: &str) -> Result<(String, String, Value)> {
        let mut text = String::new();
        let (mut finish, mut usage) = (String::new(), Value::Null);
        let mut done = false;
        for data in raw.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if data == "[DONE]" {
                done = true;
                continue;
            }
            let chunk: Value = serde_json::from_str(data)?;
            if let Some(choice) = chunk["choices"].get(0) {
                let piece = choice["text"].as_str();
                text.push_str(piece.or(choice["delta"]["text"].as_str()).unwrap_or(""));
                if let Some(f) = choice["finish_reason"].as_str() {
                    finish = f.to_string();
                }
            }
            if !chunk["usage"].is_null() {
                usage = chunk["usage"].clone();
            }
        }
        if !done {
            return Err(anyhow!("the stream has no [DONE]: {raw}"));
        }
        Ok((text, finish, usage))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_http_routes_serve_through_the_workers_as_one_process_serves() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(MODEL))?;
        std::fs::write(dir.path().join(MODEL).join("config.json"), "{}")?;
        let (cluster, executor) = disagg().await?;
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), CpuDevice::new());
        let (scheduler, serving) = router_serving(
            Arc::clone(&cluster.router),
            MODEL,
            dir.path().join(MODEL),
            executor,
            models,
        )
        .await;
        let routed = app(scheduler, serving);

        let plain_state = test_state(dir.path());
        plain_state
            .scheduler
            .load_base(MODEL, |_| seeded_full_vocab_executor(SEED))
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
        assert_eq!(got["choices"][0]["text"], want["choices"][0]["text"]);
        assert_eq!(
            got["choices"][0]["finish_reason"],
            want["choices"][0]["finish_reason"]
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
        assert_eq!(json!(text), want["choices"][0]["text"]);
        assert_eq!(json!(finish), want["choices"][0]["finish_reason"]);
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

        let mut biased = body.clone();
        biased["logit_bias"] = json!({"5": 1.0});
        let (status, raw) = post(&routed, "/v1/completions", &biased).await?;
        assert_eq!(status, 400, "{raw}");
        assert!(raw.contains("logit_bias"), "{raw}");

        let mut other = body.clone();
        other["model"] = json!("not-served");
        let (status, raw) = post(&routed, "/v1/completions", &other).await?;
        assert_eq!(status, 404, "the pinned router loads no model: {raw}");

        let embed = json!({"model": MODEL, "input": "hi"});
        let (status, raw) = post(&routed, "/v1/embeddings", &embed).await?;
        assert_eq!(status, 501, "{raw}");
        Ok(())
    }
}

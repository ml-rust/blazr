//! The generation half `/api/chat` and `/api/generate` share: prepare the
//! chat request, then stream NDJSON or answer with one object.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};

use super::ndjson::ndjson_response;
use super::ollama_error::{bad_request, from_openai, ollama_error};
use super::ollama_options::check_num_ctx;
use super::ollama_render::{Clock, ContextFn, Kind, Render};
use super::ollama_stream::{ndjson_items, Splitter, StreamSetup};
use crate::server::chat_core::{prepare_chat, PreparedChat};
use crate::server::chat_types::ChatRequest;
use crate::server::generation::{apply_keep_alive, record_generation_metrics};
use crate::server::handlers::{AppState, DecodeSlot};
use crate::server::reasoning::ReasoningParser;
use crate::server::route_stream::spawn_media_stream;
use crate::server::tool_parser::ToolCallParser;

/// How a converted request replies.
pub(super) struct Reply {
    /// The model name the client sent, echoed in every body.
    pub requested: String,
    pub stream: bool,
    pub kind: Kind,
    /// The request arrived.
    pub start: Instant,
    /// `options.num_ctx`.
    pub num_ctx: Option<usize>,
}

/// Run `request` and answer it as `reply` asks.
pub(super) async fn reply(state: &Arc<AppState>, request: &ChatRequest, reply: Reply) -> Response {
    let prepared = match prepare_chat(state, request).await {
        Ok(prepared) => prepared,
        Err(response) => return from_openai(*response, &reply.requested).await,
    };
    let max_context_len = prepared
        .placement
        .executor
        .config()
        .inference
        .max_context_len;
    if let Err(message) = check_num_ctx(reply.num_ctx, max_context_len) {
        return bad_request(&message);
    }
    let render = Render {
        kind: reply.kind,
        context: context_fn(reply.kind, request, &prepared),
    };
    let clock = Clock {
        start: reply.start,
        load: Duration::from_millis(prepared.load_duration_ms),
        gen_start: Instant::now(),
    };
    if reply.stream {
        stream_reply(state, request, prepared, render, clock, reply.requested)
    } else {
        complete_reply(state, request, prepared, render, clock, &reply.requested).await
    }
}

/// The `context` of a non-raw generate reply: the prompt ids, then the
/// generated text encoded as ordinary text. It approximates the ids the
/// model produced: a re-encode can split the text differently.
fn context_fn(kind: Kind, request: &ChatRequest, prepared: &PreparedChat<'_>) -> Option<ContextFn> {
    if kind != Kind::Generate || request.raw.unwrap_or(false) {
        return None;
    }
    let executor = Arc::clone(&prepared.placement.executor);
    let prompt = prepared.prompt_tokens.clone();
    Some(Box::new(move |generated: &str| {
        let mut ids = prompt.clone();
        match executor
            .tokenizer()
            .encode_with(generated, &splintr::SpecialMode::Ordinary)
        {
            Ok(more) => ids.extend(more),
            Err(e) => tracing::warn!(error = %e, "context: the generated text did not encode"),
        }
        ids
    }))
}

/// The tool-call parser of the tools the model sees, when it sees any.
fn tool_parser(prepared: &PreparedChat<'_>) -> Option<ToolCallParser> {
    let tools = prepared.visible_tools;
    (!tools.is_empty()).then(|| ToolCallParser::new(tools.to_vec()))
}

/// The NDJSON reply of a streamed request. The keep-alive applies once the
/// last token is out.
fn stream_reply(
    state: &Arc<AppState>,
    request: &ChatRequest,
    prepared: PreparedChat<'_>,
    render: Render,
    clock: Clock,
    requested: String,
) -> Response {
    let tools = tool_parser(&prepared);
    let PreparedChat {
        placement,
        route,
        prompt,
        prompt_tokens,
        media,
        gen_config,
        permit,
        reasoning_format,
        ..
    } = prepared;
    let keep_alive_state = Arc::clone(state);
    let keep_alive_model = request.model.clone();
    let keep_alive = request.keep_alive.clone();
    let tokens = spawn_media_stream(
        permit,
        placement,
        route,
        prompt_tokens,
        media,
        gen_config,
        async move {
            apply_keep_alive(&keep_alive_state, &keep_alive_model, &keep_alive).await;
        },
    );
    let setup = StreamSetup {
        model: requested,
        splitter: Splitter::new(
            ReasoningParser::for_output(reasoning_format, &prompt),
            tools,
        ),
        render,
        clock,
    };
    ndjson_response(ndjson_items(setup, Box::pin(tokens)))
}

/// The one-object reply of a request with `stream` false.
async fn complete_reply(
    state: &AppState,
    request: &ChatRequest,
    prepared: PreparedChat<'_>,
    render: Render,
    clock: Clock,
    requested: &str,
) -> Response {
    let _slot = DecodeSlot::enter();
    let result = match prepared.generate_once(&prepared.gen_config).await {
        Ok(result) => result,
        Err(e) => return ollama_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let end = Instant::now();
    let reasoning = ReasoningParser::for_output(prepared.reasoning_format, &prepared.prompt);
    let piece = Splitter::new(reasoning, tool_parser(&prepared)).split(&result.text, true);
    let timings = clock.completed(
        end,
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_eval_duration_ms,
    );
    record_generation_metrics(
        &request.model,
        &request.user,
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_eval_duration_ms,
        end.saturating_duration_since(clock.gen_start),
    );
    apply_keep_alive(state, &request.model, &request.keep_alive).await;
    let body = render.done(
        requested,
        piece,
        result.finish_reason.as_str(),
        timings,
        &result.text,
    );
    (StatusCode::OK, Json(body)).into_response()
}

/// End-to-end runs through the router on a model whose KV pool runs dry.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::Result;
    use serde_json::{json, Value};

    use super::super::e2e_rig::{Rig, REQUESTED};
    use crate::engine::test_support::PagedPool;

    /// A pool of 56 token slots, below the model's 64-token context.
    const POOL: PagedPool = PagedPool {
        block_size: 8,
        blocks: 7,
    };

    /// Asks for more tokens than the pool holds.
    const NUM_PREDICT: usize = 60;

    /// Checks that `lines` is content chunks, then one error line, and that
    /// no line is a done object.
    fn assert_ends_with_error(lines: &[Value], field: impl Fn(&Value) -> &Value) {
        let (error, chunks) = lines.split_last().expect("at least the error line");
        assert!(
            !chunks.is_empty(),
            "the error came before any chunk: {lines:?}"
        );
        for chunk in chunks {
            assert_eq!(chunk["done"], false, "{chunk}");
            assert_eq!(chunk["model"], REQUESTED, "{chunk}");
            assert!(field(chunk).is_string(), "{chunk}");
        }
        let object = error.as_object().expect("the last line is an object");
        assert_eq!(object.len(), 1, "{error}");
        let message = error["error"].as_str().unwrap_or_default();
        assert!(message.contains("KV blocks"), "{error}");
    }

    #[tokio::test]
    async fn a_chat_error_after_the_first_chunk_ends_the_stream_with_an_error_line() -> Result<()> {
        let rig = Rig::with_pool(POOL).await?;
        let body = json!({
            "model": REQUESTED,
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        });
        let lines = rig.lines("/api/chat", &body).await?;
        assert_ends_with_error(&lines, |chunk| &chunk["message"]["content"]);
        Ok(())
    }

    #[tokio::test]
    async fn a_generate_error_after_the_first_chunk_ends_the_stream_with_an_error_line(
    ) -> Result<()> {
        let rig = Rig::with_pool(POOL).await?;
        let body = json!({
            "model": REQUESTED,
            "prompt": "The sky is",
            "raw": true,
            "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        });
        let lines = rig.lines("/api/generate", &body).await?;
        assert_ends_with_error(&lines, |chunk| &chunk["response"]);
        Ok(())
    }
}

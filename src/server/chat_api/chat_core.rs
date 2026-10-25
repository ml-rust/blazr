//! The dialect-neutral half of a chat request: checks, prompt, admission, placement.

use std::sync::Arc;
use std::time::Instant;

use axum::{http::StatusCode, response::Response};

use super::chat::named_template;
use super::chat_types::ChatRequest;
use super::chat_vision::{decode_request_media, image_route, ImageRoute, RequestMedia};
use super::route_stream::generate_for_media;
use super::tool_choice::ToolChoice;
use super::tools::{build_tools_system_prompt, request_msg_to_chat_msg, Tool};
use crate::config::{GenerationConfig, ReasoningFormat};
use crate::engine::GenerationResult;
use crate::model::chat_template::{encode_chat_prompt_with_markers, ChatMessage};
use crate::server::fim_prompt::{build_fim_prompt, FIM_MARKERS};
use crate::server::generation::{
    admit_generation, chat_prompt_error_response, decode_context_prefix, error_response,
    overloaded_response, policy_error_response, HasSamplingFields,
};
use crate::server::handlers::{select_executor, AppState, BudgetPermit, Route};

/// A chat request that passed every check and holds its token budget.
///
/// `visible_tools` borrows the request's `tools`.
pub(in crate::server) struct PreparedChat<'r> {
    /// The executor and scheduler the request runs on.
    pub placement: Route,
    /// The generator the request's media picks.
    pub route: ImageRoute,
    /// The rendered prompt text.
    pub prompt: String,
    /// The encoded prompt.
    pub prompt_tokens: Vec<u32>,
    /// The decoded images and audio of the messages.
    pub media: RequestMedia,
    /// The sampling config, with the admitted constraint.
    pub gen_config: GenerationConfig,
    /// The token budget reservation. Dropping it releases the tokens.
    pub permit: BudgetPermit,
    /// The tools the model sees under `tool_choice`.
    pub visible_tools: &'r [Tool],
    /// How the output splits reasoning from the answer.
    pub reasoning_format: ReasoningFormat,
    /// Time spent selecting, and if needed loading, the executor.
    pub load_duration_ms: u64,
}

impl PreparedChat<'_> {
    /// Generate one result under `gen_config` on this request's media route.
    pub(in crate::server) async fn generate_once(
        &self,
        gen_config: &GenerationConfig,
    ) -> anyhow::Result<GenerationResult> {
        generate_for_media(
            &self.placement,
            self.route,
            &self.prompt_tokens,
            &self.media,
            gen_config,
        )
        .await
    }
}

/// The 400 response of a request field that did not pass a check.
fn bad_request(message: &str) -> Box<Response> {
    Box::new(error_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
    ))
}

/// Check `request`, encode its prompt, admit it and place it.
///
/// # Errors
///
/// The OpenAI error response to send: `400` for a bad field or prompt, the
/// executor lookup's own response, or `503` when the token budget is full.
pub(in crate::server) async fn prepare_chat<'r>(
    state: &Arc<AppState>,
    request: &'r ChatRequest,
) -> Result<PreparedChat<'r>, Box<Response>> {
    if let Err(e) = request.check_fields() {
        return Err(bad_request(&e));
    }

    if request.messages.is_empty() {
        return Err(bad_request("messages array must not be empty"));
    }

    // `tool_choice` decides which tools the model sees and whether it must call one.
    let tool_choice = ToolChoice::parse(request.tool_choice.as_ref(), request.tools.as_deref())
        .map_err(|message| bad_request(&message))?;
    let visible_tools: &[Tool] = tool_choice.visible(request.tools.as_deref().unwrap_or(&[]));

    let load_start = Instant::now();
    let executor = select_executor(state, &request.model, request.lora_adapter.as_deref()).await?;
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    let context_prefix = decode_context_prefix(&executor, &request.context)
        .map_err(|e| bad_request(&format!("Failed to decode context tokens: {}", e)))?;

    // The request can name a built-in format. Otherwise the model's template applies.
    let override_template = request
        .template
        .as_deref()
        .map(named_template)
        .transpose()
        .map_err(|message| bad_request(&message))?;
    let template = override_template
        .as_ref()
        .unwrap_or_else(|| executor.chat_template());
    // A Jinja template takes the tools and the structured turns itself; a
    // built-in format takes a tool description as a system turn.
    let native_tools = template.renders_tools_natively();
    let tools_system_prompt = if native_tools {
        None
    } else {
        build_tools_system_prompt(visible_tools, tool_choice.forces_call())
            .map_err(|message| bad_request(&message))?
    };

    // Decode the media parts, then pick the path that serves them: the
    // vision tower, the model's own encoders, or a 400.
    let media = decode_request_media(&request.messages)
        .await
        .map_err(|message| bad_request(&message))?;
    let multimodal_model = executor.is_multimodal();
    let route = image_route(executor.vision_tower().is_some(), multimodal_model, &media)
        .map_err(|message| bad_request(&message))?;
    if route == ImageRoute::Tower && request.raw.unwrap_or(false) {
        return Err(bad_request(
            "raw mode renders no image markers; send images without `raw`",
        ));
    }

    // The prompt is encoded here, once, under an explicit special-token policy:
    // whatever the server itself inserts is allowed and everything else in the
    // caller's text is refused. The ids double as the token-budget count.
    let (prompt, prompt_tokens) = if request.raw.unwrap_or(false) {
        // Raw mode inserts no delimiters of its own, so nothing in the caller's
        // text can become a control token.
        let raw_prompt = format!(
            "{}{}",
            context_prefix,
            request
                .messages
                .iter()
                .map(|m| m.content.as_ref().map(|c| c.text()).unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n")
        );
        let tokenizer = executor.tokenizer();
        let encoded = match request.fim_suffix.as_deref() {
            // Only the FIM markers this server inserts can be control tokens.
            Some(suffix) => {
                let fim = build_fim_prompt(tokenizer, &raw_prompt, suffix)
                    .map_err(|message| bad_request(&message))?;
                let ids = tokenizer.encode_with(&fim, &splintr::SpecialMode::Allow(&FIM_MARKERS));
                ids.map(|ids| (fim, ids))
            }
            None => tokenizer
                .encode_with(&raw_prompt, &splintr::SpecialMode::Ordinary)
                .map(|ids| (raw_prompt, ids)),
        };
        encoded.map_err(|e| Box::new(policy_error_response(&e)))?
    } else {
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(ref sys) = request.system {
            let has_system = request.messages.iter().any(|m| m.role == "system");
            if !has_system {
                msgs.push(ChatMessage::new("system", sys.clone()));
            }
        }
        // Inject tools system prompt
        if let Some(ref tools_prompt) = tools_system_prompt {
            msgs.push(ChatMessage::new("system", tools_prompt.clone()));
        }
        msgs.extend(
            request
                .messages
                .iter()
                .map(|m| request_msg_to_chat_msg(m, native_tools, route.image_marker())),
        );
        let options = request
            .render_options(native_tools, visible_tools)
            .map_err(|message| bad_request(&message))?;
        encode_chat_prompt_with_markers(
            template,
            executor.tokenizer(),
            &msgs,
            &context_prefix,
            &options,
            &route.server_markers(),
        )
        .map_err(|e| Box::new(chat_prompt_error_response(&e)))?
    };
    // The marker allow-list let the caller's text spell a marker too: each
    // one must appear exactly once per image.
    if let (ImageRoute::Tower, Some(tower)) = (route, executor.vision_tower()) {
        tower
            .check_marker_counts(&prompt_tokens, media.images.len())
            .map_err(|message| bad_request(&message))?;
    }

    let mut gen_config = request
        .sampling_params()
        .into_gen_config(&state.config().generation);
    admit_generation(
        state,
        &mut gen_config,
        request.constraint(&tool_choice),
        executor.tokenizer(),
    )
    .await
    .map_err(|e| bad_request(&e))?;
    let reasoning_format = request
        .reasoning_format
        .unwrap_or_else(|| state.reasoning_format());

    // Token budget admission control
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    let Some(permit) = state.try_admit(estimated_tokens) else {
        return Err(Box::new(overloaded_response()));
    };

    // The executor and scheduler this request runs on. With data
    // parallelism, the router picks a replica here.
    let placement = state.route(executor, &prompt_tokens);
    if let Some(e) = placement.refusal(&gen_config) {
        return Err(bad_request(&e));
    }

    Ok(PreparedChat {
        placement,
        route,
        prompt,
        prompt_tokens,
        media,
        gen_config,
        permit,
        visible_tools,
        reasoning_format,
        load_duration_ms,
    })
}

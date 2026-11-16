//! The routes the server answers, for the startup log. A test checks this
//! list against the router code, so the log cannot go stale.

/// One route: its method, its router path and what it does.
pub struct RouteInfo {
    /// `GET`, `POST` or `DELETE`.
    pub method: &'static str,
    /// The path as the router registers it.
    pub path: &'static str,
    /// What the route does.
    pub what: &'static str,
}

const fn route(method: &'static str, path: &'static str, what: &'static str) -> RouteInfo {
    RouteInfo { method, path, what }
}

/// Routes that answer without an API key.
pub const PUBLIC_ROUTES: &[RouteInfo] = &[
    route("GET", "/health", "Health check, no API key"),
    route("GET", "/metrics", "Prometheus metrics, no API key"),
];

/// Routes behind the API key, in every build.
pub const API_ROUTES: &[RouteInfo] = &[
    route("GET", "/v1/models", "List models"),
    route("GET", "/v1/models/{*model_id}", "Model details"),
    route("POST", "/v1/completions", "Text completion"),
    route("POST", "/v1/chat/completions", "Chat completion"),
    route("POST", "/v1/embeddings", "Embeddings"),
    route("POST", "/v1/responses", "OpenAI Responses"),
    route("POST", "/v1/messages", "Anthropic Messages"),
    route("POST", "/v1/messages/count_tokens", "Anthropic token count"),
    route("POST", "/v1/infill", "Fill-in-the-middle"),
    route("POST", "/v1/systemone", "Decision model"),
    route("POST", "/rerank", "Rerank documents"),
    route("POST", "/v1/rerank", "Rerank documents"),
    route("POST", "/tokenize", "Tokenize text"),
    route("POST", "/detokenize", "Detokenize tokens"),
    route("POST", "/apply-template", "Render a chat template"),
    route("GET", "/api/slots", "List inference slots"),
    route("POST", "/api/slots", "Create an inference slot"),
    route("DELETE", "/api/slots/{slot_id}", "Free an inference slot"),
    route("GET", "/api/tags", "List local models"),
    route("POST", "/api/show", "Model details"),
    route("GET", "/api/ps", "Loaded models"),
    route("DELETE", "/api/delete", "Delete a local model"),
    route("POST", "/api/copy", "Copy a model"),
    route("POST", "/api/pull", "Pull a model from Hugging Face"),
    route("POST", "/api/chat", "Ollama chat"),
    route("POST", "/api/generate", "Ollama generate"),
    route("POST", "/api/embed", "Ollama embeddings"),
    route("POST", "/api/embeddings", "Ollama embeddings, legacy form"),
    route("GET", "/v1/lora", "List LoRA adapters"),
    route("POST", "/v1/lora", "Load a LoRA adapter"),
    route("DELETE", "/v1/lora/{name}", "Unload a LoRA adapter"),
];

/// Routes behind the API key in an `audio` build only.
pub const AUDIO_ROUTES: &[RouteInfo] = &[
    route("POST", "/v1/audio/speech", "Text to speech"),
    route("POST", "/v1/audio/transcriptions", "Speech to text"),
    route("POST", "/v1/audio/translations", "Speech to English text"),
    route("GET", "/v1/audio/voices", "List a TTS model's voices"),
];

/// Every route this build answers, the public ones first.
pub fn served_routes() -> impl Iterator<Item = &'static RouteInfo> {
    let audio: &[RouteInfo] = if cfg!(feature = "audio") {
        AUDIO_ROUTES
    } else {
        &[]
    };
    PUBLIC_ROUTES.iter().chain(API_ROUTES).chain(audio)
}

/// `path` as the log shows it: `{*model_id}` reads `{model_id}`.
pub fn display_path(path: &str) -> String {
    path.replace("{*", "{")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Every `(method, path)` the `.route(...)` calls in `source` register.
    fn registered(source: &str) -> BTreeSet<(String, String)> {
        let mut out = BTreeSet::new();
        for part in source.split(".route(\"").skip(1) {
            let Some((path, rest)) = part.split_once('"') else {
                continue;
            };
            // The handlers of this route run up to the next `.route(`.
            let rest = rest.split(".route(").next().unwrap_or(rest);
            for (method, call) in [("GET", "get("), ("POST", "post("), ("DELETE", "delete(")] {
                if rest.contains(call) {
                    out.insert((method.to_string(), path.to_string()));
                }
            }
        }
        out
    }

    fn listed(routes: &[&[RouteInfo]]) -> BTreeSet<(String, String)> {
        routes
            .iter()
            .flat_map(|r| r.iter())
            .map(|r| (r.method.to_string(), r.path.to_string()))
            .collect()
    }

    /// The startup log used to omit `/v1/embeddings`, `/v1/messages` and
    /// a dozen other routes.
    #[test]
    fn the_list_matches_the_router() {
        let routes = include_str!("routes.rs");
        let routes = routes.split("#[cfg(test)]").next().unwrap_or(routes);
        assert_eq!(registered(routes), listed(&[API_ROUTES, AUDIO_ROUTES]));
        let startup = include_str!("serving/startup.rs");
        assert_eq!(registered(startup), listed(&[PUBLIC_ROUTES]));
    }

    #[test]
    fn a_wildcard_reads_as_a_parameter() {
        assert_eq!(
            display_path("/v1/models/{*model_id}"),
            "/v1/models/{model_id}"
        );
    }
}

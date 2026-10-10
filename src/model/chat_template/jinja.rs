//! Jinja chat-template rendering.
//!
//! Renders a model's own `chat_template` with the Python behaviour that
//! Hugging Face templates rely on: `trim_blocks`, `lstrip_blocks`, string
//! methods such as `.startswith()` and `.items()`, `raise_exception`,
//! `strftime_now`, and a `tojson` that prints like Python's `json.dumps`.

use std::fmt;
use std::sync::Arc;

use minijinja::value::{Kwargs, Serde, Value};
use minijinja::{Environment, Error, ErrorKind};
use splintr::{AnyTokenizer, FxHashSet};

use super::jinja_json::python_json;
use super::message::ChatMessage;

/// Name the template is registered under, so errors name it.
const TEMPLATE_NAME: &str = "chat_template";

/// Where a Jinja template came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateOrigin {
    /// GGUF metadata `tokenizer.chat_template`.
    Gguf,
    /// `tokenizer_config.json` field `chat_template`.
    TokenizerConfig,
    /// A template file named on the command line.
    File,
    /// A string handed in by code.
    Inline,
}

impl TemplateOrigin {
    /// Short label for `blazr info` and `/api/show`.
    pub fn label(self) -> &'static str {
        match self {
            TemplateOrigin::Gguf => "gguf jinja",
            TemplateOrigin::TokenizerConfig => "tokenizer_config jinja",
            TemplateOrigin::File => "file jinja",
            TemplateOrigin::Inline => "inline jinja",
        }
    }
}

/// Per-request render inputs beyond the messages.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Open the assistant turn at the end of the prompt.
    pub add_generation_prompt: bool,
    /// OpenAI-shaped tool definitions, or `None` when the request has none.
    pub tools: Option<Vec<serde_json::Value>>,
    /// Extra template variables, such as `enable_thinking` or
    /// `reasoning_effort`. A fixed variable name wins over an entry here.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            tools: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// A template that did not compile or did not render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateError {
    origin: TemplateOrigin,
    detail: String,
}

impl TemplateError {
    fn from_minijinja(origin: TemplateOrigin, err: &Error) -> Self {
        Self {
            origin,
            detail: err.to_string(),
        }
    }

    /// Where the failing template came from.
    pub fn origin(&self) -> TemplateOrigin {
        self.origin
    }
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "chat template ({}): {}",
            self.origin.label(),
            self.detail
        )
    }
}

impl std::error::Error for TemplateError {}

/// A compiled Jinja chat template plus the token strings it renders with.
#[derive(Clone)]
pub struct JinjaTemplate {
    source: Arc<str>,
    origin: TemplateOrigin,
    bos_token: String,
    eos_token: String,
    allowed: FxHashSet<String>,
    env: Arc<Environment<'static>>,
}

impl JinjaTemplate {
    /// Compile `source`. A syntax error is reported here, not at request time.
    pub fn new(
        source: &str,
        origin: TemplateOrigin,
        bos_token: &str,
        eos_token: &str,
    ) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        let syntax = env
            .syntax()
            .to_builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| TemplateError::from_minijinja(origin, &e))?;
        env.set_syntax(syntax);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        env.add_template_owned(TEMPLATE_NAME.to_string(), source.to_string())
            .map_err(|e| TemplateError::from_minijinja(origin, &e))?;
        Ok(Self {
            source: Arc::from(source),
            origin,
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
            allowed: FxHashSet::default(),
            env: Arc::new(env),
        })
    }

    /// Compute the special tokens this template can write into a prompt.
    ///
    /// A token counts when the template spells it, or when the template names
    /// the `bos_token`/`eos_token` variable and the tokenizer knows that token.
    pub fn bind_special_tokens(&mut self, tokenizer: &AnyTokenizer) {
        let mut allowed = FxHashSet::default();
        for name in tokenizer.special_tokens().keys() {
            if self.source.contains(name.as_str()) {
                allowed.insert(name.clone());
            }
        }
        for (variable, token) in [
            ("bos_token", &self.bos_token),
            ("eos_token", &self.eos_token),
        ] {
            if self.source.contains(variable) && tokenizer.special_token_id(token).is_some() {
                allowed.insert(token.clone());
            }
        }
        self.allowed = allowed;
    }

    /// The special tokens [`bind_special_tokens`](Self::bind_special_tokens)
    /// found. Empty until it runs.
    pub fn allowed_special(&self) -> &FxHashSet<String> {
        &self.allowed
    }

    /// Where the template came from.
    pub fn origin(&self) -> TemplateOrigin {
        self.origin
    }

    /// The template text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render `messages` under `options`.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> Result<String, TemplateError> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| TemplateError::from_minijinja(self.origin, &e))?;
        let context = render_context(messages, options, &self.bos_token, &self.eos_token);
        template
            .render(Serde(&context))
            .map_err(|e| TemplateError::from_minijinja(self.origin, &e))
    }
}

impl fmt::Debug for JinjaTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JinjaTemplate")
            .field("origin", &self.origin)
            .field("bytes", &self.source.len())
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .field("allowed", &self.allowed)
            .finish()
    }
}

impl PartialEq for JinjaTemplate {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.origin == other.origin
            && self.bos_token == other.bos_token
            && self.eos_token == other.eos_token
            && self.allowed == other.allowed
    }
}

impl Eq for JinjaTemplate {}

/// The variables the template sees.
fn render_context(
    messages: &[ChatMessage],
    options: &RenderOptions,
    bos_token: &str,
    eos_token: &str,
) -> serde_json::Value {
    let mut ctx = serde_json::Map::new();
    for (key, value) in &options.extra {
        ctx.insert(key.clone(), value.clone());
    }
    ctx.insert(
        "messages".to_string(),
        serde_json::Value::Array(messages.iter().map(ChatMessage::template_value).collect()),
    );
    ctx.insert(
        "tools".to_string(),
        match &options.tools {
            Some(tools) => serde_json::Value::Array(tools.clone()),
            None => serde_json::Value::Null,
        },
    );
    ctx.insert(
        "add_generation_prompt".to_string(),
        serde_json::Value::Bool(options.add_generation_prompt),
    );
    ctx.insert("bos_token".to_string(), bos_token.into());
    ctx.insert("eos_token".to_string(), eos_token.into());
    serde_json::Value::Object(ctx)
}

/// `raise_exception(message)`: abort the render with `message`.
fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`: the local time, formatted like Python's `strftime`.
fn strftime_now(format: String) -> Result<Value, Error> {
    use std::fmt::Write;
    let mut out = String::new();
    write!(out, "{}", chrono::Local::now().format(&format)).map_err(|_| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("strftime_now: bad format {format:?}"),
        )
    })?;
    Ok(Value::from(out))
}

/// `tojson`: Python `json.dumps` output, the way transformers overrides it.
///
/// Keyword arguments `indent`, `ensure_ascii` and `sort_keys` are honoured.
/// `separators` is accepted and ignored.
fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let _separators: Option<Value> = kwargs.get("separators")?;
    let json = serde_json::to_value(value)
        .map_err(|e| Error::new(ErrorKind::BadSerialization, e.to_string()))?;
    Ok(Value::from(python_json(
        &json,
        indent,
        ensure_ascii.unwrap_or(false),
        sort_keys.unwrap_or(false),
    )))
}

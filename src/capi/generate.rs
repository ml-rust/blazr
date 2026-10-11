//! `blazr_generate` and `blazr_cancel`.

use std::ffi::{c_char, c_void};
use std::pin::pin;
use std::sync::atomic::Ordering;

use futures::StreamExt;

use super::error::{guard, BlazrStatus, FfiError};
use super::handle::{self, with_executor, BlazrModel, EngineClient, ModelInner};
use super::string::{into_c_string, read_str, write_out};
use crate::config::GenerationConfig;
use crate::engine::Executor;
use boostr::{DType, Runtime};

/// Sampling settings for `blazr_generate`. Start from
/// `blazr_generate_params_default` and set the fields you need.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlazrGenerateParams {
    /// The most tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature. Zero picks the most likely token each step.
    pub temperature: f32,
    /// Nucleus sampling mass. 1.0 turns it off.
    pub top_p: f32,
    /// Keep the k most likely tokens. Zero turns it off.
    pub top_k: u32,
    /// Drop tokens below this fraction of the top probability. Zero turns it off.
    pub min_p: f32,
    /// Penalty for repeated tokens. 1.0 turns it off.
    pub repeat_penalty: f32,
    /// The sampling seed. Read only when `use_seed` is nonzero.
    pub seed: u64,
    /// Nonzero makes sampling repeatable with `seed`.
    pub use_seed: u8,
}

/// Called once per generated token with its UTF-8 text piece.
///
/// `piece` is NUL-terminated and `len` excludes the NUL. A piece can be
/// empty while a multi-byte character is incomplete. The pointer is valid
/// only during the call. Return nonzero to stop generation.
pub type BlazrTokenCallback =
    Option<unsafe extern "C" fn(piece: *const c_char, len: usize, user_data: *mut c_void) -> i32>;

/// The default sampling settings, the same as the CLI defaults.
#[no_mangle]
pub extern "C" fn blazr_generate_params_default() -> BlazrGenerateParams {
    let config = GenerationConfig::default();
    BlazrGenerateParams {
        max_tokens: u32::try_from(config.max_tokens).unwrap_or(u32::MAX),
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: u32::try_from(config.top_k).unwrap_or(u32::MAX),
        min_p: config.min_p,
        repeat_penalty: config.repeat_penalty,
        seed: 0,
        use_seed: 0,
    }
}

fn generation_config(params: &BlazrGenerateParams) -> GenerationConfig {
    GenerationConfig {
        max_tokens: params.max_tokens as usize,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k as usize,
        min_p: params.min_p,
        repeat_penalty: params.repeat_penalty,
        seed: (params.use_seed != 0).then_some(params.seed),
        ..GenerationConfig::default()
    }
}

/// Generate text for `prompt` and stream each token to `callback`.
///
/// `params` can be null for the defaults. `callback` can be null. When
/// `out_text` is not null, `*out_text` receives the whole generated text.
/// Free it with `blazr_string_free`. Returns `BLAZR_STATUS_CANCELLED` when
/// the callback or `blazr_cancel` stopped generation early.
///
/// # Safety
///
/// `prompt` is a NUL-terminated string. `params` is null or points to a
/// `BlazrGenerateParams`. `callback`, when set, is safe to call with
/// `user_data` from this thread. `out_text` is null or writable.
#[no_mangle]
pub unsafe extern "C" fn blazr_generate(
    model: *mut BlazrModel,
    prompt: *const c_char,
    params: *const BlazrGenerateParams,
    callback: BlazrTokenCallback,
    user_data: *mut c_void,
    out_text: *mut *mut c_char,
) -> BlazrStatus {
    guard(|| {
        if !out_text.is_null() {
            // SAFETY: checked non-null, the caller guarantees it is writable.
            unsafe { write_out(out_text, std::ptr::null_mut(), "out_text") }?;
        }
        let inner = handle::lookup(model)?;
        // SAFETY: forwarded from the caller.
        let prompt = unsafe { read_str(prompt, "prompt") }?;
        let params = if params.is_null() {
            blazr_generate_params_default()
        } else {
            // SAFETY: checked non-null, the caller guarantees it is valid.
            unsafe { *params }
        };
        let config = generation_config(&params);
        let sink = TokenSink {
            callback,
            user_data,
        };
        let (text, status) = with_executor!(&inner.executor, executor => {
            run(executor, &inner, prompt, &config, &sink)
        })?;
        if !out_text.is_null() {
            let text = into_c_string(text)?;
            // SAFETY: checked non-null above.
            unsafe { write_out(out_text, text, "out_text") }?;
        }
        Ok(status)
    })
}

/// The caller's callback and its context pointer.
struct TokenSink {
    callback: BlazrTokenCallback,
    user_data: *mut c_void,
}

impl TokenSink {
    /// Pass one piece to the callback. `true` asks generation to stop.
    fn send(&self, piece: &str) -> bool {
        let Some(callback) = self.callback else {
            return false;
        };
        let mut bytes = Vec::with_capacity(piece.len() + 1);
        bytes.extend_from_slice(piece.as_bytes());
        bytes.push(0);
        // SAFETY: the caller of `blazr_generate` vouches for the callback.
        unsafe { callback(bytes.as_ptr().cast(), piece.len(), self.user_data) != 0 }
    }
}

fn run<R>(
    executor: &Executor<R>,
    inner: &ModelInner,
    prompt: &str,
    config: &GenerationConfig,
    sink: &TokenSink,
) -> Result<(String, BlazrStatus), FfiError>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
{
    let prompt_ids = executor.tokenizer().encode(prompt);
    if prompt_ids.is_empty() {
        return Err(FfiError::invalid(
            "prompt encodes to no tokens. Pass a non-empty prompt",
        ));
    }
    inner.cancel.store(false, Ordering::SeqCst);
    inner.runtime.block_on(async {
        let mut text = String::new();
        let mut tokens = pin!(executor.generate(&prompt_ids, config));
        while let Some(token) = tokens.next().await {
            let token =
                token.map_err(|e| FfiError::new(BlazrStatus::Generate, format!("{e:#}")))?;
            text.push_str(&token.text);
            let stop = sink.send(&token.text);
            if stop || inner.cancel.load(Ordering::SeqCst) {
                return Ok((text, BlazrStatus::Cancelled));
            }
        }
        Ok((text, BlazrStatus::Ok))
    })
}

/// Stop the generation running on `model`. It ends before its next token
/// and returns `BLAZR_STATUS_CANCELLED`.
///
/// Call it from any thread. Each generation clears the request when it
/// starts, so a cancel with no generation running has no effect.
#[no_mangle]
pub extern "C" fn blazr_cancel(model: *mut BlazrModel) -> BlazrStatus {
    guard(|| {
        handle::lookup(model)?.cancel.store(true, Ordering::SeqCst);
        Ok(BlazrStatus::Ok)
    })
}

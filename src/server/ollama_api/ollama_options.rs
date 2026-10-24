//! Ollama `options` mapped onto the fields of a chat request.
//!
//! - `num_predict` is `max_tokens`. A negative value takes the config
//!   default. Zero is a 400 from the chat request's own check.
//! - `repeat_last_n` -1 covers the whole history. Zero turns the repeat,
//!   presence and frequency penalties off.
//! - A negative `seed` draws a random one.
//! - `num_ctx` is a 400 above the model's context length, else ignored.
//! - Every other key, such as `num_gpu` or `num_thread`, is ignored.

use serde::Serialize;
use serde_json::{Map, Value};

use super::ollama_types::OllamaOptions;

/// Insert `value` under `key` when it is set.
fn put<T: Serialize>(fields: &mut Map<String, Value>, key: &str, value: Option<T>) {
    if let Some(v) = value.and_then(|v| serde_json::to_value(v).ok()) {
        fields.insert(key.to_string(), v);
    }
}

impl OllamaOptions {
    /// Write the mapped options into `fields`, the JSON of a chat request.
    ///
    /// # Errors
    ///
    /// A `repeat_last_n` below -1. The handler answers 400.
    pub(super) fn apply(&self, fields: &mut Map<String, Value>) -> Result<(), String> {
        put(fields, "temperature", self.temperature);
        put(fields, "top_p", self.top_p);
        put(fields, "top_k", self.top_k);
        put(fields, "min_p", self.min_p);
        put(fields, "typical_p", self.typical_p);
        put(fields, "repeat_penalty", self.repeat_penalty);
        put(fields, "presence_penalty", self.presence_penalty);
        put(fields, "frequency_penalty", self.frequency_penalty);
        put(fields, "mirostat", self.mirostat);
        put(fields, "mirostat_tau", self.mirostat_tau);
        put(fields, "mirostat_eta", self.mirostat_eta);
        put(fields, "stop", self.stop.clone());
        put(fields, "seed", self.seed.filter(|s| *s >= 0));
        put(fields, "max_tokens", self.num_predict.filter(|n| *n >= 0));
        match self.repeat_last_n {
            None => {}
            // blazr reads a window of 0 as the whole history.
            Some(-1) => put(fields, "repeat_last_n", Some(0)),
            Some(0) => {
                put(fields, "repeat_penalty", Some(1.0));
                put(fields, "presence_penalty", Some(0.0));
                put(fields, "frequency_penalty", Some(0.0));
            }
            Some(n) if n > 0 => put(fields, "repeat_last_n", Some(n)),
            Some(n) => {
                return Err(format!(
                    "repeat_last_n must be -1, 0 or a positive count, got {n}"
                ))
            }
        }
        for key in self.ignored.keys() {
            tracing::debug!(option = key.as_str(), "Ollama option ignored");
        }
        Ok(())
    }
}

/// Check `num_ctx` against the model's context length `max_context_len`.
///
/// # Errors
///
/// A `num_ctx` above a known context length. The handler answers 400.
pub(super) fn check_num_ctx(
    num_ctx: Option<usize>,
    max_context_len: Option<usize>,
) -> Result<(), String> {
    match (num_ctx, max_context_len) {
        (Some(asked), Some(max)) if asked > max => Err(format!(
            "num_ctx {asked} is above the model's context length of {max}. \
             Set num_ctx to {max} or less"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::chat_types::ChatRequest;
    use serde_json::json;

    fn mapped(options: Value) -> Result<Map<String, Value>, String> {
        let options: OllamaOptions = serde_json::from_value(options).map_err(|e| e.to_string())?;
        let mut fields = Map::new();
        options.apply(&mut fields)?;
        Ok(fields)
    }

    /// The chat request of `options` over one user message.
    fn request(options: Value) -> ChatRequest {
        let mut fields = mapped(options).unwrap();
        fields.insert("model".into(), json!("m"));
        fields.insert(
            "messages".into(),
            json!([{"role": "user", "content": "hi"}]),
        );
        serde_json::from_value(Value::Object(fields)).unwrap()
    }

    #[test]
    fn each_sampling_key_reaches_its_field() {
        let req = request(json!({
            "temperature": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.1,
            "typical_p": 0.8, "repeat_penalty": 1.2, "repeat_last_n": 32,
            "presence_penalty": 0.3, "frequency_penalty": 0.4, "seed": 7,
            "mirostat": 2, "mirostat_tau": 4.0, "mirostat_eta": 0.2,
            "stop": ["a", "b"], "num_predict": 12,
        }));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.min_p, Some(0.1));
        assert_eq!(req.typical_p, Some(0.8));
        assert_eq!(req.repeat_penalty, Some(1.2));
        assert_eq!(req.repeat_last_n, Some(32));
        assert_eq!(req.presence_penalty, Some(0.3));
        assert_eq!(req.frequency_penalty, Some(0.4));
        assert_eq!(req.seed, Some(7));
        assert_eq!(req.mirostat, Some(2));
        assert_eq!(req.mirostat_tau, Some(4.0));
        assert_eq!(req.mirostat_eta, Some(0.2));
        assert_eq!(req.stop, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(req.max_tokens, Some(12));
    }

    #[test]
    fn negative_num_predict_takes_the_default_and_zero_is_an_error() {
        for n in [-1, -2] {
            assert_eq!(request(json!({"num_predict": n})).max_tokens, None, "{n}");
        }
        let zero = request(json!({"num_predict": 0}));
        assert_eq!(zero.max_tokens, Some(0));
        assert!(zero.check_fields().is_err());
    }

    #[test]
    fn repeat_last_n_minus_one_covers_the_history_and_zero_disables_penalties() {
        assert_eq!(request(json!({"repeat_last_n": -1})).repeat_last_n, Some(0));
        let off = request(json!({"repeat_last_n": 0, "repeat_penalty": 1.5}));
        assert_eq!(off.repeat_penalty, Some(1.0));
        assert_eq!(off.presence_penalty, Some(0.0));
        assert_eq!(off.frequency_penalty, Some(0.0));
        assert!(mapped(json!({"repeat_last_n": -2})).is_err());
    }

    #[test]
    fn a_negative_seed_draws_a_random_one() {
        assert_eq!(request(json!({"seed": -1})).seed, None);
    }

    #[test]
    fn ignored_keys_set_nothing() {
        let fields = mapped(json!({
            "num_keep": 4, "tfs_z": 1.0, "penalize_newline": true, "num_batch": 8,
            "num_gpu": 1, "main_gpu": 0, "num_thread": 4, "use_mmap": true,
            "no_such_key": 1, "num_ctx": 2048,
        }))
        .unwrap();
        assert!(fields.is_empty(), "{fields:?}");
    }

    #[test]
    fn num_ctx_is_an_error_only_above_the_context_length() {
        assert!(check_num_ctx(Some(8192), Some(4096)).is_err());
        assert!(check_num_ctx(Some(4096), Some(4096)).is_ok());
        assert!(check_num_ctx(Some(8192), None).is_ok());
        assert!(check_num_ctx(None, Some(4096)).is_ok());
    }

    #[test]
    fn more_than_four_stop_sequences_is_an_error() {
        let five = request(json!({"stop": ["1", "2", "3", "4", "5"]}));
        assert!(five.check_fields().is_err());
        let one = request(json!({"stop": "END"}));
        assert_eq!(one.stop, Some(vec!["END".to_string()]));
        assert!(one.check_fields().is_ok());
    }
}

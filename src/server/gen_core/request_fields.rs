//! Request field forms that chat and completions share: `stop`, `logprobs`
//! and `stream_options`.

use serde::{Deserialize, Deserializer};

/// The `stream_options` field of a streamed request.
///
/// A non-streaming request ignores it.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamOptions {
    /// Send a usage chunk with empty `choices` before `[DONE]`.
    /// Every other chunk then carries `"usage": null`.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

impl StreamOptions {
    /// Whether `options` asks for the usage chunk. Absent or `null` reads as no.
    pub fn include_usage(options: Option<&Self>) -> bool {
        options.and_then(|o| o.include_usage).unwrap_or(false)
    }
}

/// The most stop sequences a request can carry.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// The most top logprobs a completions `logprobs` integer can ask for.
pub const MAX_COMPLETION_LOGPROBS: i64 = 5;

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

/// Read `stop` as one string or an array of strings.
///
/// Use with `#[serde(default, deserialize_with = "string_or_array")]`.
/// `null` reads as absent.
pub fn string_or_array<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(
        Option::<OneOrMany>::deserialize(deserializer)?.map(|value| match value {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }),
    )
}

/// Check the stop sequence count. The handler answers 400 on an error.
pub fn check_stop(stop: Option<&[String]>) -> Result<(), String> {
    match stop {
        Some(s) if s.len() > MAX_STOP_SEQUENCES => Err(format!(
            "stop accepts at most {MAX_STOP_SEQUENCES} sequences, got {}",
            s.len()
        )),
        _ => Ok(()),
    }
}

/// The completions `logprobs` field: a boolean, or an integer 0 to 5.
///
/// The integer is the number of top logprobs per token.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(untagged)]
pub enum LogprobsRequest {
    Flag(bool),
    Top(i64),
}

impl LogprobsRequest {
    /// Check the integer range. The handler answers 400 on an error.
    pub fn check(self) -> Result<(), String> {
        match self {
            Self::Top(k) if !(0..=MAX_COMPLETION_LOGPROBS).contains(&k) => Err(format!(
                "logprobs must be an integer from 0 to {MAX_COMPLETION_LOGPROBS}, got {k}"
            )),
            _ => Ok(()),
        }
    }

    /// The generation config pair: whether to return logprobs, and the
    /// top logprobs count when the request fixes one.
    ///
    /// An out-of-range integer clamps. [`Self::check`] rejects it first.
    pub fn resolve(self) -> (bool, Option<usize>) {
        match self {
            Self::Flag(on) => (on, None),
            Self::Top(k) => (true, Some(k.clamp(0, MAX_COMPLETION_LOGPROBS) as usize)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Body {
        #[serde(default, deserialize_with = "string_or_array")]
        stop: Option<Vec<String>>,
        #[serde(default)]
        logprobs: Option<LogprobsRequest>,
    }

    fn body(json: &str) -> Body {
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn stop_reads_as_a_string_an_array_or_null() {
        assert_eq!(body(r#"{"stop":"END"}"#).stop, Some(vec!["END".into()]));
        assert_eq!(
            body(r#"{"stop":["a","b"]}"#).stop,
            Some(vec!["a".into(), "b".into()])
        );
        assert_eq!(body(r#"{"stop":null}"#).stop, None);
        assert_eq!(body("{}").stop, None);
    }

    #[test]
    fn stop_allows_four_sequences_and_refuses_five() {
        let four: Vec<String> = (0..4).map(|i| i.to_string()).collect();
        assert!(check_stop(Some(&four)).is_ok());
        let five: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert!(check_stop(Some(&five)).unwrap_err().contains("at most 4"));
        assert!(check_stop(None).is_ok());
    }

    #[test]
    fn logprobs_reads_as_a_boolean_or_an_integer() {
        assert_eq!(
            body(r#"{"logprobs":true}"#).logprobs,
            Some(LogprobsRequest::Flag(true))
        );
        assert_eq!(
            body(r#"{"logprobs":3}"#).logprobs,
            Some(LogprobsRequest::Top(3))
        );
    }

    #[test]
    fn logprobs_maps_to_the_generation_pair() {
        assert_eq!(LogprobsRequest::Flag(false).resolve(), (false, None));
        assert_eq!(LogprobsRequest::Flag(true).resolve(), (true, None));
        assert_eq!(LogprobsRequest::Top(0).resolve(), (true, Some(0)));
        assert_eq!(LogprobsRequest::Top(5).resolve(), (true, Some(5)));
    }

    #[derive(Deserialize)]
    struct StreamBody {
        #[serde(default)]
        stream_options: Option<StreamOptions>,
    }

    fn include_usage(json: &str) -> bool {
        let parsed: StreamBody = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}"));
        StreamOptions::include_usage(parsed.stream_options.as_ref())
    }

    #[test]
    fn include_usage_reads_true_only_when_set() {
        assert!(include_usage(
            r#"{"stream_options":{"include_usage":true}}"#
        ));
        assert!(!include_usage(
            r#"{"stream_options":{"include_usage":false}}"#
        ));
        assert!(!include_usage(
            r#"{"stream_options":{"include_usage":null}}"#
        ));
        assert!(!include_usage(r#"{"stream_options":{}}"#));
        assert!(!include_usage(r#"{"stream_options":null}"#));
        assert!(!include_usage("{}"));
    }

    #[test]
    fn a_non_boolean_include_usage_is_a_parse_error() {
        let parsed =
            serde_json::from_str::<StreamBody>(r#"{"stream_options":{"include_usage":1}}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn logprobs_outside_zero_to_five_is_an_error() {
        assert!(LogprobsRequest::Top(0).check().is_ok());
        assert!(LogprobsRequest::Top(5).check().is_ok());
        assert!(LogprobsRequest::Top(6).check().is_err());
        assert!(LogprobsRequest::Top(-1).check().is_err());
        assert!(LogprobsRequest::Flag(true).check().is_ok());
    }
}

//! `/v1/audio/transcriptions` response types, and the `srt`/`vtt` renderers
//! that turn segments into their respective subtitle text formats.

use serde::Serialize;

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct VerboseTranscriptionResponse {
    pub text: String,
    pub language: String,
    pub duration: f64,
    /// Present when `segment` is in the requested granularities, which is
    /// every request except one that names `word` only — matching OpenAI's
    /// default granularity of `["segment"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
    /// Present only when `word` is in the requested `timestamp_granularities[]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// One OpenAI-shaped word timestamp: `{word, start, end}`, seconds on the
/// full input's clock (a later chunk's words are already offset).
#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

pub(super) const VALID_TRANSCRIPTION_FORMATS: &[&str] =
    &["json", "text", "verbose_json", "vtt", "srt"];
pub(super) const DEFAULT_TRANSCRIPTION_FORMAT: &str = "json";

/// The only values OpenAI accepts inside `timestamp_granularities[]`.
pub(super) const VALID_TIMESTAMP_GRANULARITIES: &[&str] = &["word", "segment"];

/// What `verbose_json` renders, derived from the request's
/// `timestamp_granularities[]` values and its `response_format`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Granularities {
    pub want_segments: bool,
    pub want_words: bool,
}

/// Validates `timestamp_granularities[]` against OpenAI's rules and resolves
/// what the response renders:
///
/// - Any value outside [`VALID_TIMESTAMP_GRANULARITIES`] is a 400.
/// - A non-empty list is only valid with `response_format=verbose_json`; a 400
///   names the field and tells the caller to set that format.
/// - No values given defaults to `["segment"]`, matching OpenAI. `word` adds
///   `words`; `segment`'s absence (a `word`-only request) drops `segments`.
///
/// Returns `Err(message)` for either rejection, `Ok(Granularities)` otherwise.
pub(super) fn validate_timestamp_granularities(
    values: &[String],
    response_format: &str,
) -> Result<Granularities, String> {
    for v in values {
        if !VALID_TIMESTAMP_GRANULARITIES.contains(&v.as_str()) {
            return Err(format!(
                "Invalid timestamp_granularities[] value '{}'. Must be one of: {}",
                v,
                VALID_TIMESTAMP_GRANULARITIES.join(", ")
            ));
        }
    }
    if !values.is_empty() && response_format != "verbose_json" {
        return Err(format!(
            "timestamp_granularities[] requires response_format='verbose_json' (got '{}'); \
             set response_format=verbose_json to use it",
            response_format
        ));
    }
    if values.is_empty() {
        return Ok(Granularities {
            want_segments: true,
            want_words: false,
        });
    }
    Ok(Granularities {
        want_segments: values.iter().any(|v| v == "segment"),
        want_words: values.iter().any(|v| v == "word"),
    })
}

/// Format a duration in seconds as an SRT timestamp `HH:MM:SS,mmm`.
pub(super) fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms / 60_000) % 60;
    let s = (total_ms / 1000) % 60;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format a duration in seconds as a WebVTT timestamp `HH:MM:SS.mmm`.
pub(super) fn format_vtt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms / 60_000) % 60;
    let s = (total_ms / 1000) % 60;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

pub(super) fn render_srt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub(super) fn render_vtt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        out.push_str(&format!(
            "{} --> {}\n",
            format_vtt_time(seg.start),
            format_vtt_time(seg.end)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_response_serialization() {
        let resp = TranscriptionResponse {
            text: "Hello world".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["text"], "Hello world");
    }

    #[test]
    fn verbose_transcription_response_serialization() {
        let resp = VerboseTranscriptionResponse {
            text: "Hello".to_string(),
            language: "en".to_string(),
            duration: 1.5,
            segments: Some(vec![TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: "Hello".to_string(),
            }]),
            words: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["language"], "en");
        assert_eq!(json["duration"], 1.5);
        assert_eq!(json["segments"][0]["id"], 0);
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
        assert_eq!(json["segments"][0]["text"], "Hello");
        assert!(json.get("words").is_none());
    }

    #[test]
    fn verbose_transcription_response_with_words() {
        let resp = VerboseTranscriptionResponse {
            text: "Hi".to_string(),
            language: "en".to_string(),
            duration: 1.0,
            segments: None,
            words: Some(vec![TranscriptionWord {
                word: "Hi".to_string(),
                start: 0.0,
                end: 0.5,
            }]),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json.get("segments").is_none());
        assert_eq!(json["words"][0]["word"], "Hi");
        assert_eq!(json["words"][0]["start"], 0.0);
        assert_eq!(json["words"][0]["end"], 0.5);
    }

    #[test]
    fn transcription_format_validation() {
        for fmt in VALID_TRANSCRIPTION_FORMATS {
            assert!(VALID_TRANSCRIPTION_FORMATS.contains(fmt));
        }
        assert!(!VALID_TRANSCRIPTION_FORMATS.contains(&"invalid"));
    }

    #[test]
    fn srt_time_format() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(61.123), "00:01:01,123");
        assert_eq!(format_srt_time(3661.001), "01:01:01,001");
    }

    #[test]
    fn vtt_time_format() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
        assert_eq!(format_vtt_time(3661.001), "01:01:01.001");
    }

    #[test]
    fn srt_render_segments() {
        let segs = vec![
            TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: "Hello".into(),
            },
            TranscriptionSegment {
                id: 1,
                start: 1.5,
                end: 3.2,
                text: "world".into(),
            },
        ];
        let out = render_srt(&segs);
        assert!(out.contains("1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"));
        assert!(out.contains("2\n00:00:01,500 --> 00:00:03,200\nworld\n\n"));
    }

    #[test]
    fn vtt_render_segments() {
        let segs = vec![TranscriptionSegment {
            id: 0,
            start: 0.0,
            end: 2.5,
            text: "Hi".into(),
        }];
        let out = render_vtt(&segs);
        assert!(out.starts_with("WEBVTT\n\n"));
        assert!(out.contains("00:00:00.000 --> 00:00:02.500\nHi"));
    }

    #[test]
    fn validates_new_formats() {
        assert!(VALID_TRANSCRIPTION_FORMATS.contains(&"srt"));
        assert!(VALID_TRANSCRIPTION_FORMATS.contains(&"vtt"));
    }

    #[test]
    fn granularities_default_is_segment_only() {
        let g = validate_timestamp_granularities(&[], "verbose_json").unwrap();
        assert!(g.want_segments);
        assert!(!g.want_words);
    }

    #[test]
    fn granularities_word_only_drops_segments() {
        let g = validate_timestamp_granularities(&["word".to_string()], "verbose_json").unwrap();
        assert!(!g.want_segments);
        assert!(g.want_words);
    }

    #[test]
    fn granularities_both_keeps_both() {
        let g = validate_timestamp_granularities(
            &["segment".to_string(), "word".to_string()],
            "verbose_json",
        )
        .unwrap();
        assert!(g.want_segments);
        assert!(g.want_words);
    }

    #[test]
    fn granularities_rejects_unknown_value() {
        let err = validate_timestamp_granularities(&["sentence".to_string()], "verbose_json")
            .unwrap_err();
        assert!(err.contains("sentence"));
    }

    #[test]
    fn granularities_reject_without_verbose_json() {
        let err = validate_timestamp_granularities(&["word".to_string()], "json").unwrap_err();
        assert!(err.contains("timestamp_granularities"));
        assert!(err.contains("verbose_json"));
    }

    #[test]
    fn granularities_empty_ignores_response_format() {
        // No timestamp_granularities[] at all is valid regardless of format —
        // OpenAI only rejects a *non-empty* list paired with a non-verbose format.
        assert!(validate_timestamp_granularities(&[], "json").is_ok());
    }
}

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
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

pub(super) const VALID_TRANSCRIPTION_FORMATS: &[&str] =
    &["json", "text", "verbose_json", "vtt", "srt"];
pub(super) const DEFAULT_TRANSCRIPTION_FORMAT: &str = "json";

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
            segments: vec![TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: "Hello".to_string(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["language"], "en");
        assert_eq!(json["duration"], 1.5);
        assert_eq!(json["segments"][0]["id"], 0);
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
        assert_eq!(json["segments"][0]["text"], "Hello");
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
}

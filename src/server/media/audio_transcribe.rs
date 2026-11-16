//! Whisper transcription of any-length audio for the HTTP routes.
//!
//! blazr holds no core logic. The long-form reading is
//! [`voxr_narrate::transcribe_narration`], the path `blazr transcribe`
//! runs: pieces of at most 20 s cut in silences, each read alone, words
//! in silence dropped, recognizer loops collapsed, and word times settled
//! on the sound. This module adapts a loaded [`WhisperBundle`] to that
//! path's [`Transcriber`], and groups the words it returns into the
//! segments the response formats need.

use anyhow::{anyhow, Result};
use boostr::Runtime;

use voxr::WhisperBundle;
use voxr_narrate::{
    envelope, pieces, transcribe_narration, HeardWord, TranscribeOptions, Transcriber,
    TRANSCRIPT_RATE,
};

use crate::server::handlers::{ServerDevice, ServerRuntime};

/// Sample rate the audio must arrive at.
pub const WHISPER_SAMPLE_RATE: usize = TRANSCRIPT_RATE as usize;

/// Languages Whisper writes without spaces between words. voxr's word
/// split keeps the same list: it splits these on Unicode boundaries only.
const NO_SPACE_LANGUAGES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];

/// What a request asks of the recognizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhisperRequest<'a> {
    /// The request's language code. `None` detects it per piece.
    pub language: Option<&'a str>,
    /// Emit `<|translate|>` (to English) instead of `<|transcribe|>`.
    pub translate: bool,
    /// Decoder step budget per piece.
    pub max_new_tokens: usize,
}

/// A loaded Whisper checkpoint read as the long-form path's
/// [`Transcriber`], under one request's language and task.
///
/// The long-form path tells each read a language, `en` when the request
/// names none. This reader uses the request's own instead, so a request
/// with no language still detects it.
///
/// A checkpoint with `alignment_heads` returns timed words. One without
/// returns each piece's text as one untimed word: the long-form path then
/// spans it over its whole piece.
pub struct WhisperReader<'a> {
    bundle: &'a WhisperBundle<ServerRuntime>,
    client: <ServerRuntime as Runtime>::Client,
    request: WhisperRequest<'a>,
    language: Option<&'static str>,
}

impl<'a> WhisperReader<'a> {
    /// Reads with `bundle` on `device`, whose context is made current on
    /// the calling thread.
    pub fn new(
        bundle: &'a WhisperBundle<ServerRuntime>,
        device: &ServerDevice,
        request: WhisperRequest<'a>,
    ) -> Self {
        Self {
            bundle,
            client: ServerRuntime::default_client(device),
            request,
            language: None,
        }
    }

    /// The language the first piece decoded under: the request's, or the
    /// detected one. `None` before any read, and for an english-only
    /// checkpoint.
    pub fn language(&self) -> Option<&'static str> {
        self.language
    }
}

impl Transcriber for WhisperReader<'_> {
    fn transcribe_words(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
    ) -> voxr_narrate::Result<Vec<HeardWord>> {
        let budget = self
            .bundle
            .generate_options()
            .max_new_tokens
            .min(self.request.max_new_tokens);
        let opts = voxr::TranscribeOptions {
            language: self.request.language,
            translate: self.request.translate,
            max_new_tokens: Some(budget),
        };
        let recognizer_error = |e: voxr::Error| voxr_narrate::Error::Recognizer {
            reason: format!(
                "Whisper checkpoint {} reading {} samples at {sample_rate} Hz: {e}",
                self.bundle.checkpoint_dir.display(),
                samples.len()
            ),
        };
        let rate = sample_rate as usize;
        let (result, words) = if self.bundle.generation.alignment_heads.is_some() {
            let (result, words) = self
                .bundle
                .transcribe_words(&self.client, samples, rate, &opts)
                .map_err(recognizer_error)?;
            (result, words.iter().map(HeardWord::from).collect())
        } else {
            let result = self
                .bundle
                .transcribe(&self.client, samples, rate, &opts)
                .map_err(recognizer_error)?;
            let whole = vec![HeardWord {
                text: result.text.clone(),
                start: f64::NAN,
                end: f64::NAN,
            }];
            (result, whole)
        };
        if self.language.is_none() {
            self.language = result.language;
        }
        Ok(words)
    }
}

/// One stretch of the audio the long-form path read alone, with the words
/// heard in it.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start in seconds.
    pub start: f64,
    /// End in seconds.
    pub end: f64,
    /// The words, times in seconds on the whole input's clock.
    pub words: Vec<HeardWord>,
}

impl Segment {
    /// The words joined by `separator`.
    pub fn text(&self, separator: &str) -> String {
        join_words(self.words.iter(), separator)
    }
}

/// Any-length audio read into segments.
#[derive(Debug, Clone, PartialEq)]
pub struct LongTranscript {
    /// The segments that hold words, in time order.
    pub segments: Vec<Segment>,
}

impl LongTranscript {
    /// Every word, in time order.
    pub fn words(&self) -> impl Iterator<Item = &HeardWord> {
        self.segments.iter().flat_map(|s| s.words.iter())
    }

    /// The whole text, the words joined by `separator`.
    pub fn text(&self, separator: &str) -> String {
        join_words(self.words(), separator)
    }
}

/// The text between two words in `language`: none for a language written
/// without spaces, one space otherwise.
pub fn word_separator(language: Option<&str>) -> &'static str {
    match language {
        Some(l) if NO_SPACE_LANGUAGES.contains(&l) => "",
        _ => " ",
    }
}

fn join_words<'a>(words: impl Iterator<Item = &'a HeardWord>, separator: &str) -> String {
    words
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(separator)
}

/// Reads 16 kHz mono `samples` of any length with `transcriber` through
/// the long-form path, and groups the words by the piece their start
/// falls in. `language` is the request's, passed on to each read.
pub fn transcribe_long(
    samples: &[f32],
    transcriber: &mut dyn Transcriber,
    language: Option<&str>,
) -> Result<LongTranscript> {
    if samples.is_empty() {
        return Err(anyhow!("no audio samples to transcribe"));
    }
    let opts = TranscribeOptions {
        language: language.map(str::to_string),
    };
    let read = transcribe_narration(samples, transcriber, &opts)
        .map_err(|e| anyhow!("long-form transcription: {e}"))?;
    let rate = WHISPER_SAMPLE_RATE as f64;
    let mut segments: Vec<Segment> = pieces(&envelope(samples), samples.len())
        .into_iter()
        .map(|(a, b)| Segment {
            start: a as f64 / rate,
            end: b as f64 / rate,
            words: Vec::new(),
        })
        .collect();
    let Some(last) = segments.len().checked_sub(1) else {
        return Ok(LongTranscript {
            segments: Vec::new(),
        });
    };
    for w in read.words {
        let at = segments.partition_point(|s| s.end <= w.start).min(last);
        segments[at].words.push(w);
    }
    segments.retain(|s| !s.words.is_empty());
    Ok(LongTranscript { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hears one word a second into every piece, and records each read.
    struct Scripted {
        reads: Vec<(usize, Option<String>)>,
    }

    impl Transcriber for Scripted {
        fn transcribe_words(
            &mut self,
            samples: &[f32],
            _sample_rate: u32,
            language: Option<&str>,
        ) -> voxr_narrate::Result<Vec<HeardWord>> {
            self.reads
                .push((samples.len(), language.map(str::to_string)));
            Ok(vec![HeardWord {
                text: format!(" w{}", self.reads.len()),
                start: 1.0,
                end: 2.0,
            }])
        }
    }

    /// 70 s of 4 s sounds, each followed by 1 s of silence.
    fn speech() -> Vec<f32> {
        let rate = WHISPER_SAMPLE_RATE;
        let mut x = vec![0.0f32; 70 * rate];
        for k in 0..14 {
            let from = k * 5 * rate;
            x[from..from + 4 * rate].fill(0.3);
        }
        x
    }

    #[test]
    fn long_audio_reads_in_pieces_under_twenty_seconds() -> Result<()> {
        let x = speech();
        let mut t = Scripted { reads: Vec::new() };
        let out = transcribe_long(&x, &mut t, Some("ms"))?;
        assert!(t.reads.len() >= 4, "{} reads", t.reads.len());
        for (len, language) in &t.reads {
            assert!(*len <= 20 * WHISPER_SAMPLE_RATE, "a read of {len} samples");
            assert_eq!(language.as_deref(), Some("ms"));
        }
        // Each piece's word lands on the input's clock, in its own segment.
        assert_eq!(out.segments.len(), t.reads.len());
        for (i, s) in out.segments.iter().enumerate() {
            assert_eq!(s.words.len(), 1);
            assert!(s.words[0].start >= s.start && s.words[0].start < s.end);
            assert_eq!(s.text(" "), format!("w{}", i + 1));
        }
        assert!(out.segments.windows(2).all(|p| p[0].end <= p[1].start));
        assert_eq!(out.segments[0].start, 0.0);
        let names: Vec<String> = (1..=t.reads.len()).map(|i| format!("w{i}")).collect();
        assert_eq!(out.text(" "), names.join(" "));
        Ok(())
    }

    #[test]
    fn silence_has_no_segments() -> Result<()> {
        let x = vec![0.0f32; 5 * WHISPER_SAMPLE_RATE];
        let mut t = Scripted { reads: Vec::new() };
        let out = transcribe_long(&x, &mut t, None)?;
        assert!(out.segments.is_empty());
        assert_eq!(out.text(" "), "");
        Ok(())
    }

    #[test]
    fn empty_audio_is_an_error() {
        let mut t = Scripted { reads: Vec::new() };
        assert!(transcribe_long(&[], &mut t, None).is_err());
    }

    #[test]
    fn languages_without_spaces_join_with_none() {
        assert_eq!(word_separator(Some("zh")), "");
        assert_eq!(word_separator(Some("ja")), "");
        assert_eq!(word_separator(Some("ms")), " ");
        assert_eq!(word_separator(None), " ");
    }
}

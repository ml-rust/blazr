//! WAV/PCM response body encoding for `/v1/audio/speech`.

use boostr_audio::{encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16, Error};

/// Encode synthesized samples per the request's `response_format`, returning
/// body bytes and the HTTP content type. Caller has already validated
/// `response_format` against [`super::request::VALID_TTS_FORMATS`] /
/// [`super::request::NATIVE_TTS_FORMATS`].
pub(super) fn encode_speech(
    response_format: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(Vec<u8>, &'static str), Error> {
    match response_format {
        "pcm" => Ok((encode_pcm16_raw(samples), "audio/L16")),
        "f32" => encode_wav_f32(samples, sample_rate).map(|b| (b, "audio/wav")),
        _ => encode_wav_pcm16(samples, sample_rate).map(|b| (b, "audio/wav")),
    }
}

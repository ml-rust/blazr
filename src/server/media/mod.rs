//! Request media: image and audio inputs, base64, and audio upload decoding.

#[cfg(feature = "audio")]
pub(super) mod audio_decode;
#[cfg(feature = "audio")]
pub(super) mod audio_transcribe;
pub(super) mod encoding;
pub mod multimodal;
pub(super) mod multimodal_audio;

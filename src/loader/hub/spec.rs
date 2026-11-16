//! Telling a path from a Hugging Face id in a model spec.

/// Whether `spec` names a path rather than a Hugging Face id: it starts
/// with `.`, `/` or `~`, holds a backslash or a drive prefix such as `C:`, or
/// holds more than one `/`. An id is `name` or `owner/name`.
pub fn looks_like_path(spec: &str) -> bool {
    let drive = spec.as_bytes().get(1) == Some(&b':');
    spec.starts_with(['.', '/', '~'])
        || spec.contains('\\')
        || drive
        || spec.matches('/').count() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_is_told_from_an_id() {
        for spec in [
            "./models/whisper",
            "../whisper",
            "/opt/models/whisper-base",
            "~/models/whisper",
            "models/openai/whisper-base",
            "C:\\models\\whisper",
            ".hidden",
        ] {
            assert!(looks_like_path(spec), "{spec}");
        }
        for id in ["openai/whisper-base", "kokoro", "facebook/mms-tts-zlm"] {
            assert!(!looks_like_path(id), "{id}");
        }
    }
}

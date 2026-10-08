//! The narration skill's config (`$XDG_CONFIG_HOME/motion-video/config.json`),
//! read with each field's type checked.

use std::path::{Path, PathBuf};

use serde_json::Value;
use voxr_narrate::CommandError;

use super::paths::config_file;
use super::tts_section::TtsSection;

/// The user's config, shared with the narration skill.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MotionConfig {
    /// The file it was read from.
    pub file: PathBuf,
    /// `"tts"`.
    pub tts: TtsSection,
    /// `"asr.model"`: the recognizer for every language.
    pub asr_model: Option<String>,
    /// `"models"`: the model folder.
    pub models: Option<String>,
}

/// A config field of the wrong type, naming the field and the fix.
fn wrong(file: &Path, field: &str, want: &str, got: &Value) -> CommandError {
    CommandError::Failed(format!(
        "{}: \"{field}\" is {got}, not {want}. Fix it, or delete it to use the defaults.",
        file.display()
    ))
}

/// The fields of one config block, read with their types checked.
struct Block<'a> {
    file: &'a Path,
    prefix: &'a str,
    value: Option<&'a Value>,
}

impl Block<'_> {
    fn get(&self, key: &str) -> Option<&Value> {
        self.value.and_then(|v| v.get(key)).filter(|v| !v.is_null())
    }

    fn name(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    fn text(&self, key: &str) -> Result<Option<String>, CommandError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(v) => Err(wrong(self.file, &self.name(key), "a string", v)),
        }
    }

    fn number(&self, key: &str) -> Result<Option<f64>, CommandError> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| wrong(self.file, &self.name(key), "a number", v)),
        }
    }

    fn whole(&self, key: &str) -> Result<Option<u32>, CommandError> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| wrong(self.file, &self.name(key), "a whole number", v)),
        }
    }

    fn flag(&self, key: &str) -> Result<Option<bool>, CommandError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(wrong(self.file, &self.name(key), "true or false", v)),
        }
    }
}

/// Reads the config. A missing file is the defaults. A file that is not
/// JSON, or a field of the wrong type, fails the run with the skill's fix.
pub fn read_config() -> Result<MotionConfig, CommandError> {
    let file = config_file();
    if !file.exists() {
        return Ok(MotionConfig {
            file,
            ..MotionConfig::default()
        });
    }
    let bytes = std::fs::read(&file).map_err(|e| {
        CommandError::Failed(format!(
            "reading {}: {e}. Fix its permissions, or delete it to use the defaults.",
            file.display()
        ))
    })?;
    let raw: Value = serde_json::from_slice(&bytes).map_err(|e| {
        CommandError::Failed(format!(
            "{} is not valid JSON ({e}). Fix it, or delete it to use the defaults.",
            file.display()
        ))
    })?;
    let tts = Block {
        file: &file,
        prefix: "tts.",
        value: raw.get("tts"),
    };
    let section = TtsSection {
        engine: tts.text("engine")?,
        command: tts.text("command")?,
        model: tts.text("model")?,
        voice: tts.text("voice")?,
        reference: tts.text("reference")?,
        style: tts.text("style")?,
        speed: tts.number("speed")?,
        pace: tts.number("pace")?,
        seed: tts.number("seed")?,
        cfg: tts.number("cfg")?,
        steps: tts.whole("steps")?,
        hifi: tts.flag("hifi")?,
        one_call: tts.flag("oneCall")?,
        checkpoint: tts.text("checkpoint")?,
    };
    let asr = Block {
        file: &file,
        prefix: "asr.",
        value: raw.get("asr"),
    };
    let top = Block {
        file: &file,
        prefix: "",
        value: Some(&raw),
    };
    Ok(MotionConfig {
        tts: section,
        asr_model: asr.text("model")?,
        models: top.text("models")?,
        file,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_of_the_wrong_type_names_it() {
        let file = PathBuf::from("/c/config.json");
        let value = serde_json::json!({"speed": "fast", "steps": 1.5, "hifi": 1});
        let block = Block {
            file: &file,
            prefix: "tts.",
            value: Some(&value),
        };
        assert_eq!(
            block.number("speed").err().map(|e| e.message().to_string()),
            Some("/c/config.json: \"tts.speed\" is \"fast\", not a number. Fix it, or delete it to use the defaults.".to_string())
        );
        assert!(block.whole("steps").is_err());
        assert!(block.flag("hifi").is_err());
        assert_eq!(block.text("voice").ok().flatten(), None);
    }
}

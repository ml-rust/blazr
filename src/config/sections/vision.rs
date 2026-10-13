//! Vision tower settings taken from the command line.

use std::path::PathBuf;

use super::InferenceConfig;

/// `--mmproj` and the image token limits from the command line. Each set
/// value replaces the model config's; `None` keeps it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VisionOverrides {
    pub mmproj: Option<PathBuf>,
    pub image_min_tokens: Option<usize>,
    pub image_max_tokens: Option<usize>,
}

impl VisionOverrides {
    /// Write the set values over `inference`.
    pub fn apply(&self, inference: &mut InferenceConfig) {
        if let Some(mmproj) = &self.mmproj {
            inference.mmproj = Some(mmproj.clone());
        }
        if let Some(min) = self.image_min_tokens {
            inference.image_min_tokens = Some(min);
        }
        if let Some(max) = self.image_max_tokens {
            inference.image_max_tokens = Some(max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_overrides_replace_only_set_values() {
        let mut inference = InferenceConfig {
            mmproj: Some(PathBuf::from("from-config.gguf")),
            image_max_tokens: Some(1024),
            ..Default::default()
        };
        VisionOverrides {
            mmproj: None,
            image_min_tokens: Some(16),
            image_max_tokens: None,
        }
        .apply(&mut inference);
        assert_eq!(inference.mmproj, Some(PathBuf::from("from-config.gguf")));
        assert_eq!(inference.image_min_tokens, Some(16));
        assert_eq!(inference.image_max_tokens, Some(1024));

        VisionOverrides {
            mmproj: Some(PathBuf::from("cli.gguf")),
            ..Default::default()
        }
        .apply(&mut inference);
        assert_eq!(inference.mmproj, Some(PathBuf::from("cli.gguf")));
    }
}

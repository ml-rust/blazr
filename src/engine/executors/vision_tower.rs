//! The vision tower an executor holds beside a `qwen35` model: the encoder
//! loaded from an mmproj GGUF, its preprocessing config with the token
//! limits applied, and the three marker ids the tokenizer names.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::model::qwen35::{ImageEmbeds, ImageGrid, VisionMarkers};
use boostr::model::vision::qwen3vl::{
    load_qwen3vl_vision_gguf, preprocess_bytes, PreprocessedImage, Qwen3VlVision,
    Qwen3VlVisionConfig,
};
use boostr::model::ModelClient;
use boostr::{DType, Runtime};
use splintr::AnyTokenizer;

/// Text of the marker before an image's rows.
pub const VISION_START: &str = "<|vision_start|>";
/// Text of the marker after an image's rows.
pub const VISION_END: &str = "<|vision_end|>";
/// Text of the placeholder the prompt plan expands to an image's rows.
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// The three markers in the order the server inserts them per image.
pub const MARKER_NAMES: [&str; 3] = [VISION_START, IMAGE_PAD, VISION_END];
/// What the server writes into a message at each image's position.
pub const IMAGE_MARKER_TEXT: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Error text for an image request on an executor without a tower.
pub const NO_TOWER_ERROR: &str = "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj (--mmproj)";

/// Output-token bounds a tower resizes every image into.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageTokenLimits {
    /// Fewest output tokens; `None` keeps the mmproj default.
    pub min_tokens: Option<usize>,
    /// Most output tokens; `None` keeps the mmproj default.
    pub max_tokens: Option<usize>,
}

/// A loaded Qwen3-VL tower with its marker ids.
pub struct VisionTower<R: Runtime<DType = DType>> {
    tower: Qwen3VlVision<R>,
    config: Qwen3VlVisionConfig,
    markers: VisionMarkers,
}

impl<R: Runtime<DType = DType>> VisionTower<R>
where
    R::Client: ModelClient<R>,
{
    /// Load the tower from `mmproj` onto `device` and look the marker ids up
    /// in `tokenizer`.
    ///
    /// # Errors
    ///
    /// The mmproj does not load, a marker is not a special token of the
    /// tokenizer, or `limits` is empty or inverted.
    pub fn load(
        mmproj: &Path,
        device: &R::Device,
        tokenizer: &AnyTokenizer,
        limits: ImageTokenLimits,
    ) -> Result<Self> {
        let markers = lookup_markers(tokenizer)?;
        let tower = load_qwen3vl_vision_gguf::<R, _>(mmproj, device)
            .map_err(|e| anyhow!("mmproj {} failed to load: {e}", mmproj.display()))?;
        let mut config = tower.config().clone();
        if let Some(min) = limits.min_tokens {
            config.image_min_tokens = min;
        }
        if let Some(max) = limits.max_tokens {
            config.image_max_tokens = max;
        }
        if config.image_min_tokens == 0 || config.image_min_tokens > config.image_max_tokens {
            return Err(anyhow!(
                "image token limits min={} max={} are not 1 <= min <= max",
                config.image_min_tokens,
                config.image_max_tokens
            ));
        }
        Ok(Self {
            tower,
            config,
            markers,
        })
    }

    /// The preprocessing config in force, token limits applied.
    pub fn config(&self) -> &Qwen3VlVisionConfig {
        &self.config
    }

    /// The marker ids of the tokenizer this tower was loaded against.
    pub fn markers(&self) -> VisionMarkers {
        self.markers
    }

    /// Resize and normalize every image; `images[i]` becomes `out[i]`.
    ///
    /// # Errors
    ///
    /// An image does not decode or has an empty side; the message names its
    /// index.
    pub fn preprocess(&self, images: &[Vec<u8>]) -> Result<Vec<PreprocessedImage>> {
        images
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                preprocess_bytes(bytes, &self.config)
                    .map_err(|e| anyhow!("image {index} failed to preprocess: {e}"))
            })
            .collect()
    }

    /// Encode every preprocessed image, one tower call each, in order.
    pub fn encode<C>(&self, client: &C, images: &[PreprocessedImage]) -> Result<Vec<ImageEmbeds<R>>>
    where
        C: ModelClient<R>,
    {
        images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                let embeds = self
                    .tower
                    .encode_image(client, image)
                    .map_err(|e| anyhow!("image {index} failed to encode: {e}"))?;
                Ok(ImageEmbeds {
                    embeds,
                    grid: grid_of(image),
                })
            })
            .collect()
    }

    /// Check that each marker id appears exactly `n_images` times in `ids`:
    /// the server inserts one group per image, so any other count means the
    /// text spelled a marker itself.
    pub fn check_marker_counts(&self, ids: &[u32], n_images: usize) -> Result<(), String> {
        let m = self.markers;
        for (name, id) in MARKER_NAMES
            .iter()
            .zip([m.vision_start, m.image_pad, m.vision_end])
        {
            let count = ids.iter().filter(|&&t| t == id).count();
            if count != n_images {
                return Err(format!(
                    "message text spells {name}: {count} in the prompt, {n_images} image(s) attached"
                ));
            }
        }
        Ok(())
    }
}

/// The token grid of a preprocessed image.
pub fn grid_of(image: &PreprocessedImage) -> ImageGrid {
    ImageGrid {
        nx: image.nx,
        ny: image.ny,
    }
}

/// The three marker ids from the tokenizer's special-token table.
fn lookup_markers(tokenizer: &AnyTokenizer) -> Result<VisionMarkers> {
    let id = |name: &str| {
        tokenizer
            .special_token_id(name)
            .ok_or_else(|| anyhow!("tokenizer has no special token {name}; mmproj needs it"))
    };
    Ok(VisionMarkers {
        vision_start: id(VISION_START)?,
        vision_end: id(VISION_END)?,
        image_pad: id(IMAGE_PAD)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_text_is_the_three_names_in_order() {
        assert_eq!(IMAGE_MARKER_TEXT, MARKER_NAMES.concat());
    }
}

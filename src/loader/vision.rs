//! Vision projector (mmproj) loading for multimodal GGUF models
//!
//! Handles loading and remapping of vision encoder + projector weights
//! from separate GGUF mmproj files.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::Gguf;
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarBuilder, VarMap};

use crate::config::BlazrConfig;

use super::gguf::config_from_gguf_metadata;

/// Load vision projector weights from a separate GGUF mmproj file.
///
/// GGUF mmproj files contain vision encoder + projector weights.
/// Tensor names are remapped from GGUF convention to boostr's
/// MultimodalModel weight naming convention.
pub fn load_mmproj_tensors<R: Runtime<DType = DType>, P: AsRef<Path>>(
    mmproj_path: P,
    device: &R::Device,
) -> Result<VarMap<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let mmproj_path = mmproj_path.as_ref();

    let mut var_map = VarMap::<R>::from_gguf(mmproj_path, device)
        .map_err(|e| anyhow!("Failed to load mmproj GGUF: {}", e))?;

    tracing::info!("Loaded {} tensors from mmproj GGUF", var_map.len());

    // Remap tensor names from GGUF convention to boostr convention
    let names: Vec<String> = var_map.names().map(|s| s.to_string()).collect();
    let mut remapped = VarMap::<R>::new();
    for name in names {
        let new_name = remap_mmproj_name(&name);
        if let Ok(weight) = var_map.take(&name) {
            remapped.insert_weight(new_name, weight);
        }
    }

    Ok(remapped)
}

/// Load a model from GGUF with a separate mmproj vision file.
///
/// The main GGUF contains the LLM weights, and the mmproj GGUF contains
/// the vision encoder + projector weights. Both are merged into a single
/// VarMap for MultimodalModel construction.
pub fn load_gguf_with_mmproj<R: Runtime<DType = DType>, P: AsRef<Path>>(
    model_path: P,
    mmproj_path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let model_path = model_path.as_ref();
    let mmproj_path = mmproj_path.as_ref();

    let gguf = Gguf::open(model_path).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
    let mut config = config_from_gguf_metadata(&gguf)?;

    // Load main model tensors
    let mut var_map = VarMap::<R>::from_gguf(model_path, device)
        .map_err(|e| anyhow!("Failed to load model GGUF tensors: {}", e))?;

    // Load and merge mmproj tensors
    let mmproj_map = load_mmproj_tensors::<R, _>(mmproj_path, device)?;
    var_map.merge(mmproj_map);

    tracing::info!("Merged {} total tensors (model + mmproj)", var_map.len());

    // If config doesn't have vision settings, detect from mmproj metadata
    if config.model.vision.is_none() {
        let mmproj_gguf = Gguf::open(mmproj_path).ok();
        config.model.vision = detect_vision_config_from_mmproj(mmproj_gguf.as_ref());
    }

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Remap GGUF mmproj tensor name to boostr convention.
///
/// Common mappings:
/// - `v.enc.blk.{i}.attn_q.weight` -> `vision_model.encoder.layers.{i}.q_proj.weight`
/// - `v.patch_embd.weight` -> `vision_model.patch_embedding.weight`
/// - `mm.0.weight` -> `vision_model.projector.linear1.weight`
fn remap_mmproj_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("v.enc.blk.") {
        let mapped = rest
            .replace("attn_q.", "q_proj.")
            .replace("attn_k.", "k_proj.")
            .replace("attn_v.", "v_proj.")
            .replace("attn_out.", "out_proj.")
            .replace("ffn_down.", "fc1.")
            .replace("ffn_up.", "fc2.");
        format!("vision_model.encoder.layers.{mapped}")
    } else if let Some(rest) = name.strip_prefix("v.") {
        let mapped = rest
            .replace("patch_embd", "patch_embedding")
            .replace("position_embd", "position_embedding")
            .replace("class_embd", "class_embedding")
            .replace("pre_ln", "ln_pre")
            .replace("post_ln", "ln_post");
        format!("vision_model.{mapped}")
    } else if let Some(rest) = name.strip_prefix("mm.") {
        let mapped = rest.replace("0.", "linear1.").replace("2.", "linear2.");
        format!("vision_model.projector.{mapped}")
    } else {
        format!("vision_model.{name}")
    }
}

/// Try to detect vision config from mmproj GGUF metadata.
fn detect_vision_config_from_mmproj(gguf: Option<&Gguf>) -> Option<boostr::model::VisionConfig> {
    let gguf = gguf?;
    let meta = gguf.metadata();

    let image_size = meta.get_u32("clip.vision.image_size")? as usize;
    let patch_size = meta.get_u32("clip.vision.patch_size")? as usize;
    let hidden_size = meta.get_u32("clip.vision.embedding_length")? as usize;
    let num_layers = meta.get_u32("clip.vision.block_count")? as usize;
    let num_heads = meta.get_u32("clip.vision.head_count")? as usize;
    let intermediate_size = meta
        .get_u32("clip.vision.feed_forward_length")
        .unwrap_or(hidden_size as u32 * 4) as usize;

    Some(boostr::model::VisionConfig {
        encoder_type: "clip".to_string(),
        hidden_size,
        num_layers,
        num_heads,
        patch_size,
        image_size,
        intermediate_size,
        projector_type: "mlp".to_string(),
        projector_depth: 2,
        select_layer: Some(-2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_mmproj_name_vision_encoder() {
        assert_eq!(
            remap_mmproj_name("v.enc.blk.0.attn_q.weight"),
            "vision_model.encoder.layers.0.q_proj.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.11.attn_k.weight"),
            "vision_model.encoder.layers.11.k_proj.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.3.attn_v.weight"),
            "vision_model.encoder.layers.3.v_proj.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.5.attn_out.weight"),
            "vision_model.encoder.layers.5.out_proj.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.11.ffn_down.weight"),
            "vision_model.encoder.layers.11.fc1.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.2.ffn_up.weight"),
            "vision_model.encoder.layers.2.fc2.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.0.ln1.weight"),
            "vision_model.encoder.layers.0.ln1.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.enc.blk.0.ln2.weight"),
            "vision_model.encoder.layers.0.ln2.weight"
        );
    }

    #[test]
    fn test_remap_mmproj_name_vision_global() {
        assert_eq!(
            remap_mmproj_name("v.patch_embd.weight"),
            "vision_model.patch_embedding.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.position_embd.weight"),
            "vision_model.position_embedding.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.class_embd"),
            "vision_model.class_embedding"
        );
        assert_eq!(
            remap_mmproj_name("v.pre_ln.weight"),
            "vision_model.ln_pre.weight"
        );
        assert_eq!(
            remap_mmproj_name("v.post_ln.weight"),
            "vision_model.ln_post.weight"
        );
    }

    #[test]
    fn test_remap_mmproj_name_projector() {
        assert_eq!(
            remap_mmproj_name("mm.0.weight"),
            "vision_model.projector.linear1.weight"
        );
        assert_eq!(
            remap_mmproj_name("mm.2.weight"),
            "vision_model.projector.linear2.weight"
        );
        assert_eq!(
            remap_mmproj_name("mm.0.bias"),
            "vision_model.projector.linear1.bias"
        );
    }

    #[test]
    fn test_remap_mmproj_name_unknown() {
        assert_eq!(
            remap_mmproj_name("some.other.tensor"),
            "vision_model.some.other.tensor"
        );
    }
}

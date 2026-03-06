//! LoRA adapter hot-loading and registry.
//!
//! Supports loading HuggingFace PEFT-format adapter files (`adapter_model.safetensors`)
//! and managing multiple named adapters per model instance.
//!
//! # HuggingFace PEFT tensor naming convention
//!
//! ```text
//! base_model.model.layers.0.self_attn.q_proj.lora_A.weight  →  [rank, in_features]
//! base_model.model.layers.0.self_attn.q_proj.lora_B.weight  →  [out_features, rank]
//! ```
//!
//! The layer key stripped of the PEFT prefix and `.lora_{A,B}.weight` suffix is used as
//! the lookup key when applying the adapter.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::{DType, Runtime, Tensor};

// ── LoraAdapter ──────────────────────────────────────────────────────────────

/// A single loaded LoRA adapter.
///
/// Stores per-layer A and B weight matrices plus the scaling alpha and rank.
/// The `weights` map is keyed by the layer path as it appears in the base model
/// (e.g. `"model.layers.0.self_attn.q_proj"`).
pub struct LoraAdapter<R: Runtime> {
    /// Human-readable adapter name (provided by the caller)
    pub name: String,
    /// LoRA rank (dimension of the low-rank decomposition)
    pub rank: usize,
    /// Alpha scaling factor. The effective scale applied to LoRA output is `alpha / rank`.
    pub alpha: f32,
    /// Per-layer weights: layer_key → (lora_A [rank, in], lora_B [out, rank])
    pub weights: HashMap<String, (Tensor<R>, Tensor<R>)>,
}

impl<R: Runtime> LoraAdapter<R> {
    /// Effective scaling factor `alpha / rank`.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Look up the A and B tensors for a given layer key.
    ///
    /// Returns `None` if this adapter has no weight for the requested layer.
    pub fn get_layer(&self, layer_key: &str) -> Option<(&Tensor<R>, &Tensor<R>)> {
        self.weights.get(layer_key).map(|(a, b)| (a, b))
    }
}

// ── LoraAdapterRegistry ───────────────────────────────────────────────────────

/// Thread-safe registry of named LoRA adapters.
///
/// Held as `Arc<LoraAdapterRegistry<R>>` inside the `Executor` so that adapters
/// can be loaded and unloaded concurrently with ongoing inference.
pub struct LoraAdapterRegistry<R: Runtime> {
    adapters: RwLock<HashMap<String, Arc<LoraAdapter<R>>>>,
}

impl<R: Runtime> LoraAdapterRegistry<R> {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a loaded adapter.
    ///
    /// If an adapter with the same name already exists it is replaced.
    pub fn insert(&self, adapter: LoraAdapter<R>) {
        let name = adapter.name.clone();
        let mut guard = self
            .adapters
            .write()
            .expect("LoraAdapterRegistry lock poisoned");
        guard.insert(name, Arc::new(adapter));
    }

    /// Remove an adapter by name. Returns `true` if the adapter existed.
    pub fn remove(&self, name: &str) -> bool {
        let mut guard = self
            .adapters
            .write()
            .expect("LoraAdapterRegistry lock poisoned");
        guard.remove(name).is_some()
    }

    /// Retrieve an adapter by name.
    pub fn get(&self, name: &str) -> Option<Arc<LoraAdapter<R>>> {
        let guard = self
            .adapters
            .read()
            .expect("LoraAdapterRegistry lock poisoned");
        guard.get(name).cloned()
    }

    /// List names of all currently loaded adapters.
    pub fn list(&self) -> Vec<String> {
        let guard = self
            .adapters
            .read()
            .expect("LoraAdapterRegistry lock poisoned");
        guard.keys().cloned().collect()
    }
}

impl<R: Runtime> Default for LoraAdapterRegistry<R> {
    fn default() -> Self {
        Self::new()
    }
}

// ── load_lora_adapter ─────────────────────────────────────────────────────────

/// Load a HuggingFace PEFT LoRA adapter from a directory or single safetensors file.
///
/// Expects the standard PEFT layout:
/// ```text
/// <path>/
///   adapter_model.safetensors      (tensor weights)
///   adapter_config.json            (optional — used to read `lora_alpha` and `r`)
/// ```
///
/// If `adapter_config.json` is absent or unparseable, `alpha` defaults to `rank` (scaling = 1.0).
///
/// # Arguments
/// * `path`   – Path to the adapter directory or directly to `adapter_model.safetensors`.
/// * `name`   – Logical name under which the adapter is registered.
/// * `device` – Device on which weight tensors are allocated.
pub fn load_lora_adapter<R: Runtime<DType = DType>>(
    path: &Path,
    name: &str,
    device: &R::Device,
) -> Result<LoraAdapter<R>> {
    // Resolve the safetensors file path
    let st_path = if path.is_file() {
        path.to_path_buf()
    } else {
        let candidate = path.join("adapter_model.safetensors");
        if candidate.exists() {
            candidate
        } else {
            return Err(anyhow!(
                "No adapter_model.safetensors found in {}",
                path.display()
            ));
        }
    };

    // Read optional adapter_config.json for rank and alpha
    let config_dir = if path.is_file() {
        path.parent().unwrap_or(Path::new("."))
    } else {
        path
    };
    let (config_rank, config_alpha) = read_adapter_config(config_dir);

    // Open the safetensors file via boostr's loader
    let mut loader = SafeTensorsLoader::open(&st_path).map_err(|e| {
        anyhow!(
            "Failed to open adapter safetensors '{}': {}",
            st_path.display(),
            e
        )
    })?;

    let tensor_names: Vec<String> = loader.tensor_names();

    // Partition tensor names into lora_A and lora_B groups.
    // PEFT key format: `base_model.model.<layer_path>.lora_A.weight`
    let mut a_keys: HashMap<String, String> = HashMap::new(); // layer_key → tensor_name
    let mut b_keys: HashMap<String, String> = HashMap::new();

    for tensor_name in &tensor_names {
        if let Some(layer_key) = strip_lora_suffix(tensor_name, "lora_A") {
            a_keys.insert(layer_key, tensor_name.clone());
        } else if let Some(layer_key) = strip_lora_suffix(tensor_name, "lora_B") {
            b_keys.insert(layer_key, tensor_name.clone());
        }
    }

    if a_keys.is_empty() {
        return Err(anyhow!(
            "No lora_A weights found in '{}'. Expected tensor names like \
             'base_model.model.<layer>.lora_A.weight'.",
            st_path.display()
        ));
    }

    // Load matched A+B pairs
    let mut weights: HashMap<String, (Tensor<R>, Tensor<R>)> = HashMap::new();
    let mut detected_rank: Option<usize> = None;

    for (layer_key, a_name) in &a_keys {
        let b_name = match b_keys.get(layer_key) {
            Some(n) => n,
            None => {
                tracing::warn!(
                    adapter = name,
                    layer = %layer_key,
                    "lora_A found but no matching lora_B — skipping layer"
                );
                continue;
            }
        };

        let tensor_a = loader
            .load_tensor::<R>(a_name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", a_name, e))?;
        let tensor_b = loader
            .load_tensor::<R>(b_name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", b_name, e))?;

        // Infer rank from lora_A shape: [rank, in_features]
        if detected_rank.is_none() && tensor_a.shape().len() == 2 {
            detected_rank = Some(tensor_a.shape()[0]);
        }

        weights.insert(layer_key.clone(), (tensor_a, tensor_b));
    }

    if weights.is_empty() {
        return Err(anyhow!(
            "No complete lora_A + lora_B pairs found in '{}'",
            st_path.display()
        ));
    }

    let rank = config_rank
        .or(detected_rank)
        .ok_or_else(|| anyhow!("Could not determine LoRA rank from adapter file"))?;

    let alpha = config_alpha.unwrap_or(rank as f32);

    tracing::info!(
        adapter = name,
        rank = rank,
        alpha = alpha,
        layers = weights.len(),
        "LoRA adapter loaded"
    );

    Ok(LoraAdapter {
        name: name.to_string(),
        rank,
        alpha,
        weights,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Strip the PEFT prefix (`base_model.model.`) and the `.<part>.weight` suffix from a tensor name.
///
/// Returns `Some(layer_key)` if the name ends with `.<lora_part>.weight`, `None` otherwise.
///
/// Example:
/// ```text
/// "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
///   → Some("model.layers.0.self_attn.q_proj")
/// ```
fn strip_lora_suffix(tensor_name: &str, lora_part: &str) -> Option<String> {
    let suffix = format!(".{}.weight", lora_part);
    let stripped = tensor_name.strip_suffix(&suffix)?;

    // Remove the "base_model." prefix if present
    let layer_key = stripped
        .strip_prefix("base_model.")
        .unwrap_or(stripped)
        .to_string();

    Some(layer_key)
}

/// Attempt to parse `adapter_config.json` for `r` (rank) and `lora_alpha`.
/// Returns `(rank, alpha)` as `Option`s — `None` if absent or unparseable.
fn read_adapter_config(dir: &Path) -> (Option<usize>, Option<f32>) {
    let config_path = dir.join("adapter_config.json");
    if !config_path.exists() {
        return (None, None);
    }

    let Ok(content) = std::fs::read_to_string(&config_path) else {
        return (None, None);
    };

    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        return (None, None);
    };

    let rank = json.get("r").and_then(|v| v.as_u64()).map(|v| v as usize);
    let alpha = json
        .get("lora_alpha")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    (rank, alpha)
}

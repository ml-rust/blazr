//! SafeTensors model loading
//!
//! Uses boostr's SafeTensorsLoader for both detection and tensor loading.
//! Supports both single-file and sharded (multi-file) models transparently.
//!
//! # Device Mapping
//!
//! For large models that don't fit in GPU memory, use `load_safetensors_with_offloading`
//! which automatically calculates how many layers fit in VRAM and offloads the rest to CPU.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
#[cfg(feature = "cuda")]
use boostr::format::{DevicePlacement, LayerDeviceMap};
use boostr::model::{HuggingFaceConfig, LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::VarBuilder;
use boostr::VarMap;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;
use crate::model::detect::DetectedConfig;

/// VRAM reserved for KV cache and working memory (2GB default)
const DEFAULT_KV_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// Load a model from SafeTensors format (no offloading)
///
/// This function loads the entire model to GPU. If the model doesn't fit,
/// it will fail with an OOM error. For large models, use load_safetensors_with_offloading.
///
/// This function:
/// 1. Opens the SafeTensors file(s) - supports both single and sharded
/// 2. Detects the model architecture from tensor names
/// 3. Loads the configuration (from config.json if available)
/// 4. Creates a VarBuilder and loads the model
pub fn load_safetensors<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>,
{
    load_safetensors_regular(path, device)
}

/// Load a regular (non-AWQ) SafeTensors model
fn load_safetensors_regular<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();

    // Determine config directory
    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    // Use boostr's unified SafeTensorsLoader (handles single/sharded transparently)
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    // Log loading info
    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::info!(
            "Loading single-file model, size: {:.2} GB",
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Detect architecture from tensor names and shapes
    let detected = detect_architecture_from_loader(&loader)?;

    // Try to load config.json if available
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected)?
    } else {
        config_from_detected(&detected)
    };

    // Load all tensors into a VarMap
    let var_map = load_tensors_from_loader::<R>(&mut loader, device)?;

    // Leak the VarMap to get a 'static reference (safe: model owns the weights for lifetime of program)
    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, device);

    // Load model using boostr's LoadedModel
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Load options for GPU+CPU offloading
#[derive(Debug, Clone)]
pub struct OffloadingOptions {
    /// Available VRAM in bytes (auto-detect if None)
    pub vram_limit: Option<u64>,
    /// Reserve this much VRAM for KV cache and working memory
    pub kv_cache_reserve: u64,
    /// Force specific number of layers on GPU (auto-calculate if None)
    pub gpu_layers: Option<usize>,
}

impl Default for OffloadingOptions {
    fn default() -> Self {
        Self {
            vram_limit: None,
            kv_cache_reserve: DEFAULT_KV_CACHE_RESERVE,
            gpu_layers: None,
        }
    }
}

impl OffloadingOptions {
    /// Create with specific VRAM limit
    pub fn with_vram_limit(vram_bytes: u64) -> Self {
        Self {
            vram_limit: Some(vram_bytes),
            ..Default::default()
        }
    }

    /// Set number of layers to keep on GPU
    pub fn gpu_layers(mut self, layers: usize) -> Self {
        self.gpu_layers = Some(layers);
        self
    }
}

/// Load a model with GPU+CPU offloading
///
/// Automatically calculates how many layers fit in GPU memory and loads
/// the rest to CPU first, then transfers to GPU layer-by-layer.
///
/// # Strategy
///
/// For models larger than VRAM, this function:
/// 1. Loads "primary" tensors (early layers, embeddings) directly to GPU
/// 2. Loads "secondary" tensors (later layers) to CPU first
/// 3. Transfers CPU tensors to GPU one-by-one (streaming)
///
/// This allows loading models larger than VRAM by using CPU as a staging area.
/// The model still runs entirely on GPU; this just reduces peak memory during loading.
///
/// # Arguments
/// * `path` - Path to the model directory or safetensors file
/// * `gpu_device` - Primary GPU device
/// * `options` - Offloading configuration
///
/// # Returns
/// Returns the loaded model along with info about layer placement.
#[cfg(feature = "cuda")]
pub fn load_safetensors_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    gpu_device: &R::Device,
    options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R>,
{
    use boostr::{CpuDevice, CpuRuntime};

    let path = path.as_ref();

    // Determine config directory
    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    // Open the model
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let model_bytes = loader.total_size();

    // Detect architecture
    let detected = detect_architecture_from_loader(&loader)?;
    let total_layers = detected.num_layers;

    // Load config
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected)?
    } else {
        config_from_detected(&detected)
    };

    // Get VRAM limit
    let vram_limit = options
        .vram_limit
        .unwrap_or_else(|| get_available_vram::<R>(gpu_device).unwrap_or(10 * 1024 * 1024 * 1024));

    // Calculate how many layers fit
    let gpu_layers = options.gpu_layers.unwrap_or_else(|| {
        // Estimate layers that fit: usable_vram / bytes_per_layer
        let usable_vram = vram_limit.saturating_sub(options.kv_cache_reserve);
        let bytes_per_layer = if total_layers > 0 {
            model_bytes / total_layers as u64
        } else {
            model_bytes
        };
        if bytes_per_layer == 0 {
            total_layers
        } else {
            ((usable_vram / bytes_per_layer) as usize).min(total_layers)
        }
    });

    // Create device map
    let device_map = LayerDeviceMap::with_gpu_layers(total_layers, gpu_layers);
    let layer_prefix = detect_layer_prefix(&detected);

    // Check if model will fit on GPU at all
    // Even with staging, the entire model must fit on GPU for inference
    let model_gb = model_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let vram_gb = vram_limit as f64 / (1024.0 * 1024.0 * 1024.0);

    if model_bytes > vram_limit {
        return Err(anyhow!(
            "Model ({:.2} GB) exceeds available GPU memory ({:.2} GB).\n\n\
            CPU staging can reduce peak memory during loading, but the model\n\
            still needs to fit entirely on GPU for inference.\n\n\
            Options:\n\
            1. Use a quantized version (GGUF Q4_K_M) which uses ~3-4x less memory\n\
            2. Use a smaller model (e.g., Mistral-7B needs ~14GB, try 3B models)\n\
            3. Use a GPU with more VRAM\n\
            4. Wait for true layer-by-layer CPU offloading in a future version",
            model_gb,
            vram_gb
        ));
    }

    tracing::info!(
        "Loading model with {} layers direct to GPU, {} via CPU staging (VRAM: {:.2} GB, model: {:.2} GB)",
        gpu_layers,
        total_layers.saturating_sub(gpu_layers),
        vram_gb,
        model_gb
    );

    let mut var_map = VarMap::<R>::new();
    let cpu_dev = CpuDevice::new();

    let tensor_names = loader.tensor_names();
    let mut gpu_direct_count = 0;
    let mut cpu_staged_count = 0;
    let mut gpu_bytes = 0u64;
    let mut staged_bytes = 0u64;

    // First pass: collect tensors by placement
    // Determine placement by extracting the layer index from the tensor name.
    // Tensors without a layer index (embeddings, lm_head) go directly to GPU.
    let mut primary_tensors = Vec::new();
    let mut secondary_tensors = Vec::new();

    for name in tensor_names {
        let layer_idx = extract_layer_index(&name, &layer_prefix);
        let placement = match layer_idx {
            Some(idx) => device_map.placement(idx),
            None => DevicePlacement::Gpu, // embeddings/lm_head go to GPU
        };
        match placement {
            DevicePlacement::Gpu => primary_tensors.push(name),
            DevicePlacement::Cpu => secondary_tensors.push(name),
        }
    }

    // Load primary tensors directly to GPU
    tracing::info!(
        "Loading {} tensors directly to GPU...",
        primary_tensors.len()
    );
    for (idx, name) in primary_tensors.iter().enumerate() {
        if idx % 50 == 0 {
            tracing::debug!(
                "Loading GPU tensor {}/{}: {}",
                idx + 1,
                primary_tensors.len(),
                name
            );
        }

        let info = loader
            .tensor_info(name)
            .map_err(|e| anyhow!("Failed to get tensor info for '{}': {}", name, e))?;
        let tensor_bytes = info.size_bytes() as u64;

        let tensor = loader
            .load_tensor::<R>(name, gpu_device)
            .map_err(|e| anyhow!("Failed to load tensor '{}' to GPU: {}", name, e))?;
        var_map.insert(name.clone(), tensor);
        gpu_direct_count += 1;
        gpu_bytes += tensor_bytes;
    }

    // Load secondary tensors via CPU staging
    if !secondary_tensors.is_empty() {
        tracing::info!(
            "Staging {} tensors via CPU (this may take a while)...",
            secondary_tensors.len()
        );

        for (idx, name) in secondary_tensors.iter().enumerate() {
            if idx % 20 == 0 {
                tracing::debug!(
                    "Staging tensor {}/{}: {}",
                    idx + 1,
                    secondary_tensors.len(),
                    name
                );
            }

            let info = loader
                .tensor_info(name)
                .map_err(|e| anyhow!("Failed to get tensor info for '{}': {}", name, e))?;
            let tensor_bytes = info.size_bytes() as u64;

            // Load to CPU
            let cpu_tensor = loader
                .load_tensor::<CpuRuntime>(name, &cpu_dev)
                .map_err(|e| anyhow!("Failed to load tensor '{}' to CPU: {}", name, e))?;

            // Transfer CPU tensor to GPU
            // We need to get the raw bytes and create a new GPU tensor
            let bytes = cpu_tensor
                .to_bytes()
                .map_err(|e| anyhow!("Failed to get bytes from CPU tensor '{}': {}", name, e))?;
            let shape = cpu_tensor.shape().to_vec();
            let dtype = cpu_tensor.dtype();

            // Create GPU tensor from bytes
            let gpu_tensor =
                create_gpu_tensor_from_bytes::<R>(&bytes, &shape, dtype, gpu_device)
                    .map_err(|e| anyhow!("Failed to transfer tensor '{}' to GPU: {}", name, e))?;

            // Drop CPU tensor to free memory
            drop(cpu_tensor);

            var_map.insert(name.clone(), gpu_tensor);
            cpu_staged_count += 1;
            staged_bytes += tensor_bytes;
        }
    }

    tracing::info!(
        "Loaded {} tensors: {} direct ({:.2} GB), {} staged ({:.2} GB)",
        gpu_direct_count + cpu_staged_count,
        gpu_direct_count,
        gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        cpu_staged_count,
        staged_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Create VarBuilder by leaking the VarMap to get a 'static reference
    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, gpu_device);

    // Load model
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    let offloading_info = OffloadingInfo {
        gpu_layers,
        cpu_layers: total_layers.saturating_sub(gpu_layers),
        gpu_bytes: gpu_bytes + staged_bytes,
        cpu_bytes: 0, // All tensors end up on GPU
        total_layers,
    };

    Ok((model, config, offloading_info))
}

/// Non-CUDA fallback for offloading (just loads normally)
#[cfg(not(feature = "cuda"))]
pub fn load_safetensors_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    _options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R>,
{
    let (model, config) = load_safetensors(path, device)?;
    Ok((
        model,
        config,
        OffloadingInfo {
            gpu_layers: 0,
            cpu_layers: 0,
            gpu_bytes: 0,
            cpu_bytes: 0,
            total_layers: 0,
        },
    ))
}

/// Create a GPU tensor from raw bytes
#[cfg(feature = "cuda")]
fn create_gpu_tensor_from_bytes<R: Runtime<DType = DType>>(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Result<boostr::tensor::Tensor<R>>
where
    R::Client: TensorOps<R>,
{
    use boostr::tensor::Tensor;

    match dtype {
        DType::F32 => {
            let data: &[f32] = bytemuck::cast_slice(bytes);
            Ok(Tensor::from_slice(data, shape, device))
        }
        DType::BF16 => {
            let data: &[half::bf16] = bytemuck::cast_slice(bytes);
            Ok(Tensor::from_slice(data, shape, device))
        }
        DType::F16 => {
            let data: &[half::f16] = bytemuck::cast_slice(bytes);
            Ok(Tensor::from_slice(data, shape, device))
        }
        _ => Err(anyhow!("Unsupported dtype for GPU transfer: {:?}", dtype)),
    }
}

/// Information about layer offloading
#[derive(Debug, Clone)]
pub struct OffloadingInfo {
    /// Number of layers on GPU
    pub gpu_layers: usize,
    /// Number of layers on CPU
    pub cpu_layers: usize,
    /// Total bytes on GPU
    pub gpu_bytes: u64,
    /// Total bytes on CPU
    pub cpu_bytes: u64,
    /// Total number of layers
    pub total_layers: usize,
}

/// Try to get available VRAM
///
/// For CUDA devices, queries actual GPU memory via cuMemGetInfo.
/// For other runtimes, returns None (will use default fallback).
#[cfg(feature = "cuda")]
fn get_available_vram<R: Runtime>(device: &R::Device) -> Option<u64> {
    // Check if this is a CUDA runtime by checking the runtime name
    if R::name() == "CUDA" {
        // We need to downcast to CudaDevice to call memory_info()
        // Since we're compiled with CUDA feature, we can use the CUDA-specific function
        use boostr::runtime::Device;
        use boostr::CudaDevice;

        // Get device ID from the device
        let device_id = device.id();

        // Create a CudaDevice and query memory
        let cuda_device = CudaDevice::new(device_id);
        match cuda_device.memory_info() {
            Ok((free, _total)) => {
                tracing::debug!(
                    "Detected GPU memory: {:.2} GB free",
                    free as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                Some(free)
            }
            Err(e) => {
                tracing::warn!("Failed to query GPU memory: {}", e);
                None
            }
        }
    } else {
        None
    }
}

/// Extract the layer index from a tensor name given the layer prefix.
///
/// For example, with prefix "model.layers.", the tensor name
/// "model.layers.3.self_attn.q_proj.weight" returns Some(3).
/// Returns None for tensors that are not part of a numbered layer
/// (e.g. embeddings, lm_head, norm).
#[cfg(feature = "cuda")]
fn extract_layer_index(tensor_name: &str, layer_prefix: &str) -> Option<usize> {
    let rest = tensor_name.strip_prefix(layer_prefix)?;
    // rest is now something like "3.self_attn.q_proj.weight"
    let dot_pos = rest.find('.')?;
    let index_str = &rest[..dot_pos];
    index_str.parse::<usize>().ok()
}

/// Detect layer prefix from detected config
#[cfg(feature = "cuda")]
fn detect_layer_prefix(detected: &DetectedConfig) -> String {
    match detected.format {
        crate::model::detect::ModelFormat::HuggingFace => "model.layers.".to_string(),
        crate::model::detect::ModelFormat::Oxidizr => "layers.".to_string(),
    }
}

/// Detect architecture from SafeTensorsLoader
fn detect_architecture_from_loader(loader: &SafeTensorsLoader) -> Result<DetectedConfig> {
    // Get tensor names and use existing detection
    let tensor_names = loader.tensor_names();

    // Use the names-only detection for now (gets format and layer types)
    let mut config = crate::model::detect::detect_architecture_from_names(&tensor_names)?;

    // Now fill in the size-dependent parameters from tensor info
    let format = config.format;
    let prefix = match format {
        crate::model::detect::ModelFormat::HuggingFace => "model.",
        crate::model::detect::ModelFormat::Oxidizr => "",
    };

    // Detect hidden_size and vocab_size from embedding layer
    let embed_key = format!("{}embed_tokens.weight", prefix);
    if let Ok(info) = loader.tensor_info(&embed_key) {
        if info.shape.len() == 2 {
            config.vocab_size = info.shape[0];
            config.hidden_size = info.shape[1];
        }
    }

    // Detect intermediate_size from MLP if present
    let mlp_gate_key = format!("{}layers.0.mlp.gate_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&mlp_gate_key) {
        if info.shape.len() == 2 {
            config.intermediate_size = Some(info.shape[0]);
        }
    }

    // Detect number of attention heads from q_proj if present
    let q_proj_key = format!("{}layers.0.self_attn.q_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&q_proj_key) {
        if info.shape.len() == 2 && config.hidden_size > 0 {
            let head_dim = 128; // Default head dimension for most models
            config.num_attention_heads = Some(info.shape[0] / head_dim);
            config.head_dim = Some(head_dim);
        }
    }

    // Detect KV heads from k_proj
    let k_proj_key = format!("{}layers.0.self_attn.k_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&k_proj_key) {
        if info.shape.len() == 2 {
            let head_dim = config.head_dim.unwrap_or(128);
            config.num_kv_heads = Some(info.shape[0] / head_dim);
        }
    }

    Ok(config)
}

/// Load all tensors from a SafeTensorsLoader into a VarMap
fn load_tensors_from_loader<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    device: &R::Device,
) -> Result<VarMap<R>>
where
    R::Client: TensorOps<R>,
{
    let mut var_map = VarMap::<R>::new();

    // Get tensor names
    let tensor_names = loader.tensor_names();
    let total = tensor_names.len();

    tracing::info!("Loading {} tensors to device...", total);

    // Load all tensors
    for (idx, name) in tensor_names.into_iter().enumerate() {
        if idx % 50 == 0 {
            tracing::debug!("Loading tensor {}/{}: {}", idx + 1, total, name);
        }

        let tensor = loader
            .load_tensor::<R>(&name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))?;
        var_map.insert(name, tensor);
    }

    tracing::info!("Loaded {} tensors successfully", var_map.len());

    Ok(var_map)
}

/// Load a model from SafeTensors with explicit configuration
pub fn load_safetensors_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: &UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();

    // Use boostr's unified SafeTensorsLoader (handles single/sharded transparently)
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    // Log loading info
    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Load all tensors into a VarMap
    let var_map = load_tensors_from_loader::<R>(&mut loader, device)?;

    // Leak the VarMap to get a 'static reference (safe: model owns the weights for lifetime of program)
    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, device);

    // Load model using boostr's LoadedModel
    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}

/// Detect dtype from config.json's torch_dtype field
fn detect_dtype_from_config(dir: &Path) -> &'static str {
    let config_path = dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            match json.get("torch_dtype").and_then(|v| v.as_str()) {
                Some("bfloat16") => return "bf16",
                Some("float16") => return "f16",
                Some("float32") => return "f32",
                _ => {}
            }
        }
    }
    "f16" // Default
}

/// Load config or create from detected architecture
fn load_or_create_config(dir: &Path, detected: &DetectedConfig) -> Result<BlazrConfig> {
    let config_path = dir.join("config.json");
    let dtype = detect_dtype_from_config(dir);

    if config_path.exists() {
        // Try to load existing config
        match BlazrConfig::from_json(&config_path) {
            Ok(mut config) => {
                // Override dtype from torch_dtype if detected
                config.inference.dtype = dtype.to_string();
                return Ok(config);
            }
            Err(_) => {
                // If BlazrConfig fails, try loading JSON and parsing as UniversalConfig
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    if let Ok(model_config) = serde_json::from_str::<UniversalConfig>(&content) {
                        return Ok(BlazrConfig::from_universal_with_dtype(model_config, dtype));
                    }
                    // Try parsing as HuggingFace config format (using boostr's parser)
                    if let Ok(hf_config) = HuggingFaceConfig::from_json(&content) {
                        return Ok(BlazrConfig::from_universal_with_dtype(
                            hf_config.to_universal(),
                            dtype,
                        ));
                    }
                }
                // Fall through to create from detected
            }
        }
    }

    // Create config from detected architecture
    Ok(config_from_detected_with_dtype(detected, dtype))
}

/// Create BlazrConfig from detected architecture
fn config_from_detected(detected: &DetectedConfig) -> BlazrConfig {
    config_from_detected_with_dtype(detected, "f16")
}

/// Create BlazrConfig from detected architecture with specified dtype
fn config_from_detected_with_dtype(detected: &DetectedConfig, dtype: &str) -> BlazrConfig {
    use crate::model::Config;

    // Use the legacy Config::from_detected then convert to BlazrConfig
    let legacy_config = Config::from_detected(detected.clone());

    // Convert to UniversalConfig
    let model = detected_to_universal(detected, &legacy_config);

    BlazrConfig::from_universal_with_dtype(model, dtype)
}

/// Convert DetectedConfig + legacy Config to UniversalConfig
fn detected_to_universal(
    detected: &DetectedConfig,
    legacy: &crate::model::Config,
) -> UniversalConfig {
    use crate::model::detect::LayerType;
    use boostr::model::{AttentionConfig, HybridConfig, MoeConfig, SsmConfig};

    // Determine model type
    let model_type = determine_model_type(&detected.layer_types);

    // Convert blazr's RopeScalingConfig to boostr's RopeScalingConfig
    let rope_scaling = legacy
        .rope_scaling
        .as_ref()
        .map(|rs| boostr::model::RopeScalingConfig {
            scaling_type: rs.rope_type.clone().unwrap_or_else(|| "llama3".to_string()),
            factor: rs.factor as f32,
            original_max_position_embeddings: Some(rs.original_max_position_embeddings),
            low_freq_factor: Some(rs.low_freq_factor as f32),
            high_freq_factor: Some(rs.high_freq_factor as f32),
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
        });

    // Build attention config if applicable
    let attention = if detected.num_attention_heads.is_some() || detected.num_kv_heads.is_some() {
        Some(AttentionConfig {
            num_heads: detected.num_attention_heads.unwrap_or(32),
            num_kv_heads: detected.num_kv_heads,
            head_dim: detected.head_dim,
            rope_theta: legacy.rope_theta as f32,
            rope_scaling,
            kv_latent_dim: detected.kv_latent_dim,
            q_latent_dim: detected.q_latent_dim,
            d_rope: detected.d_rope,
            sliding_window: None,
        })
    } else {
        None
    };

    // Build SSM config if applicable
    let ssm = if detected.mamba2_num_heads.is_some() {
        Some(SsmConfig {
            variant: if detected.mamba3_enabled.unwrap_or(false) {
                "mamba3"
            } else {
                "mamba2"
            }
            .to_string(),
            num_heads: detected.mamba2_num_heads.unwrap_or(64),
            head_dim: detected.mamba2_head_dim.unwrap_or(64),
            state_size: detected.mamba2_state_size.unwrap_or(64),
            chunk_size: legacy.mamba2_chunk_size,
            n_groups: legacy.mamba2_n_groups,
            conv_kernel: detected.mamba2_conv_kernel.unwrap_or(4),
            expand: detected.mamba2_expand.unwrap_or(2),
            complex_rope: detected.mamba3_complex_rope,
            mimo_rank: detected.mamba3_mimo_rank,
            use_conv: detected.mamba3_use_conv,
        })
    } else {
        None
    };

    // Build MoE config if applicable
    let moe = if detected.num_experts.is_some() {
        Some(MoeConfig {
            num_experts: detected.num_experts.unwrap_or(8),
            experts_per_tok: legacy.experts_per_tok,
            shared_expert: Some(detected.shared_expert_enabled),
            intermediate_size: detected.intermediate_size,
            load_balance_alpha: 0.01,
            z_loss_alpha: 1e-3,
        })
    } else {
        None
    };

    // Build hybrid layer config if mixed types
    let hybrid_layers = if has_mixed_layer_types(&detected.layer_types) {
        let ssm_layers: Vec<usize> = detected
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, lt)| matches!(lt, LayerType::Mamba2 | LayerType::Mamba3))
            .map(|(i, _)| i)
            .collect();

        let attention_layers: Vec<usize> = detected
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, lt)| !matches!(lt, LayerType::Mamba2 | LayerType::Mamba3))
            .map(|(i, _)| i)
            .collect();

        Some(HybridConfig {
            ssm_layers,
            attention_layers,
        })
    } else {
        None
    };

    UniversalConfig {
        model_type,
        vocab_size: detected.vocab_size,
        hidden_size: detected.hidden_size,
        num_layers: detected.num_layers,
        max_seq_len: legacy.max_seq_len,
        intermediate_size: detected.intermediate_size,
        rms_norm_eps: legacy.rms_norm_eps,
        attention,
        ssm,
        moe,
        hybrid_layers,
        tie_word_embeddings: false,
    }
}

/// Determine model type from layer types
fn determine_model_type(layer_types: &[crate::model::detect::LayerType]) -> String {
    use crate::model::detect::LayerType;

    if layer_types.is_empty() {
        return "llama".to_string();
    }

    let has_mamba = layer_types
        .iter()
        .any(|lt| matches!(lt, LayerType::Mamba2 | LayerType::Mamba3));
    let has_attention = layer_types
        .iter()
        .any(|lt| !matches!(lt, LayerType::Mamba2 | LayerType::Mamba3));

    if has_mamba && has_attention {
        "hybrid".to_string()
    } else if has_mamba {
        if layer_types.iter().any(|lt| matches!(lt, LayerType::Mamba3)) {
            "mamba3".to_string()
        } else {
            "mamba2".to_string()
        }
    } else if layer_types
        .iter()
        .any(|lt| matches!(lt, LayerType::MlaWithMoe | LayerType::MlaWithMlp))
    {
        "hybrid".to_string() // MLA is typically in hybrid models
    } else {
        "llama".to_string() // Default to llama for standard transformer
    }
}

/// Check if layer types are mixed (requires hybrid config)
fn has_mixed_layer_types(layer_types: &[crate::model::detect::LayerType]) -> bool {
    use crate::model::detect::LayerType;

    if layer_types.is_empty() {
        return false;
    }

    let first_is_mamba = matches!(layer_types[0], LayerType::Mamba2 | LayerType::Mamba3);

    layer_types.iter().any(|lt| {
        let is_mamba = matches!(lt, LayerType::Mamba2 | LayerType::Mamba3);
        is_mamba != first_is_mamba
    })
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_huggingface_config_rope_scaling() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 128256,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        }"#;

        let hf_config = HuggingFaceConfig::from_json(json).expect("Failed to parse config");
        println!("rope_scaling: {:?}", hf_config.rope_scaling);

        let universal = hf_config.to_universal();
        let attn = universal.attention.expect("No attention config");
        println!("rope_scaling in universal: {:?}", attn.rope_scaling);

        assert!(attn.rope_scaling.is_some(), "rope_scaling should be Some");
        let rs = attn.rope_scaling.unwrap();
        assert_eq!(rs.scaling_type, "llama3");
        assert_eq!(rs.factor, 32.0);
        assert_eq!(rs.low_freq_factor, Some(1.0));
        assert_eq!(rs.high_freq_factor, Some(4.0));
    }
}

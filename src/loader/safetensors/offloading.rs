//! GPU+CPU offloading for SafeTensors loading

#[cfg(feature = "cuda")]
use std::path::Path;

#[cfg(feature = "cuda")]
use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use boostr::format::{DevicePlacement, LayerDeviceMap, SafeTensorsLoader};
#[cfg(feature = "cuda")]
use boostr::model::LoadedModel;
#[cfg(feature = "cuda")]
use boostr::ops::TensorOps;
#[cfg(feature = "cuda")]
use boostr::VarBuilder;
#[cfg(feature = "cuda")]
use boostr::VarMap;
#[cfg(feature = "cuda")]
use boostr::{DType, Runtime};

#[cfg(feature = "cuda")]
use crate::config::BlazrConfig;
#[cfg(feature = "cuda")]
use crate::model::detect::DetectedConfig;

#[cfg(feature = "cuda")]
use super::config::{config_from_detected, load_or_create_config};
#[cfg(feature = "cuda")]
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};

/// VRAM reserved for KV cache and working memory (2GB default)
const DEFAULT_KV_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

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

#[cfg(feature = "cuda")]
pub fn load_safetensors_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    gpu_device: &R::Device,
    options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    use boostr::{CpuDevice, CpuRuntime};

    let path = path.as_ref();

    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    // Check if this is a GPTQ model — delegate to GPTQ loader
    if detect_gptq(&loader, config_dir) {
        let (model, config) = super::gptq::load_safetensors_gptq(path, gpu_device)?;
        let info = OffloadingInfo {
            gpu_layers: 0,
            cpu_layers: 0,
            gpu_bytes: 0,
            cpu_bytes: 0,
            total_layers: 0,
        };
        return Ok((model, config, info));
    }

    // Check if this is an AWQ model — delegate to AWQ loader which handles
    // qweight/qzeros/scales triplets and DecomposedQuantTensor construction
    if detect_awq(&loader, config_dir) {
        let (model, config) = super::awq::load_safetensors_awq(path, gpu_device)?;
        let info = OffloadingInfo {
            gpu_layers: 0,
            cpu_layers: 0,
            gpu_bytes: 0,
            cpu_bytes: 0,
            total_layers: 0,
        };
        return Ok((model, config, info));
    }

    let model_bytes = loader.total_size();
    let detected = detect_architecture_from_loader(&loader)?;
    let total_layers = detected.num_layers;

    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected)?
    } else {
        config_from_detected(&detected)
    };

    let vram_limit = options
        .vram_limit
        .unwrap_or_else(|| get_available_vram::<R>(gpu_device).unwrap_or(10 * 1024 * 1024 * 1024));

    let gpu_layers = options.gpu_layers.unwrap_or_else(|| {
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

    let device_map = LayerDeviceMap::with_gpu_layers(total_layers, gpu_layers);
    let layer_prefix = detect_layer_prefix(&detected);

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

    let mut primary_tensors = Vec::new();
    let mut secondary_tensors = Vec::new();

    for name in tensor_names {
        let layer_idx = extract_layer_index(&name, &layer_prefix);
        let placement = match layer_idx {
            Some(idx) => device_map.placement(idx),
            None => DevicePlacement::Gpu,
        };
        match placement {
            DevicePlacement::Gpu => primary_tensors.push(name),
            DevicePlacement::Cpu => secondary_tensors.push(name),
        }
    }

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

            let cpu_tensor = loader
                .load_tensor::<CpuRuntime>(name, &cpu_dev)
                .map_err(|e| anyhow!("Failed to load tensor '{}' to CPU: {}", name, e))?;

            let bytes = cpu_tensor
                .to_bytes()
                .map_err(|e| anyhow!("Failed to get bytes from CPU tensor '{}': {}", name, e))?;
            let shape = cpu_tensor.shape().to_vec();
            let dtype = cpu_tensor.dtype();

            let gpu_tensor =
                create_gpu_tensor_from_bytes::<R>(&bytes, &shape, dtype, gpu_device)
                    .map_err(|e| anyhow!("Failed to transfer tensor '{}' to GPU: {}", name, e))?;

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

    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, gpu_device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    let offloading_info = OffloadingInfo {
        gpu_layers,
        cpu_layers: total_layers.saturating_sub(gpu_layers),
        gpu_bytes: gpu_bytes + staged_bytes,
        cpu_bytes: 0,
        total_layers,
    };

    Ok((model, config, offloading_info))
}

#[cfg(feature = "cuda")]
fn create_gpu_tensor_from_bytes<R: Runtime<DType = DType>>(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Result<boostr::tensor::Tensor<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
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

#[cfg(feature = "cuda")]
fn get_available_vram<R: Runtime>(device: &R::Device) -> Option<u64> {
    if R::name() == "CUDA" {
        use boostr::runtime::Device;
        use boostr::CudaDevice;

        let device_id = device.id();
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

#[cfg(feature = "cuda")]
fn extract_layer_index(tensor_name: &str, layer_prefix: &str) -> Option<usize> {
    let rest = tensor_name.strip_prefix(layer_prefix)?;
    let dot_pos = rest.find('.')?;
    let index_str = &rest[..dot_pos];
    index_str.parse::<usize>().ok()
}

#[cfg(feature = "cuda")]
fn detect_layer_prefix(detected: &DetectedConfig) -> String {
    match detected.format {
        crate::model::detect::ModelFormat::HuggingFace => "model.layers.".to_string(),
        crate::model::detect::ModelFormat::Oxidizr => "layers.".to_string(),
    }
}

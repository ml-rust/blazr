//! A real System One decision model (a Qwen3.5 text checkpoint such as
//! Jebadiah or the Mapika decider) loaded through blazr's own loader, then
//! `LabelTokenHead` through `run_decision` against fixtures the family's
//! own reference code wrote.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the checkpoint's `config.json` and its
//!   SafeTensors files, single or sharded.
//! - `BLAZR_DECISION_GGUF`: a Q8_0 or Q4_K_M `.gguf` build of the same
//!   checkpoint, for the `gguf_` tests. Set one of the two model variables.
//! - `BLAZR_DECISION_FIXTURE_DIR`: the `*.json` fixtures of that model,
//!   schema `boostr.decision.label_token.v2`.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! ```bash
//! BLAZR_DECISION_MODEL_DIR=/path/to/model BLAZR_DECISION_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p blazr --test decision_loader_real_file --no-capture hf_cpu
//! BLAZR_DECISION_GGUF=/path/to/model.gguf BLAZR_DECISION_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p blazr --test decision_loader_real_file --no-capture gguf_cpu
//! ```
//!
//! # Load path
//!
//! A SafeTensors directory loads through `load_safetensors_with_report`,
//! the function `load_model_as` reaches for a regular SafeTensors
//! checkpoint, so the test also reads which tensors the load uploaded. A
//! GGUF loads through `load_gguf_with_tokenizer_as`, the function the
//! server's model scheduler calls.
//!
//! # Memory
//!
//! The CPU SafeTensors test holds every weight in F32: about four bytes per
//! parameter of the text model. The CUDA SafeTensors test keeps two bytes
//! per parameter on the device. The GGUF tests keep block-quantized
//! matrices quantized: memory near the file size.

mod decision_loader;

use std::path::Path;

use blazr::config::{DtypeRequest, DtypeSource};
use blazr::loader::{
    detect_model_source, load_gguf_with_tokenizer_as, load_safetensors_with_report,
    model_torch_dtype, ModelFormat, TensorLoadReport,
};
use boostr::format::hf_qwen35::is_dropped;
use boostr::format::{Gguf, SafeTensorsLoader};
use boostr::model::{LoadedModel, ModelClient, Qwen35Model};
use boostr::{CpuDevice, CpuRuntime, DType, Runtime};

use decision_loader::compare::{check_all, Quant, CPU_F32};
use decision_loader::fixture::{gguf_inputs, inputs};

/// The `qwen35` model inside `loaded`.
fn qwen35<R: Runtime>(loaded: LoadedModel<R>) -> Box<Qwen35Model<R>> {
    match loaded {
        LoadedModel::Qwen35(model) => model,
        other => panic!("blazr loaded {other:?}, not a qwen35 model"),
    }
}

/// The report says every tensor of `dir` was either uploaded or skipped,
/// and the skipped ones are exactly the MTP and vision tensors.
fn check_report(dir: &Path, report: &TensorLoadReport) {
    let names = SafeTensorsLoader::open(dir)
        .unwrap_or_else(|e| panic!("SafeTensorsLoader::open {}: {e}", dir.display()))
        .tensor_names();
    let mut dropped: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| is_dropped(n))
        .collect();
    let mut skipped: Vec<&str> = report.skipped.iter().map(String::as_str).collect();
    dropped.sort_unstable();
    skipped.sort_unstable();
    assert_eq!(skipped, dropped, "skipped tensors");
    let uploaded_dropped: Vec<&String> = report.uploaded.iter().filter(|n| is_dropped(n)).collect();
    assert!(
        uploaded_dropped.is_empty(),
        "MTP or vision tensors were uploaded: {uploaded_dropped:?}"
    );
    assert_eq!(
        report.uploaded.len() + report.skipped.len(),
        names.len(),
        "every tensor is uploaded or skipped"
    );
    println!(
        "{}: {} tensors uploaded, {} MTP and vision tensors left on disk",
        dir.display(),
        report.uploaded.len(),
        report.skipped.len()
    );
}

/// The model of the SafeTensors directory `dir` through blazr's loader at
/// the compute dtype `dtype`, after checking the dtype resolution and what
/// the load read.
fn load_hf<R>(dir: &Path, device: &R::Device, dtype: DType) -> Box<Qwen35Model<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let source = detect_model_source(dir)
        .unwrap_or_else(|e| panic!("detect_model_source {}: {e}", dir.display()));
    assert_eq!(source.format, ModelFormat::SafeTensors);
    // Both decision checkpoints name bfloat16: decider at the top level
    // under `dtype`, Jebadiah only under `text_config.dtype`.
    assert_eq!(
        model_torch_dtype(dir),
        Some(DType::BF16),
        "config.json dtype"
    );

    let request = DtypeRequest {
        dtype,
        source: DtypeSource::Flag,
    };
    let (loaded, config, report) =
        load_safetensors_with_report::<R, _>(dir, device, Some(&request))
            .unwrap_or_else(|e| panic!("load_safetensors_with_report {}: {e:#}", dir.display()));
    let report = report.unwrap_or_else(|| panic!("{} loaded as GPTQ or AWQ", dir.display()));
    assert_eq!(config.model.model_type, "qwen35");
    assert!(config.model.tie_word_embeddings, "the 4B head is tied");
    assert_eq!(
        config.compute_dtype().ok(),
        Some(dtype),
        "config compute dtype"
    );
    assert_eq!(report.compute_dtype, dtype, "report compute dtype");
    check_report(dir, &report);

    let model = qwen35(loaded);
    assert_eq!(model.dtype(), dtype, "model dtype");
    model
}

/// The model of the GGUF `file` through blazr's server load path, and its
/// quant. The head must be tied exactly when the file has no
/// `output.weight`.
fn load_gguf<R>(file: &Path, device: &R::Device) -> (Box<Qwen35Model<R>>, Quant)
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + boostr::quant::DequantOps<R>,
{
    let source = detect_model_source(file)
        .unwrap_or_else(|e| panic!("detect_model_source {}: {e}", file.display()));
    assert_eq!(source.format, ModelFormat::Gguf);
    let has_head = Gguf::open(file)
        .unwrap_or_else(|e| panic!("open {}: {e}", file.display()))
        .tensor_info("output.weight")
        .is_ok();
    let quant = Quant::of(file);

    let (loaded, config, _tokenizer) = load_gguf_with_tokenizer_as::<R, _>(file, device, None)
        .unwrap_or_else(|e| panic!("load_gguf_with_tokenizer_as {}: {e:#}", file.display()));
    assert_eq!(config.model.model_type, "qwen35");
    assert_eq!(config.model.tie_word_embeddings, !has_head, "tied head");
    println!(
        "{}: {quant:?}, lm_head {}",
        file.display(),
        if has_head {
            "own output.weight"
        } else {
            "tied to token_embd.weight"
        }
    );
    let model = qwen35(loaded);
    assert_eq!(model.dtype(), DType::F32, "a GGUF model computes in F32");
    (model, quant)
}

/// The SafeTensors checkpoint on the CPU, every weight F32.
#[test]
fn hf_cpu_f32_matches_reference() {
    let Some((dir, fixtures)) = inputs() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let model = load_hf::<CpuRuntime>(&dir, &device, DType::F32);
    check_all::<CpuRuntime>(&client, &model, &fixtures, CPU_F32);
}

/// The SafeTensors checkpoint on CUDA device 0 in BF16. The 4B models need
/// about 9 GB of free device memory.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn hf_cuda_bf16_matches_reference() {
    use boostr::{CudaDevice, CudaRuntime};
    use decision_loader::compare::CUDA_BF16;
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some((dir, fixtures)) = inputs() else {
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let model = load_hf::<CudaRuntime>(&dir, &device, DType::BF16);
    check_all::<CudaRuntime>(&client, &model, &fixtures, CUDA_BF16);
}

/// The GGUF on the CPU: block-quantized matrices stay quantized.
#[test]
fn gguf_cpu_matches_reference() {
    let Some((file, fixtures)) = gguf_inputs() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let (model, quant) = load_gguf::<CpuRuntime>(&file, &device);
    check_all::<CpuRuntime>(&client, &model, &fixtures, quant.tolerance());
}

/// The GGUF on CUDA device 0, the quantized matrices on the device.
#[cfg(feature = "cuda")]
#[test]
fn gguf_cuda_matches_reference() {
    use boostr::{CudaDevice, CudaRuntime};
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some((file, fixtures)) = gguf_inputs() else {
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let (model, quant) = load_gguf::<CudaRuntime>(&file, &device);
    check_all::<CudaRuntime>(&client, &model, &fixtures, quant.tolerance());
}

//! Test fixtures shared by the engine's unit tests.

use anyhow::Result;

use boostr::model::traits::Model;
use boostr::model::LoadedModel;
use boostr::{CpuDevice, CpuRuntime};

use crate::config::BlazrConfig;
use crate::engine::Executor;

/// Config of a one-layer llama small enough to build in a unit test.
pub(crate) const TINY_LLAMA_YAML: &str =
    "model_type: llama\nvocab_size: 32\nhidden_size: 16\nnum_layers: 1\n\
     max_seq_len: 16\nintermediate_size: 32\nattention:\n  num_heads: 2\n";

/// A started executor over a tiny random llama.
pub(crate) fn tiny_executor() -> Result<Executor<CpuRuntime>> {
    let device = CpuDevice::new();
    let config: boostr::model::UniversalConfig = serde_saphyr::from_str(TINY_LLAMA_YAML)?;
    let model = boostr::model::llama::Llama::<CpuRuntime>::from_config(&config, &device)?;
    let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
    Executor::new(
        LoadedModel::Llama(Box::new(model)),
        BlazrConfig::from_universal_with_dtype(config, "f32"),
        tokenizer,
        device,
        16,
    )
}

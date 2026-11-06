//! Every id `/v1/models` lists from a real model directory reads its config and
//! tokenizer, the check a load runs before any eviction.
//!
//! Reads `$BOOSTR_MODELS_DIR`, and skips when it is unset.

use std::path::PathBuf;

use blazr::engine::Scheduler;
use blazr::loader::load_model_meta;
use boostr::{CpuDevice, CpuRuntime};

#[test]
fn every_listed_id_passes_the_load_check() {
    let Ok(dir) = std::env::var("BOOSTR_MODELS_DIR") else {
        eprintln!("skipping: BOOSTR_MODELS_DIR is unset");
        return;
    };
    let scheduler = Scheduler::<CpuRuntime>::new(PathBuf::from(dir), CpuDevice::new());
    let models = scheduler.list_available().expect("lists");
    let refused: Vec<String> = models
        .iter()
        .filter_map(|m| {
            let error = load_model_meta::<CpuRuntime>(&m.path).err()?;
            Some(format!("{}: {error:#}", m.name))
        })
        .collect();
    assert!(refused.is_empty(), "listed but refused: {refused:#?}");
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    eprintln!("{} ids pass: {names:?}", names.len());
    for absent in [
        "whisper-tiny",
        "whisper-base",
        "whisper_mel_fixture.safetensors",
        "neucodec",
        "Kokoro-82M",
    ] {
        assert!(!names.contains(&absent), "{absent} is listed");
    }
}

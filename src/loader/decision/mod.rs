//! System One decision models: family detection, sidecar files, the
//! `/v1/models` report, and the renderer a model's files build.

pub mod detect;
pub mod family_files;
pub mod info;
pub mod renderer;
pub mod sidecar;

pub use detect::{
    detect_decision, detect_decision_with, read_gguf_facts, DecisionSource, GgufFacts,
};
pub use info::{decision_info, CalibrationInfo, ContractInfo, DecisionInfo};
pub use renderer::{load_renderer, LoadedRenderer};
pub use sidecar::Calibration;

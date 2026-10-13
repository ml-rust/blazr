//! System One decision models: family detection, sidecar files, Kev
//! directories, the user's fitted temperatures, the `/v1/models` report,
//! and the renderer a model's files build.

pub mod blazr_calibration;
pub mod detect;
pub mod family_files;
pub mod info;
pub mod kev;
pub mod renderer;
pub mod sidecar;

pub use blazr_calibration::{
    read_calibration_file, served_calibration, write_calibration_file, CalibrationFile, TypeFit,
    BLAZR_CALIBRATION,
};
pub use detect::{
    detect_decision, detect_decision_with, read_gguf_facts, DecisionSource, GgufFacts,
};
pub use info::{
    decision_info, decision_info_with_base, CalibrationInfo, ContractInfo, DecisionInfo,
};
pub use kev::{is_kev_dir, read_kev_adapter, KevAdapter};
pub use renderer::{load_renderer, LoadedRenderer};
pub use sidecar::Calibration;

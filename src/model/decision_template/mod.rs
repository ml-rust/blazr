//! System One decision prompts: a Jev request rendered, per model family,
//! to token rows, answer slots and label tokens for boostr's decision
//! runner.

pub mod ainode;
pub mod decider;
pub mod decider_layout;
pub mod encode;
pub mod family;
pub mod jebadiah;
pub mod jev;
pub mod labels;
pub mod py_text;
pub mod score_keys;
pub mod systemone;

pub use decider::DeciderRenderer;
pub use decider_layout::{DeciderConfig, Layout};
pub use encode::{ChatFrame, DecisionEncoder, Origin, PromptText};
pub use family::{
    DecisionError, DecisionQuestion, DecisionRenderer, Family, RenderedDecision, ScoringRow,
};
pub use jebadiah::JebadiahRenderer;
pub use jev::{Calibration, JevQuestion, JevRequest};

//! Text generation per model family, and the collected text result.

pub(in crate::engine) mod executor_generate;
pub(in crate::engine) mod executor_generate_hybrid;
pub(in crate::engine) mod executor_generate_qwen35;
pub(in crate::engine) mod generate_text;

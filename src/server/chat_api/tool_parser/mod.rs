//! Incremental tool-call parser for model output.
//!
//! One parser reads a stream piece by piece and a complete output in one
//! piece. It releases text as content as soon as no call can start in it,
//! and holds a call back until its boundary. Raw argument fragments never
//! go out as content.

mod parser;
mod steps;

pub use parser::ToolCallParser;

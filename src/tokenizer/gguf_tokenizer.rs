//! GGUF-embedded tokenizer — re-exports from boostr.
//!
//! The tokenizer implementation lives in boostr so it's available to all
//! GGUF consumers (blazr, compressr).

pub use boostr::format::gguf_tokenizer::GgufTokenizer;

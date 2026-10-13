//! Built-in defaults for the user keys that have no section struct.
//!
//! Section keys take their defaults from the `Default` impl of
//! `ServerConfig`, `InferenceConfig` and `GenerationConfig`.

/// Initial KV cache size in tokens. The cache grows as needed.
pub const DEFAULT_NUM_CTX: usize = 2048;

/// Model search directory.
pub const DEFAULT_MODEL_DIR: &str = "./models";

//! Configuration system for blazr
//!
//! BlazrConfig is a superset of boostr's UniversalConfig, adding
//! inference-specific and server settings.

mod blazr;
mod generation;
mod inference;
mod server;
mod user;

pub use blazr::{parse_dtype, BlazrConfig};
pub use generation::GenerationConfig;
pub use inference::{DeviceConfig, InferenceConfig};
pub use server::ServerConfig;
pub use user::UserConfig;

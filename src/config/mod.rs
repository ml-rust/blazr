//! Configuration system for blazr
//!
//! `~/.blazr/config.yaml` holds the user keys and the `server`,
//! `inference` and `generation` sections. `resolve` applies CLI flags and
//! environment variables over it. A model's `config.json` holds the model
//! and its per-model `inference` section.

mod blazr;
mod device;
mod dtype;
mod resolution;
mod sections;
mod user_file;

pub use blazr::{parse_dtype, BlazrConfig};
pub use device::DeviceChoice;
pub use dtype::{
    compute_dtype, dtype_name, fixed_format_dtype, parse_dtype_for, torch_dtype, DtypeRequest,
    DtypeSource, BACKEND_DEFAULT_DTYPE,
};
pub use resolution::*;
pub use sections::*;
pub use user_file::*;

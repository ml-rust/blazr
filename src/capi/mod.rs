//! C ABI for embedding the engine in apps: Android, iOS, desktop.
//!
//! The header is `include/blazr.h`. `examples/capi/README.md` shows the
//! build and link steps.

pub mod embed;
pub mod error;
pub mod generate;
pub mod handle;
pub mod load;
pub mod string;
pub mod tokenize;

pub use embed::blazr_embed;
pub use error::{blazr_last_error, BlazrStatus};
pub use generate::{
    blazr_cancel, blazr_generate, blazr_generate_params_default, BlazrGenerateParams,
    BlazrTokenCallback,
};
pub use handle::BlazrModel;
pub use load::{
    blazr_load_options_default, blazr_model_load, blazr_model_unload, BlazrLoadOptions,
};
pub use string::{blazr_floats_free, blazr_string_free, blazr_tokens_free, blazr_version};
pub use tokenize::blazr_tokenize;

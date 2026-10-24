//! `GenerationConfig` to the parameter types of boostr's host-side logit shaping.

use boostr::inference::sampling::{DryParams, DynaTemp};

use crate::config::GenerationConfig;

impl From<&GenerationConfig> for DryParams {
    fn from(gen: &GenerationConfig) -> Self {
        Self {
            multiplier: gen.dry_multiplier,
            base: gen.dry_base,
            allowed_length: gen.dry_allowed_length,
        }
    }
}

impl From<&GenerationConfig> for DynaTemp {
    fn from(gen: &GenerationConfig) -> Self {
        Self {
            base: gen.temperature,
            range: gen.dynatemp_range,
            exponent: gen.dynatemp_exponent,
        }
    }
}

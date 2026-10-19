//! The load-time check and log of the `inference.speculative` section.

use anyhow::Result;

use boostr::model::{LoadedModel, ModelClient};
use boostr::{DType, Runtime};

use super::eligibility::load_fallback;
use super::Fallback;
use crate::config::InferenceConfig;

/// What a load found for speculative decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum SpeculativeStartup {
    /// The section is unset or `enabled` is off.
    Off,
    /// Eligible requests on this model decode speculatively.
    On,
    /// Speculation is on, but no request on this model can use it.
    Unreachable(Fallback),
}

/// Check the `inference.speculative` section of a load and log its state.
///
/// A set section is checked even when `enabled` is off. Speculation on
/// logs one info line, or one warning when the model cannot use it.
///
/// # Errors
///
/// An error naming the refused key of the section.
pub(in crate::engine) fn report_speculative<R>(
    model: &LoadedModel<R>,
    inference: &InferenceConfig,
) -> Result<SpeculativeStartup>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let Some(spec) = inference.speculative.as_ref() else {
        return Ok(SpeculativeStartup::Off);
    };
    spec.check()?;
    if !spec.enabled {
        return Ok(SpeculativeStartup::Off);
    }
    if let Some(path) = spec.draft_model() {
        tracing::warn!(
            draft_model = path,
            "inference.speculative.draft_model is not supported yet; drafting by prompt lookup"
        );
    }
    if let Err(fallback) = load_fallback(model) {
        tracing::warn!(
            reason = fallback.reason(),
            "speculative decoding is on, but this model decodes without it"
        );
        return Ok(SpeculativeStartup::Unreachable(fallback));
    }
    if inference.graphs {
        tracing::warn!("inference.graphs: graph-captured decode runs without speculative decoding");
    }
    tracing::info!(
        "speculative decoding on: drafter {}, num_draft_tokens {}; single-sequence \
         path on contiguous or paged attention; constrained, mirostat, DRY, typical \
         and dynatemp requests decode without it",
        spec.drafter.name(),
        spec.draft_count()
    );
    Ok(SpeculativeStartup::On)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::SpeculativeDecodingConfig;
    use crate::engine::test_support::{seeded_executor, PagedPool};

    const SEED: u64 = 0x5bec;

    fn with(spec: Option<SpeculativeDecodingConfig>, paged: bool) -> InferenceConfig {
        InferenceConfig {
            speculative: spec,
            paged_attention: paged,
            ..InferenceConfig::default()
        }
    }

    fn on() -> SpeculativeDecodingConfig {
        SpeculativeDecodingConfig {
            enabled: true,
            ..SpeculativeDecodingConfig::default()
        }
    }

    #[test]
    fn each_section_state_reports_its_startup() -> Result<()> {
        let exec = seeded_executor(SEED, None)?;
        let model = exec.model();
        let off = SpeculativeDecodingConfig::default();
        let cases = [
            (with(None, false), SpeculativeStartup::Off),
            (with(Some(off), false), SpeculativeStartup::Off),
            (with(Some(on()), false), SpeculativeStartup::On),
            (with(Some(on()), true), SpeculativeStartup::On),
        ];
        for (inference, want) in cases {
            assert_eq!(
                report_speculative(model, &inference)?,
                want,
                "{inference:?}"
            );
        }
        let draft = SpeculativeDecodingConfig {
            draft_model: Some("draft-dir".to_string()),
            ..on()
        };
        assert_eq!(
            report_speculative(model, &with(Some(draft), false))?,
            SpeculativeStartup::On,
            "a draft model warns and drafts by prompt lookup"
        );
        Ok(())
    }

    #[test]
    fn a_bad_value_fails_the_load_naming_the_key() -> Result<()> {
        let exec = seeded_executor(SEED, None)?;
        for enabled in [false, true] {
            let bad = SpeculativeDecodingConfig {
                enabled,
                num_draft_tokens: Some(0),
                ..SpeculativeDecodingConfig::default()
            };
            let err = report_speculative(exec.model(), &with(Some(bad), false))
                .map_err(|e| e.to_string());
            assert!(
                matches!(&err, Err(msg) if msg.contains("inference.speculative.num_draft_tokens")),
                "{err:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_paged_executor_loads_with_speculation_on() -> Result<()> {
        let pool = PagedPool {
            block_size: 4,
            blocks: 16,
        };
        let exec = seeded_executor(SEED, Some(pool))?;
        let mut inference = exec.config().inference.clone();
        inference.speculative = Some(on());
        assert_eq!(
            report_speculative(exec.model(), &inference)?,
            SpeculativeStartup::On
        );
        Ok(())
    }
}

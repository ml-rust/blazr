//! A request's generation config as the disaggregated protocol carries it.
//!
//! The protocol carries temperature, top-k, top-p, min-p, the repeat,
//! frequency and presence penalties, the penalty window, the seed, the
//! token cap, the EOS ids and the stop ids. A request that sets any other
//! sampling or output field is refused, never run with the field dropped.

use anyhow::{anyhow, bail, Result};

use crate::config::GenerationConfig;
use crate::distributed::disaggregated::{FinishReason as WireFinish, GenerationParams};
use crate::engine::{request_sample_rule, token_budget, FinishReason};

/// Every field `gen` sets that the protocol cannot carry, by its request name.
///
/// The test of each field matches the single-process sampler's: a field
/// counts as set when that sampler acts on it.
fn unsupported_fields(gen: &GenerationConfig) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if gen.constraint.is_some() || gen.compiled.is_some() {
        fields.push("grammar (or response_format / tool constraint)");
    }
    if gen.mirostat_mode != 0 {
        fields.push("mirostat");
    }
    if gen.dry_multiplier > 0.0 {
        fields.push("dry_multiplier");
    }
    if gen.typical_p > 0.0 {
        fields.push("typical_p");
    }
    if gen.dynatemp_range > 0.0 && !gen.is_greedy() {
        fields.push("dynatemp_range");
    }
    if !gen.logit_bias.is_empty() {
        fields.push("logit_bias");
    }
    if gen.logprobs {
        fields.push("logprobs");
    }
    fields
}

/// Why disaggregated serving cannot run `gen`, or `None`.
///
/// The text names every unsupported field, or the sampling field out of
/// range, and the next action. The server answers it with 400.
pub(crate) fn disagg_refusal(gen: &GenerationConfig) -> Option<String> {
    let fields = unsupported_fields(gen);
    if !fields.is_empty() {
        return Some(format!(
            "disaggregated serving cannot run {}: its workers sample with temperature, \
             top_k, top_p, min_p, the repeat/frequency/presence penalties and a seed only. \
             Remove the field, or send the request to a `blazr serve` server",
            fields.join(", ")
        ));
    }
    request_sample_rule(gen)
        .check()
        .err()
        .map(|e| format!("invalid sampling field: {e}"))
}

/// The protocol params of a request with `prompt_len` prompt ids on a model
/// with a context of `context_len` tokens.
///
/// The token cap is `max_tokens` clamped to the room the prompt leaves, as
/// the single-process path clamps it. `eos_tokens` end the stream with
/// `Eos`. The config carries no stop ids: stop sequences cut the text on
/// the router.
///
/// # Errors
///
/// A refusal of [`disagg_refusal`], or a prompt that leaves no room for a token.
pub(crate) fn generation_params(
    gen: &GenerationConfig,
    prompt_len: usize,
    context_len: usize,
    eos_tokens: &[u32],
) -> Result<GenerationParams> {
    if let Some(refusal) = disagg_refusal(gen) {
        bail!("{refusal}");
    }
    let budget = token_budget(gen.max_tokens, context_len, prompt_len);
    if budget == 0 {
        bail!(
            "the request leaves no token to generate: max_tokens is {} and the prompt of \
             {prompt_len} tokens leaves {} of the {context_len}-token context. Raise \
             max_tokens or shorten the prompt",
            gen.max_tokens,
            context_len.saturating_sub(prompt_len)
        );
    }
    let max_new_tokens = u32::try_from(budget)
        .map_err(|_| anyhow!("max_tokens {budget} does not fit the protocol's u32 cap"))?;
    Ok(GenerationParams {
        rule: request_sample_rule(gen),
        max_new_tokens,
        eos_tokens: eos_tokens.to_vec(),
        stop_tokens: Vec::new(),
    })
}

/// The server's finish reason of a protocol finish reason.
pub(crate) fn finish_reason(finish: WireFinish) -> FinishReason {
    match finish {
        WireFinish::Eos => FinishReason::Eos,
        WireFinish::Stop => FinishReason::Stop,
        WireFinish::Length => FinishReason::Length,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::config::Constraint;

    fn sampled() -> GenerationConfig {
        GenerationConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.2,
            repeat_last_n: 32,
            frequency_penalty: 0.1,
            presence_penalty: 0.2,
            seed: Some(11),
            max_tokens: 50,
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn every_carried_field_reaches_the_params() -> Result<()> {
        let p = generation_params(&sampled(), 10, 100, &[2])?;
        let r = p.rule;
        assert_eq!(r.filter.temperature, 0.7);
        assert_eq!(
            (r.filter.top_k, r.filter.top_p, r.filter.min_p),
            (40, 0.9, 0.05)
        );
        assert_eq!(r.repeat_penalty, 1.2);
        assert_eq!(r.penalty_last_n, 32);
        assert_eq!((r.frequency_penalty, r.presence_penalty), (0.1, 0.2));
        assert_eq!(r.seed, Some(11));
        assert_eq!(p.max_new_tokens, 50);
        assert_eq!(p.eos_tokens, vec![2]);
        assert!(p.stop_tokens.is_empty());
        Ok(())
    }

    #[test]
    fn max_tokens_clamps_to_the_context_and_a_full_prompt_is_refused() -> Result<()> {
        assert_eq!(
            generation_params(&sampled(), 90, 100, &[])?.max_new_tokens,
            10
        );
        let err = generation_params(&sampled(), 100, 100, &[]).unwrap_err();
        assert!(err.to_string().contains("no token to generate"), "{err}");
        Ok(())
    }

    #[test]
    fn each_unsupported_field_is_refused_by_name() {
        let cases: Vec<(GenerationConfig, &str)> = vec![
            (
                GenerationConfig {
                    constraint: Some(Constraint::Gbnf("root ::= \"a\"".into())),
                    ..sampled()
                },
                "grammar",
            ),
            (
                GenerationConfig {
                    mirostat_mode: 2,
                    ..sampled()
                },
                "mirostat",
            ),
            (
                GenerationConfig {
                    dry_multiplier: 0.8,
                    ..sampled()
                },
                "dry_multiplier",
            ),
            (
                GenerationConfig {
                    typical_p: 0.9,
                    ..sampled()
                },
                "typical_p",
            ),
            (
                GenerationConfig {
                    dynatemp_range: 0.3,
                    ..sampled()
                },
                "dynatemp_range",
            ),
            (
                GenerationConfig {
                    logit_bias: HashMap::from([(3, 1.0)]),
                    ..sampled()
                },
                "logit_bias",
            ),
            (
                GenerationConfig {
                    logprobs: true,
                    ..sampled()
                },
                "logprobs",
            ),
        ];
        for (gen, field) in cases {
            let refusal = disagg_refusal(&gen).unwrap_or_default();
            assert!(refusal.contains(field), "{field}: {refusal:?}");
            assert!(refusal.contains("blazr serve"), "{refusal}");
            assert!(generation_params(&gen, 1, 100, &[]).is_err(), "{field}");
        }
        assert_eq!(disagg_refusal(&sampled()), None);
    }

    #[test]
    fn a_sampling_field_out_of_range_is_refused() {
        let gen = GenerationConfig {
            repeat_penalty: 0.0,
            ..sampled()
        };
        let refusal = disagg_refusal(&gen).unwrap_or_default();
        assert!(refusal.contains("repeat_penalty"), "{refusal}");
    }

    #[test]
    fn greedy_dynatemp_is_not_refused() {
        let gen = GenerationConfig {
            dynatemp_range: 0.3,
            ..GenerationConfig::greedy()
        };
        assert_eq!(disagg_refusal(&gen), None);
    }
}

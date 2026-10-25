//! `GenerationConfig` to the parameter types of boostr's host-side logit shaping.

use boostr::inference::sampling::{DryParams, DynaTemp};
use splintr::{AnyTokenizer, SpecialMode};

use crate::config::GenerationConfig;

/// The DRY penalty of `gen`, or `None` when `dry_multiplier` is not above 0.
///
/// Each `dry_sequence_breakers` string encodes in `SpecialMode::Ordinary`.
/// Its first token id is the breaker, as in llama.cpp. A request resolves
/// its breakers once, before its first token, never per token.
///
/// # Errors
///
/// The text of a request error: a breaker that encodes to no token, or a
/// DRY field out of range. The server answers 400 with it.
pub fn dry_params(
    tokenizer: &AnyTokenizer,
    gen: &GenerationConfig,
) -> Result<Option<DryParams>, String> {
    if gen.dry_multiplier <= 0.0 || gen.dry_multiplier.is_nan() {
        return Ok(None);
    }
    let params = DryParams {
        multiplier: gen.dry_multiplier,
        base: gen.dry_base,
        allowed_length: gen.dry_allowed_length,
        breakers: dry_breaker_ids(tokenizer, &gen.dry_sequence_breakers)?,
    };
    params.check().map_err(|e| e.to_string())?;
    Ok(Some(params))
}

/// The first token id of each breaker string, in input order.
fn dry_breaker_ids(tokenizer: &AnyTokenizer, breakers: &[String]) -> Result<Vec<u32>, String> {
    breakers
        .iter()
        .map(|breaker| {
            let ids = tokenizer
                .encode_with(breaker, &SpecialMode::Ordinary)
                .map_err(|e| format!("dry_sequence_breakers: {breaker:?} does not encode: {e}"))?;
            ids.first().copied().ok_or_else(|| {
                format!(
                    "dry_sequence_breakers: {breaker:?} encodes to no token; \
                     remove it or pass a non-empty string"
                )
            })
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::sampling::apply_dry_penalty;

    fn tokenizer() -> AnyTokenizer {
        crate::tokenizer::from_pretrained("llama3").expect("llama3 loads")
    }

    fn dry(breakers: &[&str]) -> GenerationConfig {
        GenerationConfig {
            dry_multiplier: 0.8,
            dry_sequence_breakers: breakers.iter().map(|b| (*b).to_string()).collect(),
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn each_breaker_resolves_to_its_first_token() {
        let tok = tokenizer();
        let first = |s: &str| tok.encode_with(s, &SpecialMode::Ordinary).expect("encodes")[0];
        let params = dry_params(&tok, &dry(&["\n", ":", "\"", "*", "hello world"]))
            .expect("valid")
            .expect("enabled");
        assert_eq!(
            params.breakers,
            vec![
                first("\n"),
                first(":"),
                first("\""),
                first("*"),
                first("hello")
            ]
        );
        assert_eq!(params.multiplier, 0.8);
        assert_eq!(params.base, 2);
    }

    #[test]
    fn a_resolved_breaker_reaches_the_penalty() {
        // In llama3, "#" is id 2 and "&" is id 5.
        let tok = tokenizer();
        let params = dry_params(&tok, &dry(&["#"]))
            .expect("valid")
            .expect("enabled");
        assert_eq!(params.breakers, [2]);
        // Suffix [9] repeats as [5, 9], so 2 takes 2 * 0.8 without breakers.
        let history = [5, 9, 2, 7, 5, 9];
        let mut open = vec![0.0f32; 8];
        apply_dry_penalty(
            &mut open,
            &history,
            &dry_params(&tok, &dry(&[]))
                .expect("valid")
                .expect("enabled"),
        );
        assert_eq!(open[2], -1.6);
        // Breaker 2 sits before the last [5, 9]: the match keeps its length.
        let mut far = vec![0.0f32; 8];
        apply_dry_penalty(&mut far, &history, &params);
        assert_eq!(far, open);
        // Breaker "&" (5) cuts the match to [9].
        let cut = dry_params(&tok, &dry(&["&"]))
            .expect("valid")
            .expect("enabled");
        assert_eq!(cut.breakers, [5]);
        let mut short = vec![0.0f32; 8];
        apply_dry_penalty(&mut short, &history, &cut);
        assert_eq!(short[2], -0.8);
    }

    #[test]
    fn a_control_token_spelling_resolves_as_plain_text() {
        let tok = tokenizer();
        let params = dry_params(&tok, &dry(&["<|eot_id|>"]))
            .expect("valid")
            .expect("enabled");
        let eot = tok
            .encode_with("<|eot_id|>", &SpecialMode::Ordinary)
            .expect("encodes")[0];
        assert_eq!(params.breakers, vec![eot]);
        assert_ne!(Some(eot), tok.eos_token_id());
    }

    #[test]
    fn absent_breakers_set_none() {
        let params = dry_params(&tokenizer(), &dry(&[]))
            .expect("valid")
            .expect("enabled");
        assert!(params.breakers.is_empty());
    }

    #[test]
    fn disabled_dry_resolves_nothing() {
        let gen = GenerationConfig {
            dry_multiplier: 0.0,
            ..dry(&[""])
        };
        assert_eq!(dry_params(&tokenizer(), &gen), Ok(None));
    }

    #[test]
    fn an_empty_breaker_is_a_request_error() {
        let err = dry_params(&tokenizer(), &dry(&["\n", ""])).expect_err("empty breaker");
        assert!(err.contains("dry_sequence_breakers"), "{err}");
        assert!(err.contains("encodes to no token"), "{err}");
    }

    #[test]
    fn a_dry_field_out_of_range_is_a_request_error() {
        let gen = GenerationConfig {
            dry_base: 0,
            ..dry(&[])
        };
        let err = dry_params(&tokenizer(), &gen).expect_err("base 0");
        assert!(err.contains("base"), "{err}");
    }
}

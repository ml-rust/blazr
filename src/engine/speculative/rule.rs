//! The boostr sampling rule of one request's generation config.

use boostr::inference::speculative::SampleRule;
use boostr::FilterRule;

use crate::config::GenerationConfig;

/// The speculative acceptance rule of `gen`, or `None` for the pure greedy path.
///
/// Temperature 0 with no penalty maps to `None`. Temperature 0 with a
/// penalty maps to a greedy rule: its argmax runs over penalized logits.
/// The penalty window is the last `repeat_last_n` tokens of the prefix.
/// 0, or a value at or past the prefix length, counts the whole prefix.
/// The plain sampler counts the same window.
///
/// The rule carries no mirostat, DRY, typical-p, dynamic temperature or
/// grammar. The eligibility check refuses a request that sets one.
pub(in crate::engine) fn sample_rule(gen: &GenerationConfig) -> Option<SampleRule> {
    if gen.is_greedy() && !gen.has_penalties() {
        return None;
    }
    Some(SampleRule {
        filter: FilterRule {
            temperature: gen.temperature,
            top_k: gen.top_k,
            top_p: gen.top_p,
            min_p: gen.min_p,
        },
        repeat_penalty: gen.repeat_penalty,
        frequency_penalty: gen.frequency_penalty,
        presence_penalty: gen.presence_penalty,
        penalty_last_n: gen.repeat_last_n,
        seed: gen.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use boostr::CpuRuntime;

    use crate::engine::Executor;

    fn greedy_plain() -> GenerationConfig {
        GenerationConfig {
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    #[test]
    fn greedy_without_a_penalty_is_the_pure_greedy_path() {
        assert_eq!(sample_rule(&greedy_plain()), None);
        let seeded = GenerationConfig {
            seed: Some(3),
            top_k: 7,
            ..greedy_plain()
        };
        assert_eq!(
            sample_rule(&seeded),
            None,
            "seed and filters change no argmax"
        );
    }

    #[test]
    fn greedy_with_a_penalty_is_a_greedy_rule() {
        for gen in [
            GenerationConfig::greedy(),
            GenerationConfig {
                frequency_penalty: 0.5,
                ..greedy_plain()
            },
            GenerationConfig {
                presence_penalty: -0.25,
                ..greedy_plain()
            },
        ] {
            let rule = sample_rule(&gen).expect("a penalty selects the verify path");
            assert!(rule.is_greedy());
            assert!(rule.has_penalties());
            assert_eq!(rule.repeat_penalty, gen.repeat_penalty);
            assert_eq!(rule.frequency_penalty, gen.frequency_penalty);
            assert_eq!(rule.presence_penalty, gen.presence_penalty);
            rule.check().expect("the default filters are valid");
        }
        let rule = sample_rule(&GenerationConfig::greedy()).expect("rule");
        assert_eq!((rule.repeat_penalty, rule.penalty_last_n), (1.1, 64));
    }

    #[test]
    fn a_sampled_config_maps_every_field() {
        let gen = GenerationConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.02,
            repeat_penalty: 1.2,
            repeat_last_n: 8,
            frequency_penalty: 0.1,
            presence_penalty: 0.3,
            seed: Some(42),
            ..GenerationConfig::default()
        };
        let want = SampleRule {
            filter: FilterRule {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.02,
            },
            repeat_penalty: 1.2,
            frequency_penalty: 0.1,
            presence_penalty: 0.3,
            penalty_last_n: 8,
            seed: Some(42),
        };
        assert_eq!(sample_rule(&gen), Some(want));
        let unseeded = GenerationConfig { seed: None, ..gen };
        assert_eq!(sample_rule(&unseeded).and_then(|r| r.seed), None);
    }

    /// Token counts of boostr's window: the last `n` tokens, or all for 0.
    fn boostr_counts(prefix: &[u32], n: usize) -> HashMap<i64, i32> {
        let window = if n == 0 {
            prefix
        } else {
            &prefix[prefix.len().saturating_sub(n)..]
        };
        let mut counts = HashMap::new();
        for &tok in window {
            *counts.entry(i64::from(tok)).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn the_window_matches_the_plain_sampler() {
        let prefix = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        for repeat_last_n in [0, 1, 4, 9, 10, 11, 64] {
            let gen = GenerationConfig {
                temperature: 0.5,
                repeat_last_n,
                ..GenerationConfig::default()
            };
            let rule = sample_rule(&gen).expect("sampled");
            let (ids, cnts) = Executor::<CpuRuntime>::penalty_window(&prefix, repeat_last_n);
            let plain: HashMap<i64, i32> = ids.into_iter().zip(cnts).collect();
            assert_eq!(
                boostr_counts(&prefix, rule.penalty_last_n),
                plain,
                "repeat_last_n {repeat_last_n}"
            );
        }
    }
}

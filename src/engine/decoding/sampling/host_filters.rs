//! Host-side logit shaping: dynamic temperature from entropy, the
//! penalty window, the DRY penalty and the typical filter.

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use crate::engine::executor::Executor;

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Compute dynamic temperature from logit entropy.
    ///
    /// Maps entropy to a temperature in [base - range, base + range].
    /// Low entropy (model is confident) → lower temperature.
    /// High entropy (model is uncertain) → higher temperature.
    pub(crate) fn compute_dynamic_temperature(
        &self,
        logits: &Tensor<R>,
        gen_config: &GenerationConfig,
    ) -> f32 {
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all_logits: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[offset..offset + vocab_size];

        // Compute softmax and entropy
        let max_logit = last_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = last_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();

        let mut entropy = 0.0f32;
        for &p in &probs {
            let normalized = p / sum;
            if normalized > 0.0 {
                entropy -= normalized * normalized.ln();
            }
        }

        // Max possible entropy = ln(vocab_size)
        let max_entropy = (vocab_size as f32).ln();
        // Normalize entropy to [0, 1]
        let normalized_entropy = if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Apply exponent for non-linear mapping
        let mapped = normalized_entropy.powf(gen_config.dynatemp_exponent);

        // Scale to [base - range, base + range]
        let base = gen_config.temperature;
        let range = gen_config.dynatemp_range;
        let dynamic_temp = base - range + 2.0 * range * mapped;
        dynamic_temp.max(0.01) // floor to prevent division by zero
    }

    /// Compute unique token IDs and counts from the penalty window, returning
    /// (token_ids, token_counts) as slices ready for the penalty kernel.
    pub(crate) fn penalty_window(
        recent_tokens: &[u32],
        repeat_last_n: usize,
    ) -> (Vec<i64>, Vec<i32>) {
        let window = if repeat_last_n > 0 && repeat_last_n < recent_tokens.len() {
            &recent_tokens[recent_tokens.len() - repeat_last_n..]
        } else {
            recent_tokens
        };

        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tok in window {
            *counts.entry(tok).or_insert(0) += 1;
        }

        let mut ids: Vec<i64> = Vec::with_capacity(counts.len());
        let mut cnts: Vec<i32> = Vec::with_capacity(counts.len());
        for (&tok, &count) in &counts {
            ids.push(tok as i64);
            cnts.push(count as i32);
        }
        (ids, cnts)
    }

    /// Apply DRY (Don't Repeat Yourself) penalty to logits.
    ///
    /// Scans token history for n-gram patterns that could be extended by the next token.
    /// For each such token, subtracts `multiplier * match_length` from its logit.
    pub(crate) fn apply_dry_penalty(
        logits_vec: &mut [f32],
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) {
        let multiplier = gen_config.dry_multiplier;
        let base = gen_config.dry_base;
        let history = if gen_config.dry_allowed_length > 0
            && gen_config.dry_allowed_length < recent_tokens.len()
        {
            &recent_tokens[recent_tokens.len() - gen_config.dry_allowed_length..]
        } else {
            recent_tokens
        };

        if history.len() < base {
            return;
        }

        // For each possible next token, check if it would extend a repeated n-gram.
        // The last (base-1) tokens form the current suffix we're matching against.
        let suffix_start = history.len().saturating_sub(base - 1);
        let suffix = &history[suffix_start..];

        // Scan history for matches of this suffix and penalize the token that followed
        for start in 0..suffix_start {
            // Check if history[start..start+suffix.len()] == suffix
            let end = start + suffix.len();
            if end >= history.len() {
                break;
            }
            if history[start..end] == *suffix {
                // This suffix appeared before at position `start`. The token after it
                // at history[end] extended the pattern. Find the longest match.
                let mut match_len = suffix.len();
                // Extend backwards to find longer n-gram matches
                let mut s = start;
                let mut e = suffix_start;
                while s > 0 && e > 0 && history[s - 1] == history[e - 1] {
                    match_len += 1;
                    s -= 1;
                    e -= 1;
                }

                let next_token = history[end] as usize;
                if next_token < logits_vec.len() {
                    logits_vec[next_token] -= multiplier * match_len as f32;
                }
            }
        }
    }

    /// Apply typical sampling: filter tokens to those with information content
    /// close to the expected entropy (typicality filter).
    ///
    /// Returns a modified logits vector where atypical tokens are set to -inf.
    pub(crate) fn apply_typical_filter(logits_vec: &mut [f32], typical_p: f32) {
        // Compute softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits_vec.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();

        // Compute entropy
        let mut entropy = 0.0f32;
        for &p in &probs {
            let normalized = p / sum;
            if normalized > 0.0 {
                entropy -= normalized * normalized.ln();
            }
        }

        // Compute |info_content - entropy| for each token, where info_content = -ln(p)
        let mut deviations: Vec<(usize, f32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let normalized = p / sum;
                let info = if normalized > 0.0 {
                    -normalized.ln()
                } else {
                    f32::INFINITY
                };
                let deviation = (info - entropy).abs();
                (i, deviation, normalized)
            })
            .collect();

        // Sort by deviation (most typical first)
        deviations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep tokens until cumulative probability exceeds typical_p
        let mut cumsum = 0.0f32;
        let mut keep = vec![false; logits_vec.len()];
        for &(idx, _, prob) in &deviations {
            if cumsum >= typical_p && cumsum > 0.0 {
                break;
            }
            keep[idx] = true;
            cumsum += prob;
        }

        // Zero out non-kept tokens
        for (i, logit) in logits_vec.iter_mut().enumerate() {
            if !keep[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

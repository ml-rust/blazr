//! Token sampling strategies
//!
//! Contains temperature-based sampling, mirostat v2, DRY penalty,
//! typical filtering, logprob computation, and logit bias application.

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::types::{FinishReason, GeneratedToken, TokenLogprob};

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
        + ModelClient<R>,
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

    /// Sample a token, dispatching to either standard or mirostat sampling.
    ///
    /// Returns (token_gpu_tensor, token_id). When mirostat is None, uses
    /// the fused logits_to_token kernel. When Some, uses mirostat v2.
    pub(crate) fn sample_token_dispatch(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        mirostat: &mut Option<MirostatState>,
    ) -> Result<(Tensor<R>, Option<u32>)> {
        if let Some(ref mut ms) = mirostat {
            let (tensor, id) = self.mirostat_sample_on_device(logits, ms, gen_config)?;
            Ok((tensor, Some(id)))
        } else {
            let tensor = self.logits_to_token_on_device(logits, recent_tokens, gen_config)?;
            Ok((tensor, None)) // token ID read later via pipelined D2H
        }
    }

    /// Sample a token using Mirostat v2 from logits on device.
    ///
    /// Pulls logits to CPU, runs mirostat sampling, returns token as GPU tensor.
    fn mirostat_sample_on_device(
        &self,
        logits: &Tensor<R>,
        mirostat: &mut MirostatState,
        gen_config: &GenerationConfig,
    ) -> Result<(Tensor<R>, u32)> {
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all_logits: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[offset..offset + vocab_size];

        let (token_id, _logprob) = mirostat.sample(last_logits, gen_config.temperature);

        let token_tensor = Tensor::from_slice(&[token_id as i64], &[1], logits.device());
        Ok((token_tensor, token_id))
    }

    /// Build a GeneratedToken with optional logprobs computation.
    pub(crate) fn make_token(
        &self,
        token_id: u32,
        logits: &Tensor<R>,
        gen_config: &GenerationConfig,
        finish_reason: Option<FinishReason>,
    ) -> Result<GeneratedToken> {
        let text = if finish_reason == Some(FinishReason::Eos) {
            String::new()
        } else {
            self.tokenizer().decode(&[token_id]).unwrap_or_default()
        };

        let (logprob, top_logprobs) = if gen_config.logprobs {
            let (lp, top) = self.compute_logprobs(logits, token_id, gen_config.top_logprobs)?;
            (Some(lp), Some(top))
        } else {
            (None, None)
        };

        Ok(GeneratedToken {
            token_id,
            text,
            logprob,
            top_logprobs,
            finish_reason,
        })
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

    /// Compute log probabilities for the last position in logits.
    ///
    /// Returns (logprob of chosen token, top-N logprobs) by computing log-softmax
    /// on the raw logits at the last sequence position.
    fn compute_logprobs(
        &self,
        logits: &Tensor<R>,
        chosen_token: u32,
        top_n: usize,
    ) -> Result<(f32, Vec<TokenLogprob>)> {
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all_logits: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[offset..offset + vocab_size];

        // Log-softmax: log(softmax(x)) = x - log(sum(exp(x - max)))
        let max_logit = last_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = last_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        // Compute log probs and find top-N
        let n = top_n.min(vocab_size).min(20);
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);

        for (i, &logit) in last_logits.iter().enumerate() {
            let lp = logit - log_sum_exp;
            if top.len() < n {
                top.push((i, lp));
                if top.len() == n {
                    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            } else if lp > top[n - 1].1 {
                top[n - 1] = (i, lp);
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let chosen_logprob = if (chosen_token as usize) < vocab_size {
            last_logits[chosen_token as usize] - log_sum_exp
        } else {
            f32::NEG_INFINITY
        };

        let top_logprobs: Vec<TokenLogprob> = top
            .into_iter()
            .map(|(id, lp)| TokenLogprob {
                token_id: id as u32,
                text: self.tokenizer().decode(&[id as u32]).unwrap_or_default(),
                logprob: lp,
            })
            .collect();

        Ok((chosen_logprob, top_logprobs))
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

    /// Fused logits-to-token: narrow + cast + penalties + argmax/sample in a single kernel launch.
    ///
    /// Returns `[1]` I64 tensor on device. All decode paths (greedy, greedy+penalties,
    /// non-greedy) return the same type, enabling uniform pipelined decode.
    pub(crate) fn logits_to_token_on_device(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<Tensor<R>> {
        // Apply CPU-side penalties (DRY, typical) if enabled
        let logits = if gen_config.dry_multiplier > 0.0 || gen_config.typical_p > 0.0 {
            let shape = logits.shape();
            let seq_len = shape[1];
            let vocab_size = shape[2];
            let mut all_logits: Vec<f32> = logits.to_vec();
            let offset = (seq_len - 1) * vocab_size;
            let last_logits = &mut all_logits[offset..offset + vocab_size];

            if gen_config.dry_multiplier > 0.0 {
                Self::apply_dry_penalty(last_logits, recent_tokens, gen_config);
            }
            if gen_config.typical_p > 0.0 {
                Self::apply_typical_filter(last_logits, gen_config.typical_p);
            }

            Tensor::from_slice(&all_logits, shape, logits.device())
        } else {
            logits.clone()
        };

        // Apply logit_bias before sampling if any biases are specified
        let logits = if !gen_config.logit_bias.is_empty() {
            self.apply_logit_bias(&logits, &gen_config.logit_bias)?
        } else {
            logits
        };

        let (ids, cnts) = Self::penalty_window(recent_tokens, gen_config.repeat_last_n);

        let ids_tensor = Tensor::from_slice(&ids, &[ids.len()], self.device());
        let cnts_tensor = Tensor::from_slice(&cnts, &[cnts.len()], self.device());

        let client = R::default_client(self.device());
        let temperature = if gen_config.is_greedy() {
            0.0
        } else if gen_config.dynatemp_range > 0.0 {
            self.compute_dynamic_temperature(&logits, gen_config)
        } else {
            gen_config.temperature
        };

        client
            .logits_to_token(
                &logits,
                &ids_tensor,
                &cnts_tensor,
                ids.len(),
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                gen_config.seed,
            )
            .map_err(|e| anyhow!("logits_to_token failed: {}", e))
    }

    /// Apply per-token logit biases by creating a sparse bias tensor and adding to logits.
    pub(crate) fn apply_logit_bias(
        &self,
        logits: &Tensor<R>,
        bias_map: &std::collections::HashMap<u32, f32>,
    ) -> Result<Tensor<R>> {
        let vocab_size = self.vocab_size();
        let mut bias_vec = vec![0.0f32; vocab_size];
        for (&token_id, &bias) in bias_map {
            if (token_id as usize) < vocab_size {
                bias_vec[token_id as usize] = bias;
            }
        }
        let bias_tensor = Tensor::from_slice(&bias_vec, &[1, 1, vocab_size], self.device());
        logits
            .add(&bias_tensor)
            .map_err(|e| anyhow!("logit_bias add failed: {}", e))
    }
}

/// Mirostat v2 sampler state.
///
/// Maintains a running estimate `mu` that tracks the target entropy `tau`.
/// Each step, it computes surprise = -log2(P(chosen_token)) and adjusts mu
/// so that future samples trend toward the target entropy.
pub struct MirostatState {
    mu: f32,
    tau: f32,
    eta: f32,
    rng: rand::rngs::StdRng,
}

impl MirostatState {
    pub fn new(tau: f32, eta: f32, seed: Option<u64>) -> Self {
        use rand::SeedableRng;
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        // Initialize mu = 2 * tau (standard initialization)
        Self {
            mu: 2.0 * tau,
            tau,
            eta,
            rng,
        }
    }

    /// Sample a token using Mirostat v2.
    ///
    /// Given raw logits (last position, [vocab_size]), applies temperature,
    /// computes softmax, truncates candidates to those with surprise ≤ mu,
    /// samples, and updates mu.
    pub fn sample(&mut self, logits: &[f32], temperature: f32) -> (u32, f32) {
        use rand::Rng;

        let scaled: Vec<f32> = if temperature != 1.0 && temperature > 0.0 {
            let inv_t = 1.0 / temperature;
            logits.iter().map(|&l| l * inv_t).collect()
        } else {
            logits.to_vec()
        };

        // Softmax
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let p = (l - max_logit).exp();
                (i, p)
            })
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        // Sort descending by probability
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate: keep tokens with surprise ≤ mu
        // surprise(token) = -log2(prob)
        let mu = self.mu;
        let candidates: Vec<(usize, f32)> = probs
            .iter()
            .filter(|(_, p)| *p > 0.0 && -p.log2() <= mu)
            .cloned()
            .collect();

        // If no candidates pass the filter, take top-1
        let candidates = if candidates.is_empty() {
            vec![probs[0]]
        } else {
            candidates
        };

        // Renormalize and sample
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0f32;
        let mut chosen = candidates[0].0;
        let mut chosen_prob = candidates[0].1;
        for (id, p) in &candidates {
            cumsum += p / total;
            if cumsum > r {
                chosen = *id;
                chosen_prob = *p;
                break;
            }
        }

        // Update mu: mu = mu - eta * (surprise - tau)
        let surprise = if chosen_prob > 0.0 {
            -chosen_prob.log2()
        } else {
            self.tau
        };
        self.mu -= self.eta * (surprise - self.tau);

        // log prob for the chosen token
        let logprob = chosen_prob.ln();
        (chosen as u32, logprob)
    }
}

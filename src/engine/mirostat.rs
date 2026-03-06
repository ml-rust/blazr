//! Mirostat v2 sampler state and sampling logic.
//!
//! Mirostat maintains a running estimate `mu` that tracks the target entropy `tau`.
//! Each step, it computes surprise = -log2(P(chosen_token)) and adjusts mu
//! so that future samples trend toward the target entropy.

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

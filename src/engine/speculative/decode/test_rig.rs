//! The seeded executors, prompts and comparisons the speculative stream
//! tests share.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use boostr::CpuRuntime;

use crate::config::{GenerationConfig, SpeculativeDecodingConfig};
use crate::engine::executor::Executor;
use crate::engine::requests::batch_test_rig::{alone_tokens, greedy};
use crate::engine::test_support::seeded_executor_ctx;
use crate::engine::types::{FinishReason, GeneratedToken};
use crate::telemetry::test_recorder::TestRecorder;

pub(super) type Exec = Executor<CpuRuntime>;

pub(super) const SEED: u64 = 0x5bec;
pub(super) const CTX: usize = 64;
pub(super) const MAX_TOKENS: usize = 40;
/// Smallest top-2 logprob gap of a compared greedy run.
pub(super) const NEAR_TIE: f32 = 1e-4;
/// Logprobs of the two paths agree to this much.
const LOGPROB_TOL: f32 = 1e-5;
pub(super) const RANDOM: [&[u32]; 3] = [&[1], &[5, 9, 2], &[17, 3, 28, 11]];
/// Prompts where prompt lookup finds matches.
pub(super) const REPETITIVE: [&[u32]; 3] = [
    &[3, 7, 11, 3, 7, 11, 3, 7, 11],
    &[4, 4, 4, 4, 4, 4],
    &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
];

pub(super) fn speculating(mut exec: Exec, k: usize) -> Exec {
    exec.config.inference.speculative = Some(SpeculativeDecodingConfig {
        enabled: true,
        num_draft_tokens: Some(k),
        ..SpeculativeDecodingConfig::default()
    });
    exec
}

/// The seeded llama with speculation off, and its twin with `k` drafts.
pub(super) fn twins(ctx: usize, k: usize) -> Result<(Arc<Exec>, Arc<Exec>)> {
    let plain = seeded_executor_ctx(SEED, ctx)?;
    let spec = speculating(seeded_executor_ctx(SEED, ctx)?, k);
    Ok((Arc::new(plain), Arc::new(spec)))
}

pub(super) fn shape(tokens: &[GeneratedToken]) -> Vec<(u32, String, Option<FinishReason>)> {
    tokens
        .iter()
        .map(|t| (t.token_id, t.text.clone(), t.finish_reason))
        .collect()
}

pub(super) fn count(recorder: &TestRecorder, name: &str) -> f64 {
    recorder.value(name, &[]).unwrap_or(0.0)
}

pub(super) fn with_logprobs(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        logprobs: true,
        top_logprobs: 2,
        ..greedy(max_tokens)
    }
}

/// The smallest top-2 logprob gap over `tokens`.
pub(super) fn margin(tokens: &[GeneratedToken]) -> f32 {
    tokens
        .iter()
        .filter_map(|t| match t.top_logprobs.as_deref() {
            Some([a, b, ..]) => Some((a.logprob - b.logprob).abs()),
            _ => None,
        })
        .fold(f32::INFINITY, f32::min)
}

/// The first of `prompts` whose plain greedy run has no near-tie.
pub(super) async fn tie_free(
    plain: &Exec,
    prompts: &[&'static [u32]],
    max: usize,
) -> Result<&'static [u32]> {
    for &prompt in prompts {
        if margin(&alone_tokens(plain, prompt, &with_logprobs(max)).await?) >= NEAR_TIE {
            return Ok(prompt);
        }
    }
    Err(anyhow!("every prompt of {prompts:?} has a near-tie"))
}

pub(super) fn assert_logprobs_close(got: &[GeneratedToken], want: &[GeneratedToken], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (Some(gl), Some(wl)) = (g.logprob, w.logprob) else {
            panic!("{what}: token {i} has no logprob");
        };
        assert!(
            (gl - wl).abs() < LOGPROB_TOL,
            "{what}: token {i}: {gl} vs {wl}"
        );
        let top = |t: &GeneratedToken| -> Vec<f32> {
            t.top_logprobs.iter().flatten().map(|l| l.logprob).collect()
        };
        let (gt, wt) = (top(g), top(w));
        assert_eq!(gt.len(), wt.len(), "{what}: token {i} top count");
        for (a, b) in gt.iter().zip(&wt) {
            assert!(
                (a - b).abs() < LOGPROB_TOL,
                "{what}: token {i} top: {a} vs {b}"
            );
        }
    }
}

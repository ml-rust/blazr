//! The model forward closures of the disaggregated prefill and decode workers.
//!
//! - `build_prefill_fn`: the prompt forward, its logits and its serialized KV cache
//! - `build_decode_open_fn`: one request's decode steps over the cache prefill sent
//! - `check_disagg_model`: the models whose whole state is a contiguous KV cache
//!
//! Both closures run the executor's own contiguous-cache forward, so a
//! request split across the workers computes the logits a single process
//! computes. The workers sample. A closure returns logits or an error,
//! never a token.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use boostr::{DType, IndexingOps, Runtime};

use crate::distributed::disaggregated::{
    DecodeOpenFn, DecodeStepFn, PrefillCancel as RouterCancel, PrefillFn, PrefillOutput,
};
use crate::distributed::kv_serialize::{deserialize_kv_cache, serialize_kv_cache};
use crate::engine::executor::EngineClient;
use crate::engine::{Executor, PrefillCancel};

/// Checks that `executor`'s model state crosses the network whole.
///
/// The KV handoff carries one contiguous `LayeredKvCache`. A model with SSM
/// or Gated DeltaNet state, or one sharded across tensor-parallel ranks,
/// holds state the handoff does not carry.
///
/// # Errors
///
/// An error naming `role`, the model type and why the model cannot serve.
pub(crate) fn check_disagg_model<R>(executor: &Executor<R>, role: &str) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
{
    let model = executor.model()?;
    let kind = model.model_type();
    if model.needs_gdn_state() {
        bail!("the disaggregated {role} worker cannot serve {kind}: its Gated DeltaNet state does not cross the KV handoff. Serve it with `blazr serve`");
    }
    if model.needs_ssm_state() || !model.needs_kv_cache() {
        bail!("the disaggregated {role} worker cannot serve {kind}: its SSM state does not cross the KV handoff. Serve it with `blazr serve`");
    }
    let tp = executor.config().inference.tensor_parallel_size;
    if tp > 1 {
        bail!("the disaggregated {role} worker cannot serve a model sharded {tp} ways: the KV handoff carries one rank's cache. Set tensor_parallel_size to 1");
    }
    Ok(())
}

/// The prefill closure over `executor`.
///
/// Each call builds a fresh contiguous cache and runs the prompt in chunks
/// of `inference.prefill_chunk_size` tokens. It returns the last chunk's
/// `[1, n, vocab]` logits with the serialized cache. The router's cancel is
/// read before each chunk, so a cancelled request stops at the next chunk
/// boundary and drops its cache.
pub(crate) fn build_prefill_fn<R>(executor: Arc<Executor<R>>) -> PrefillFn<R>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R> + IndexingOps<R>,
{
    Box::new(move |prompt: &[u32], cancel: &RouterCancel| {
        if prompt.is_empty() {
            bail!("the prompt has no tokens");
        }
        let (mut cache, spec) = executor.contiguous_kv_cache(prompt.len(), 0)?;
        let input = executor.create_input_tensor(prompt)?;
        let chunk = executor.config().inference.prefill_chunk_size;
        let left = || cancel.is_cancelled();
        let mut fresh = Some(spec);
        let prefilled = PrefillCancel::when(&left).run(0..prompt.len(), chunk, |r| {
            let piece = executor.prompt_piece(&input, prompt, r.clone())?;
            executor.forward_contiguous(&piece, &mut cache, r.start, fresh.take())
        })?;
        let Some(logits) = prefilled else {
            bail!("the prefill stopped at a chunk boundary: the router cancelled the request");
        };
        let kv_cache = serialize_kv_cache(&cache)?;
        Ok(PrefillOutput { logits, kv_cache })
    })
}

/// The decode open closure over `executor`.
///
/// Opening deserializes the handoff cache once onto the executor's device.
/// Each step feeds one token at its position and returns `[1, 1, vocab]`
/// logits. Dropping the step closure frees the cache.
pub(crate) fn build_decode_open_fn<R>(executor: Arc<Executor<R>>) -> DecodeOpenFn<R>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R> + IndexingOps<R>,
{
    Box::new(move |bytes: &[u8]| {
        let mut cache = deserialize_kv_cache::<R>(bytes, executor.device())?;
        let executor = Arc::clone(&executor);
        let step: DecodeStepFn<R> = Box::new(move |token: u32, position: u32| {
            let cached = cache.seq_len();
            if position as usize != cached {
                return Err(anyhow!(
                    "decode step at position {position}, but the KV cache holds {cached} tokens"
                ));
            }
            let input = executor.create_input_tensor(&[token])?;
            executor.forward_contiguous(&input, &mut cache, cached, None)
        });
        Ok(step)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    use boostr::CpuRuntime;

    use super::*;
    use crate::engine::test_support::seeded_executor;

    const SEED: u64 = 0x5eed;
    const PROMPT: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    fn prefill_fn(chunk: usize) -> Result<PrefillFn<CpuRuntime>> {
        let mut exec = seeded_executor(SEED, None)?;
        exec.config.inference.prefill_chunk_size = chunk;
        Ok(build_prefill_fn(Arc::new(exec)))
    }

    /// The last logits row of `out`.
    fn last_row(out: &PrefillOutput<CpuRuntime>) -> Vec<f32> {
        let rows = out.logits.to_vec::<f32>();
        let vocab = out.logits.shape().last().copied().unwrap_or(1);
        rows[rows.len() - vocab..].to_vec()
    }

    /// Chunks of two leave the cache and the last row of one forward.
    #[test]
    fn a_chunked_prefill_matches_one_forward() -> Result<()> {
        let whole = prefill_fn(0)?(&PROMPT, &RouterCancel::never())?;
        let chunked = prefill_fn(2)?(&PROMPT, &RouterCancel::never())?;
        assert_eq!(
            chunked.logits.shape()[1],
            1,
            "the last chunk holds one token"
        );
        let diff = last_row(&whole)
            .iter()
            .zip(last_row(&chunked))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-4, "the last rows differ by {diff}");
        assert_eq!(whole.kv_cache.len(), chunked.kv_cache.len());
        Ok(())
    }

    /// The router's cancel, read before the second chunk, stops the prefill.
    #[test]
    fn a_router_cancel_stops_the_prefill_at_the_next_chunk() -> Result<()> {
        let reads = Arc::new(AtomicUsize::new(0));
        let probe = {
            let reads = Arc::clone(&reads);
            RouterCancel::new(Arc::new(move || reads.fetch_add(1, SeqCst) >= 1))
        };
        let err = prefill_fn(2)?(&PROMPT, &probe).err().map(|e| e.to_string());
        assert!(
            err.is_some_and(|e| e.contains("cancelled")),
            "the prefill must stop"
        );
        assert_eq!(reads.load(SeqCst), 2, "one of five chunks ran");
        Ok(())
    }
}

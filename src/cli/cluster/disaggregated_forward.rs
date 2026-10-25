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

use crate::distributed::disaggregated::{DecodeOpenFn, DecodeStepFn, PrefillFn, PrefillOutput};
use crate::distributed::kv_serialize::{deserialize_kv_cache, serialize_kv_cache};
use crate::engine::executor::EngineClient;
use crate::engine::Executor;

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
/// Each call builds a fresh contiguous cache, runs the whole prompt and
/// returns the `[1, prompt_len, vocab]` logits with the serialized cache.
pub(crate) fn build_prefill_fn<R>(executor: Arc<Executor<R>>) -> PrefillFn<R>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R> + IndexingOps<R>,
{
    Box::new(move |prompt: &[u32]| {
        if prompt.is_empty() {
            bail!("the prompt has no tokens");
        }
        let (mut cache, spec) = executor.contiguous_kv_cache(prompt.len(), 0)?;
        let input = executor.create_input_tensor(prompt)?;
        let logits = executor.forward_contiguous(&input, &mut cache, 0, Some(spec))?;
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

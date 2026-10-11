//! Recording helpers the engine and the front ends share.

use super::names;
use super::slo;

/// Record token counts after a generation completes
pub fn record_tokens(prompt_tokens: usize, completion_tokens: usize) {
    metrics::counter!(names::TOKENS_PROMPTED).increment(prompt_tokens as u64);
    metrics::counter!(names::TOKENS_GENERATED).increment(completion_tokens as u64);
}

/// Record time-to-first-token (TTFT) in seconds. The SLO checker sees it too.
pub fn record_ttft(ttft_secs: f64) {
    metrics::histogram!(names::TIME_TO_FIRST_TOKEN).record(ttft_secs);
    slo::observe_ttft_ms(ttft_secs * 1000.0);
}

/// Record end-to-end request latency in seconds for the SLO checker.
pub fn record_e2e(e2e_secs: f64) {
    slo::observe_e2e_ms(e2e_secs * 1000.0);
}

/// Record decode throughput (tokens/second) for a completed request
pub fn record_tokens_per_second(tps: f64) {
    if tps > 0.0 {
        metrics::histogram!(names::TOKENS_PER_SECOND).record(tps);
    }
}

/// Increment/decrement the active decode slot gauge
pub fn adjust_decode_slots(delta: f64) {
    if delta >= 0.0 {
        metrics::gauge!(names::ACTIVE_DECODE_SLOTS).increment(delta);
    } else {
        metrics::gauge!(names::ACTIVE_DECODE_SLOTS).decrement(-delta);
    }
}

/// Record inter-token latency (ITL) in seconds. The SLO checker sees it too.
pub fn record_itl(itl_secs: f64) {
    if itl_secs > 0.0 {
        metrics::histogram!(names::INTER_TOKEN_LATENCY).record(itl_secs);
        slo::observe_itl_ms(itl_secs * 1000.0);
    }
}

/// Update VRAM usage gauge (bytes)
#[cfg(feature = "cuda")]
pub fn update_vram_used(device: &boostr::CudaDevice) {
    if let Ok((free, total)) = device.memory_info() {
        let used = total.saturating_sub(free);
        metrics::gauge!(names::VRAM_USED_BYTES).set(used as f64);
    }
}

/// Update KV cache block utilization ratio
pub fn update_kv_cache_utilization(allocated_blocks: usize, total_blocks: usize) {
    if total_blocks > 0 {
        let ratio = allocated_blocks as f64 / total_blocks as f64;
        metrics::gauge!(names::KV_CACHE_UTILIZATION).set(ratio);
    }
}

/// Set the replica count gauge and each replica's in-flight request gauge.
pub fn update_replicas(count: usize, load: &[usize]) {
    metrics::gauge!(names::DATA_PARALLEL_REPLICAS).set(count as f64);
    for (index, active) in load.iter().enumerate() {
        metrics::gauge!(names::REPLICA_ACTIVE_REQUESTS, "replica" => index.to_string())
            .set(*active as f64);
    }
}

/// Record a prefix cache hit
pub fn record_prefix_cache_hit() {
    metrics::counter!(names::PREFIX_CACHE_HITS).increment(1);
}

/// Record a prefix cache miss
pub fn record_prefix_cache_miss() {
    metrics::counter!(names::PREFIX_CACHE_MISSES).increment(1);
}

/// Adjust the in-flight token gauge (positive = add, negative = subtract)
pub fn adjust_inflight_tokens(delta: f64) {
    if delta >= 0.0 {
        metrics::gauge!(names::INFLIGHT_TOKENS).increment(delta);
    } else {
        metrics::gauge!(names::INFLIGHT_TOKENS).decrement(-delta);
    }
}

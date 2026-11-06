//! Metric names.

pub const REQUESTS_TOTAL: &str = "blazr_requests_total";
pub const REQUESTS_ACTIVE: &str = "blazr_requests_active";
pub const REQUEST_DURATION: &str = "blazr_request_duration_seconds";
pub const TOKENS_PROMPTED: &str = "blazr_tokens_prompted_total";
pub const TOKENS_GENERATED: &str = "blazr_tokens_generated_total";
pub const MODELS_LOADED: &str = "blazr_models_loaded";
pub const TIME_TO_FIRST_TOKEN: &str = "blazr_time_to_first_token_seconds";
pub const TOKENS_PER_SECOND: &str = "blazr_tokens_per_second";
pub const INFLIGHT_TOKENS: &str = "blazr_inflight_tokens";
pub const SCHEDULER_MODELS_AVAILABLE: &str = "blazr_scheduler_models_available";
pub const SCHEDULER_EVICTIONS_TOTAL: &str = "blazr_scheduler_evictions_total";
pub const SCHEDULER_LOADS_TOTAL: &str = "blazr_scheduler_loads_total";
pub const QUEUE_DEPTH: &str = "blazr_queue_depth";
pub const ACTIVE_DECODE_SLOTS: &str = "blazr_active_decode_slots";
pub const TOKEN_BUDGET_UTILIZATION: &str = "blazr_token_budget_utilization_ratio";
pub const INTER_TOKEN_LATENCY: &str = "blazr_inter_token_latency_seconds";
pub const VRAM_USED_BYTES: &str = "blazr_vram_used_bytes";
/// Bytes of freed buffers numr's free list keeps on the device.
pub const GPU_FREE_LIST_BYTES: &str = "blazr_gpu_free_list_bytes";
/// Bytes the CUDA pool has handed out, the free list included.
pub const GPU_POOL_USED_BYTES: &str = "blazr_gpu_pool_used_bytes";
/// Bytes the CUDA pool took from the OS.
pub const GPU_POOL_RESERVED_BYTES: &str = "blazr_gpu_pool_reserved_bytes";
pub const KV_CACHE_UTILIZATION: &str = "blazr_kv_cache_utilization_ratio";
pub const PREFIX_CACHE_HITS: &str = "blazr_prefix_cache_hits_total";
pub const PREFIX_CACHE_MISSES: &str = "blazr_prefix_cache_misses_total";
pub const SLO_VIOLATIONS_TOTAL: &str = "blazr_slo_violations_total";
pub const SLO_BATCH_HELD: &str = "blazr_slo_batch_growth_held";
pub const DATA_PARALLEL_REPLICAS: &str = "blazr_data_parallel_replicas";
pub const REPLICA_ACTIVE_REQUESTS: &str = "blazr_replica_active_requests";
pub const DECISION_REQUESTS_TOTAL: &str = "blazr_decision_requests_total";
pub const DECISION_ROWS_TOTAL: &str = "blazr_decision_rows_total";
pub const DECISION_PREFILL_TOKENS_TOTAL: &str = "blazr_decision_prefill_tokens_total";
pub const DECISION_LATENCY: &str = "blazr_decision_latency_seconds";
pub const DECISION_QUEUE_ROWS: &str = "blazr_decision_queue_rows";
pub const DECISION_PREFIX_CACHE_HITS_TOTAL: &str = "blazr_decision_prefix_cache_hits_total";
pub const DECISION_PREFIX_CACHE_MISSES_TOTAL: &str = "blazr_decision_prefix_cache_misses_total";
pub const DECISION_PREFIX_CACHE_EVICTIONS_TOTAL: &str =
    "blazr_decision_prefix_cache_evictions_total";
pub const DECISION_PREFIX_CACHE_BYTES: &str = "blazr_decision_prefix_cache_bytes";
/// Drafts fed to the target.
pub const SPECULATIVE_DRAFTED_TOTAL: &str = "blazr_speculative_drafted_total";
/// Drafts the target accepted.
pub const SPECULATIVE_ACCEPTED_TOTAL: &str = "blazr_speculative_accepted_total";
/// Tokens speculative decoding emitted.
pub const SPECULATIVE_EMITTED_TOTAL: &str = "blazr_speculative_emitted_total";
/// Target forwards speculative decoding ran.
pub const SPECULATIVE_TARGET_FORWARDS_TOTAL: &str = "blazr_speculative_target_forwards_total";
/// Requests with speculation on that decoded without it, by `reason`.
pub const SPECULATIVE_FALLBACKS_TOTAL: &str = "blazr_speculative_fallbacks_total";

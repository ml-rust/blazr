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
pub const KV_CACHE_UTILIZATION: &str = "blazr_kv_cache_utilization_ratio";
pub const PREFIX_CACHE_HITS: &str = "blazr_prefix_cache_hits_total";
pub const PREFIX_CACHE_MISSES: &str = "blazr_prefix_cache_misses_total";
pub const SLO_VIOLATIONS_TOTAL: &str = "blazr_slo_violations_total";
pub const SLO_BATCH_HELD: &str = "blazr_slo_batch_growth_held";
pub const DATA_PARALLEL_REPLICAS: &str = "blazr_data_parallel_replicas";
pub const REPLICA_ACTIVE_REQUESTS: &str = "blazr_replica_active_requests";

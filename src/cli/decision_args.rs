//! The System One decision flags of `blazr serve` and `blazr decision`.

use clap::Args;

use crate::config::{DecisionFamily, DecisionFlags};

/// Decision model flags. Each unset flag keeps `inference.decision.*`.
#[derive(Args, Debug, Clone, Default)]
pub struct DecisionArgs {
    /// Family of the start model on `POST /v1/systemone`, over its sidecar
    /// files and GGUF header. Without --model it applies to every model.
    /// [default: inference.decision.family, else detected]
    #[arg(long, value_enum, value_name = "FAMILY")]
    pub decision_family: Option<DecisionFamily>,

    /// Sidecar file for a model that ships none beside it, such as a bare
    /// `.gguf`: `decider_config.json`, `prompt_contract.json` or
    /// `jebadiah.json`. Same scope as --decision-family.
    /// [default: inference.decision.config]
    #[arg(long, value_name = "FILE")]
    pub decision_config: Option<std::path::PathBuf>,

    /// Shortest shared prefix, in tokens, that one request runs once.
    /// Shorter prefixes run as full rows merged with other requests.
    /// [default: inference.decision.shared_min_tokens, else 768]
    #[arg(long, value_name = "N")]
    pub decision_shared_min_tokens: Option<usize>,

    /// Lane-state budget of one decision forward pass in MiB, 0 for none.
    /// [default: inference.decision.lane_budget_mb, else 8192]
    #[arg(long, value_name = "MIB")]
    pub decision_lane_budget_mb: Option<usize>,

    /// Most prompt rows one request expands to (413 above).
    /// [default: inference.decision.max_rows, else 1024]
    #[arg(long, value_name = "N")]
    pub decision_max_rows: Option<usize>,

    /// Most tokens in one row (413 above).
    /// [default: inference.decision.max_row_tokens, else state cap + 4096]
    #[arg(long, value_name = "N")]
    pub decision_max_row_tokens: Option<usize>,

    /// Most tokens over all rows of one request (413 above).
    /// [default: inference.decision.max_request_tokens, else 1048576]
    #[arg(long, value_name = "N")]
    pub decision_max_request_tokens: Option<usize>,

    /// Most rows queued or running per model (503 above).
    /// [default: inference.decision.max_queue_rows, else 4096]
    #[arg(long, value_name = "N")]
    pub decision_max_queue_rows: Option<usize>,

    /// Milliseconds the decision worker waits after a request for more
    /// requests to merge with it. [default: inference.decision.batch_wait_ms, else 0]
    #[arg(long, value_name = "MS")]
    pub decision_batch_wait_ms: Option<u64>,

    /// Serve a Jebadiah model whose prompt contract does not match the
    /// bundled renderer. [default: inference.decision.allow_contract_mismatch]
    #[arg(long)]
    pub allow_contract_mismatch: bool,

    /// Base model a Kev adapter applies to: a path, or a model name in the
    /// model directory. Same scope as --decision-family.
    /// [default: inference.decision.base, else the adapter's base_model_name_or_path]
    #[arg(long, value_name = "PATH|NAME")]
    pub decision_base: Option<String>,

    /// Kev: read the first 65536 tokens of a longer state instead of
    /// answering 422; every response then says whether it did.
    /// [default: inference.decision.truncate_states]
    #[arg(long)]
    pub decision_truncate_states: bool,
}

impl DecisionArgs {
    /// The flags as config overrides.
    pub fn into_flags(self) -> DecisionFlags {
        DecisionFlags {
            family: self.decision_family,
            config: self.decision_config,
            shared_min_tokens: self.decision_shared_min_tokens,
            lane_budget_mb: self.decision_lane_budget_mb,
            max_rows: self.decision_max_rows,
            max_row_tokens: self.decision_max_row_tokens,
            max_request_tokens: self.decision_max_request_tokens,
            max_queue_rows: self.decision_max_queue_rows,
            batch_wait_ms: self.decision_batch_wait_ms,
            allow_contract_mismatch: self.allow_contract_mismatch,
            base: self.decision_base,
            truncate_states: self.decision_truncate_states,
        }
    }
}

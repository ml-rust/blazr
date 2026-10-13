//! The row and token limits a decision request is checked against before it
//! queues.

use super::error::DecisionFailure;
use crate::config::DecisionConfig;
use crate::model::decision_template::decider::DEFAULT_MAX_STATE_TOKENS;
use crate::model::decision_template::jebadiah::DEFAULT_MAX_TOKENS;
use crate::model::decision_template::Family;

/// The limits of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionLimits {
    /// Most prompt rows.
    pub max_rows: usize,
    /// Most tokens in one row.
    pub max_row_tokens: usize,
    /// Most tokens over all rows.
    pub max_request_tokens: usize,
}

impl DecisionLimits {
    /// The limits `cfg` sets for `family`. The row cap defaults to the
    /// family's state cap (decider `Context:` block, Jebadiah prompt) plus
    /// 4096.
    pub fn new(cfg: &DecisionConfig, family: Family) -> Self {
        let state_cap = match family {
            Family::Decider => DEFAULT_MAX_STATE_TOKENS,
            Family::Jebadiah => DEFAULT_MAX_TOKENS,
        };
        Self {
            max_rows: cfg.max_rows,
            max_row_tokens: cfg.row_token_cap(state_cap),
            max_request_tokens: cfg.max_request_tokens,
        }
    }

    /// Check the prompt `rows` of one request.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::TooLarge`] naming the count, the limit and its key.
    pub fn check(&self, rows: &[Vec<u32>]) -> Result<(), DecisionFailure> {
        let n = rows.len();
        let total: usize = rows.iter().map(Vec::len).sum();
        let longest = rows.iter().map(Vec::len).max().unwrap_or(0);
        let msg = if n > self.max_rows {
            format!(
                "too many questions: the request expands to {n} rows, the limit is {} \
                 (inference.decision.max_rows)",
                self.max_rows
            )
        } else if longest > self.max_row_tokens {
            format!(
                "too many tokens: one row has {longest} tokens, the limit is {} per row \
                 (inference.decision.max_row_tokens)",
                self.max_row_tokens
            )
        } else if total > self.max_request_tokens {
            format!(
                "too many tokens: the request has {total} tokens over {n} rows, the limit is {} \
                 (inference.decision.max_request_tokens)",
                self.max_request_tokens
            )
        } else {
            return Ok(());
        };
        Err(DecisionFailure::TooLarge(msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decider_defaults_and_each_limit() {
        let limits = DecisionLimits::new(&DecisionConfig::default(), Family::Decider);
        assert_eq!(
            limits,
            DecisionLimits {
                max_rows: 1024,
                max_row_tokens: 32768 + 4096,
                max_request_tokens: 1 << 20
            }
        );
        let jeb = DecisionLimits::new(&DecisionConfig::default(), Family::Jebadiah);
        assert_eq!(jeb.max_row_tokens, 2048 + 4096);

        let small = DecisionLimits {
            max_rows: 2,
            max_row_tokens: 3,
            max_request_tokens: 5,
        };
        assert!(small.check(&[vec![1, 2], vec![3, 4, 5]]).is_ok());
        let failure = |rows: &[Vec<u32>]| match small.check(rows) {
            Err(DecisionFailure::TooLarge(m)) => m,
            other => panic!("expected TooLarge, got {other:?}"),
        };
        assert!(failure(&[vec![1], vec![2], vec![3]]).contains("max_rows"));
        assert!(failure(&[vec![1, 2, 3, 4]]).contains("max_row_tokens"));
        assert!(failure(&[vec![1, 2, 3], vec![4, 5, 6]]).contains("max_request_tokens"));
    }
}

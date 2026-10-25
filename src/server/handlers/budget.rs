//! In-flight token budget and decode-slot gauge, held by RAII guards.
//!
//! Admission returns a `BudgetPermit`. The permit gives its tokens back in
//! `Drop`, so a handler future dropped by a timeout or a client disconnect
//! releases the budget. A streaming response moves the permit into the task
//! or body that produces the stream.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::server::metrics;

/// The token budget: tokens in flight and the limit on them.
#[derive(Debug)]
pub struct TokenBudget {
    inflight: AtomicUsize,
    /// 0 means unlimited.
    max: usize,
}

impl TokenBudget {
    /// A budget of `max` tokens in flight. 0 is unlimited.
    pub fn new(max: usize) -> Arc<Self> {
        Arc::new(Self {
            inflight: AtomicUsize::new(0),
            max,
        })
    }

    /// The limit on tokens in flight. 0 is unlimited.
    pub fn max(&self) -> usize {
        self.max
    }

    /// Tokens in flight now.
    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Tokens in flight as a share of the limit. 0.0 when unlimited.
    pub fn utilization(&self) -> f64 {
        if self.max == 0 {
            return 0.0;
        }
        self.inflight() as f64 / self.max as f64
    }

    /// Reserve `tokens`. Returns `None` and reserves nothing when `tokens`
    /// more go past the limit, or when the count in flight cannot hold them.
    /// The permit gives the tokens back on drop.
    pub fn try_admit(self: &Arc<Self>, tokens: usize) -> Option<BudgetPermit> {
        let mut current = self.inflight.load(Ordering::Relaxed);
        loop {
            let next = current
                .checked_add(tokens)
                .filter(|&next| self.max == 0 || next <= self.max)?;
            match self.inflight.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(seen) => current = seen,
            }
        }
        metrics::adjust_inflight_tokens(tokens as f64);
        Some(BudgetPermit {
            budget: Arc::clone(self),
            tokens,
        })
    }

    fn release(&self, tokens: usize) {
        self.inflight.fetch_sub(tokens, Ordering::Relaxed);
        metrics::adjust_inflight_tokens(-(tokens as f64));
    }
}

/// The tokens one request reserves: its prompt plus its generation limit.
///
/// The sum saturates at `usize::MAX`. A huge `max_tokens` then asks for more
/// than any limit, and never wraps to a small count.
pub fn request_tokens(prompt_len: usize, max_tokens: usize) -> usize {
    prompt_len.saturating_add(max_tokens)
}

/// Reserved tokens. Dropping the permit releases them once.
#[derive(Debug)]
#[must_use = "dropping the permit at once releases the reserved tokens"]
pub struct BudgetPermit {
    budget: Arc<TokenBudget>,
    tokens: usize,
}

impl BudgetPermit {
    /// The tokens this permit reserves.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

impl Drop for BudgetPermit {
    fn drop(&mut self) {
        self.budget.release(self.tokens);
    }
}

/// One active decode slot in the `blazr_active_decode_slots` gauge.
/// Dropping the guard removes the slot.
#[derive(Debug)]
#[must_use = "dropping the guard at once removes the decode slot"]
pub struct DecodeSlot(());

impl DecodeSlot {
    /// Count one active decode slot.
    pub fn enter() -> Self {
        metrics::adjust_decode_slots(1.0);
        Self(())
    }
}

impl Drop for DecodeSlot {
    fn drop(&mut self) {
        metrics::adjust_decode_slots(-1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use futures::stream::StreamExt;

    #[tokio::test]
    async fn dropping_the_permit_releases_the_tokens() {
        let budget = TokenBudget::new(100);
        let permit = budget.try_admit(40);
        assert!(permit.is_some());
        assert_eq!(budget.inflight(), 40);
        drop(permit);
        assert_eq!(budget.inflight(), 0);
    }

    #[tokio::test]
    async fn a_future_dropped_by_timeout_releases() {
        let budget = TokenBudget::new(100);
        let held = Arc::clone(&budget);
        let result = tokio::time::timeout(Duration::from_millis(20), async move {
            let _permit = held.try_admit(60);
            std::future::pending::<()>().await;
        })
        .await;
        assert!(result.is_err());
        assert_eq!(budget.inflight(), 0);
    }

    #[tokio::test]
    async fn an_aborted_task_releases() {
        let budget = TokenBudget::new(100);
        let held = Arc::clone(&budget);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let _permit = held.try_admit(70);
            let _ = ready_tx.send(());
            std::future::pending::<()>().await;
        });
        assert!(ready_rx.await.is_ok());
        assert_eq!(budget.inflight(), 70);
        handle.abort();
        assert!(handle.await.is_err());
        assert_eq!(budget.inflight(), 0);
    }

    #[tokio::test]
    async fn a_stream_that_owns_the_permit_releases_when_dropped_mid_stream() {
        let budget = TokenBudget::new(100);
        let permit = budget.try_admit(50);
        assert!(permit.is_some());
        let mut body = Box::pin(futures::stream::unfold(permit, |permit| async move {
            Some((1u32, permit))
        }));
        assert_eq!(body.next().await, Some(1));
        assert_eq!(budget.inflight(), 50);
        drop(body);
        assert_eq!(budget.inflight(), 0);
    }

    #[test]
    fn request_tokens_saturates_instead_of_wrapping() {
        assert_eq!(request_tokens(10, 20), 30);
        assert_eq!(request_tokens(10, usize::MAX), usize::MAX);
        assert_eq!(request_tokens(usize::MAX, usize::MAX), usize::MAX);
    }

    #[tokio::test]
    async fn a_huge_max_tokens_is_refused_by_a_limited_budget() {
        let budget = TokenBudget::new(100);
        assert!(budget.try_admit(request_tokens(5, usize::MAX)).is_none());
        assert_eq!(budget.inflight(), 0);
        assert!(budget.try_admit(request_tokens(5, 20)).is_some());
    }

    #[tokio::test]
    async fn a_limit_of_usize_max_never_overflows() {
        let budget = TokenBudget::new(usize::MAX);
        let first = budget.try_admit(request_tokens(1, usize::MAX));
        assert!(first.is_some());
        assert!(budget.try_admit(1).is_none());
        assert_eq!(budget.inflight(), usize::MAX);
        drop(first);
        assert_eq!(budget.inflight(), 0);
    }

    #[tokio::test]
    async fn an_unlimited_budget_admits_a_huge_request_without_wrapping() {
        let budget = TokenBudget::new(0);
        let first = budget.try_admit(request_tokens(5, usize::MAX));
        assert!(first.is_some());
        assert_eq!(budget.inflight(), usize::MAX);
        // A second reservation cannot be counted, so it is refused, never wrapped.
        assert!(budget.try_admit(10).is_none());
        drop(first);
        assert_eq!(budget.inflight(), 0);
        assert!(budget.try_admit(10).is_some());
    }

    #[tokio::test]
    async fn admission_rejects_when_exhausted_and_recovers_after_drop() {
        let budget = TokenBudget::new(100);
        let first = budget.try_admit(80);
        assert!(first.is_some());
        assert!(budget.try_admit(30).is_none());
        assert_eq!(budget.inflight(), 80);
        drop(first);
        assert!(budget.try_admit(30).is_some());
    }

    #[tokio::test]
    async fn concurrent_cycles_return_utilization_to_zero() {
        let budget = TokenBudget::new(1000);
        let mut tasks = Vec::new();
        for i in 0..64usize {
            let held = Arc::clone(&budget);
            tasks.push(tokio::spawn(async move {
                for _ in 0..50 {
                    let permit = held.try_admit(10 + i);
                    tokio::task::yield_now().await;
                    drop(permit);
                }
            }));
        }
        for task in tasks {
            assert!(task.await.is_ok());
        }
        assert_eq!(budget.inflight(), 0);
        assert_eq!(budget.utilization(), 0.0);
    }

    #[tokio::test]
    async fn an_unlimited_budget_always_admits_and_balances() {
        let budget = TokenBudget::new(0);
        let permit = budget.try_admit(usize::MAX / 4);
        assert!(permit.is_some());
        drop(permit);
        assert_eq!(budget.inflight(), 0);
    }
}

//! A guard that lives as long as the work a request starts.
//!
//! A front end admits a request under a guard, such as a concurrency slot.
//! The request's future can drop before its work stops: a client that
//! leaves drops it at once, while a decode thread runs to its next step
//! and a batched sequence waits for the engine's next check. A
//! [`WorkHold`] keeps the guard until the last of these ends.
//!
//! The front end runs the request's future in [`WorkHold::within`]. Each
//! piece of work started inside takes its own clone of the hold:
//!
//! - A decode thread holds it until the thread exits.
//! - A batched request holds it until its scheduler drops its state.
//! - A task spawned in [`WorkHold::carry`] holds it until the task ends,
//!   and the work it starts takes clones in turn.
//!
//! The guard drops with the last clone.

use std::any::Any;
use std::future::Future;
use std::sync::Arc;

tokio::task_local! {
    /// The hold of the request whose future runs in this task.
    static HOLD: Option<WorkHold>;
}

/// A shared guard. Cloning shares it, and the last clone drops it.
#[derive(Clone)]
pub struct WorkHold {
    _guard: Arc<dyn Any + Send + Sync>,
}

impl WorkHold {
    /// A hold of `guard`.
    pub fn new<G: Send + Sync + 'static>(guard: G) -> Self {
        Self {
            _guard: Arc::new(guard),
        }
    }

    /// Run `future` with `hold` as the hold of the work it starts.
    pub fn within<F: Future>(hold: Option<Self>, future: F) -> impl Future<Output = F::Output> {
        HOLD.scope(hold, future)
    }

    /// Run `future` under the hold of the calling task, or under none.
    ///
    /// Wrap a future in this before it goes to `tokio::spawn`: a spawned
    /// task starts with no hold of its own.
    pub fn carry<F: Future>(future: F) -> impl Future<Output = F::Output> {
        Self::within(Self::current(), future)
    }

    /// The hold of the calling task, or `None` outside [`Self::within`].
    pub fn current() -> Option<Self> {
        HOLD.try_with(Clone::clone).ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_hold_reaches_the_scope_and_a_carried_task_only() {
        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        assert!(WorkHold::current().is_none());

        WorkHold::within(Some(hold), async {
            assert!(WorkHold::current().is_some());
            let bare = tokio::spawn(async { WorkHold::current().is_some() });
            assert!(!bare.await.unwrap_or(true), "a bare task has no hold");
            let carried = tokio::spawn(WorkHold::carry(async { WorkHold::current().is_some() }));
            assert!(
                carried.await.unwrap_or(false),
                "a carried task has the hold"
            );
        })
        .await;

        assert_eq!(Arc::strong_count(&probe), 1, "the scope dropped the hold");
    }

    /// A batched request's handle drops while the engine still runs its
    /// sequence. The hold stays until the engine drops the sequence.
    #[tokio::test]
    async fn a_batched_request_keeps_the_hold_until_the_engine_drops_it() -> anyhow::Result<()> {
        use super::super::batch_test_rig::{greedy, ids, rig, POOL};

        let mut rig = rig(POOL)?;
        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        let scheduler = &rig.scheduler;
        let mut handle = WorkHold::within(Some(hold), async {
            scheduler.submit(vec![1, 2, 3], greedy(100))
        })
        .await?;
        rig.engine.tick().await;
        assert_eq!(ids(&mut handle)?.len(), 1, "the prefill streams one token");

        drop(handle);
        assert_eq!(
            Arc::strong_count(&probe),
            2,
            "the hold dropped while the engine still carried the sequence"
        );
        for _ in 0..3 {
            rig.engine.tick().await;
        }
        assert_eq!(Arc::strong_count(&probe), 1, "the engine dropped the hold");
        Ok(())
    }

    #[tokio::test]
    async fn the_guard_drops_with_the_last_clone() {
        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        let clone = hold.clone();
        drop(hold);
        assert_eq!(Arc::strong_count(&probe), 2, "a clone keeps the guard");
        drop(clone);
        assert_eq!(Arc::strong_count(&probe), 1);
    }
}

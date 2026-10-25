//! The wakeup channel and stop checks both lane engines run around a step.
//!
//! A lane engine takes the scheduler's wakeup receiver once, waits on it
//! while no lane is occupied, drops the wakeups that arrive during a step,
//! and reads its stop future between steps.

use std::future::Future;
use std::pin::Pin;

use futures::future::{select, Either};
use futures::FutureExt;
use tokio::sync::mpsc;

use super::lane_scheduler::LaneScheduler;

/// What ended an idle wait.
pub(super) enum Wake {
    /// A wakeup arrived: the queue can hold a request.
    Work,
    /// The stop future resolved.
    Stop,
    /// The scheduler closed its wakeup channel.
    Closed,
}

/// The scheduler's wakeup receiver.
///
/// Returns `None`, and logs an error, when another engine took it first.
pub(super) fn take_wakeups(scheduler: &LaneScheduler) -> Option<mpsc::Receiver<()>> {
    let notify_rx = scheduler.take_notify_rx();
    if notify_rx.is_none() {
        tracing::error!(
            "the lane scheduler's wakeup receiver was already taken; only one engine can \
             drive one LaneScheduler"
        );
    }
    notify_rx
}

/// Wait until a wakeup arrives, the wakeup channel closes, or `stop`
/// resolves.
///
/// `stop` is polled first, so a stop that is already ready wins over a
/// queued wakeup. A closed channel is logged here.
pub(super) async fn wait_for_work<S: Future<Output = ()>>(
    stop: Pin<&mut S>,
    notify_rx: &mut mpsc::Receiver<()>,
) -> Wake {
    let wakeup = std::pin::pin!(notify_rx.recv());
    match select(stop, wakeup).await {
        Either::Left(((), _)) => Wake::Stop,
        Either::Right((Some(()), _)) => Wake::Work,
        Either::Right((None, _)) => {
            tracing::info!("lane scheduler closed its wakeup channel, shutting down");
            Wake::Closed
        }
    }
}

/// Drop the wakeups that arrived during a step.
///
/// They need no action: the next iteration admits from the queue before it
/// steps again.
pub(super) fn drain_wakeups(notify_rx: &mut mpsc::Receiver<()>) {
    while notify_rx.try_recv().is_ok() {}
}

/// Whether `stop` resolved, read without waiting.
pub(super) fn stop_requested<S: Future<Output = ()>>(stop: Pin<&mut S>) -> bool {
    stop.now_or_never().is_some()
}

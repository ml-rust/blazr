//! Disk reads and device uploads off the async executor.
//!
//! A load reads gigabytes and uploads them, so it runs on Tokio's blocking
//! pool and the runtime keeps serving other requests. With no Tokio
//! runtime current, or on a WebAssembly target that starts no threads,
//! there is no pool, and the work runs on the calling thread.

use std::any::Any;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Blocking work that stopped before it returned a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockingStopped {
    /// The work panicked, with the panic message.
    Panicked(String),
    /// The runtime shut down before the work ran.
    Cancelled,
}

impl fmt::Display for BlockingStopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panicked(message) => write!(f, "it panicked: {message}"),
            Self::Cancelled => f.write_str("the runtime shut down before it ran"),
        }
    }
}

impl std::error::Error for BlockingStopped {}

/// The message of a panic payload.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with no message".to_string()
    }
}

/// Whether blocking work can leave the calling thread: a Tokio runtime is
/// current and the target starts threads.
pub(crate) fn has_blocking_pool() -> bool {
    cfg!(not(target_family = "wasm")) && tokio::runtime::Handle::try_current().is_ok()
}

/// Run `work`, which reads the disk or uploads to the device, on the
/// blocking pool, or on the calling thread when
/// [`has_blocking_pool`] is false. Work that a dropped caller leaves keeps
/// running to its end; its value drops.
///
/// # Errors
///
/// [`BlockingStopped`] when `work` panics, or when the runtime shuts down
/// before it runs.
pub(crate) async fn run_blocking<T, F>(work: F) -> Result<T, BlockingStopped>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if !has_blocking_pool() {
        return catch_unwind(AssertUnwindSafe(work))
            .map_err(|payload| BlockingStopped::Panicked(panic_message(payload.as_ref())));
    }
    tokio::task::spawn_blocking(work).await.map_err(|e| {
        if e.is_panic() {
            BlockingStopped::Panicked(panic_message(e.into_panic().as_ref()))
        } else {
            BlockingStopped::Cancelled
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn work_returns_its_value_and_a_panic_becomes_an_error() {
        assert!(has_blocking_pool());
        assert_eq!(run_blocking(|| 7).await, Ok(7));
        let stopped = run_blocking(|| -> u8 { panic!("disk gone") }).await;
        assert_eq!(stopped, Err(BlockingStopped::Panicked("disk gone".into())));
        let owned = run_blocking(|| -> u8 { panic!("{}", String::from("owned")) }).await;
        assert_eq!(owned, Err(BlockingStopped::Panicked("owned".into())));
    }

    #[test]
    fn without_a_runtime_work_runs_inline() {
        assert!(!has_blocking_pool());
        let caller = std::thread::current().id();
        let ran_on = futures::executor::block_on(run_blocking(|| std::thread::current().id()));
        assert_eq!(ran_on, Ok(caller));
        let stopped = futures::executor::block_on(run_blocking(|| -> u8 { panic!("inline") }));
        assert_eq!(stopped, Err(BlockingStopped::Panicked("inline".into())));
    }

    #[tokio::test]
    async fn work_leaves_the_runtime_thread() {
        let caller = std::thread::current().id();
        let ran_on = run_blocking(|| std::thread::current().id()).await;
        assert_ne!(ran_on, Ok(caller));
    }
}

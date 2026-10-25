//! The decode thread itself: start it, run the job, report a panic.

use std::panic::{catch_unwind, AssertUnwindSafe};

use tokio::sync::{mpsc, oneshot};

use boostr::{DType, Runtime};

use super::super::executor::{EngineClient, Executor};
use super::decode_job::DecodeJob;
use super::decode_output::{send_item, DecodeOutput};
use crate::engine::load::blocking::panic_message;
use crate::engine::WorkHold;

/// The name of every decode thread.
const DECODE_THREAD_NAME: &str = "blazr-decode";

/// Start the decode thread for `job`. The returned receiver resolves
/// once the thread has dropped the stream, `out`, `tx` and the caller's
/// [`WorkHold`].
///
/// The thread takes the caller's hold, so a guard behind it outlives the
/// decode even when the caller's future drops first.
pub(in crate::engine::executors) fn spawn_decode<R, O>(
    job: DecodeJob<R>,
    out: O,
    tx: mpsc::Sender<O::Item>,
) -> std::io::Result<oneshot::Receiver<()>>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
    Executor<R>: Send + Sync + 'static,
    O: DecodeOutput,
{
    let (exit_tx, exited) = oneshot::channel::<()>();
    let hold = WorkHold::current();
    std::thread::Builder::new()
        .name(DECODE_THREAD_NAME.to_string())
        .spawn(move || {
            // Declared first, so it drops last: the waiter wakes after
            // every other value of the thread is gone.
            let _exit = exit_tx;
            // Drops after the stream, before the exit signal.
            let _hold = hold;
            let (job, mut out, tx) = (job, out, tx);
            run(job, &mut out, &tx);
        })?;
    Ok(exited)
}

/// The body of a decode thread. A panic becomes one error item.
fn run<R, O>(job: DecodeJob<R>, out: &mut O, tx: &mpsc::Sender<O::Item>)
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
    O: DecodeOutput,
{
    let idle = job.idle;
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            let message = format!("the decode thread cannot build its runtime: {e}");
            tracing::error!("{message}");
            let _ = tx.try_send(O::failure(message));
            return;
        }
    };
    let ran = catch_unwind(AssertUnwindSafe(|| runtime.block_on(job.pump(out, tx))));
    if let Err(payload) = ran {
        let message = panic_message(payload.as_ref());
        tracing::error!("generation panicked on the decode thread: {message}");
        let failure = O::failure(format!("generation panicked: {message}"));
        runtime.block_on(send_item(tx, failure, idle));
    }
}

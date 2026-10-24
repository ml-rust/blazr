//! What a decode thread sends: the per-token mapping and the send loop.

use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::SendTimeoutError;

use super::types::GeneratedToken;

/// What the send loop does with one mapped engine item.
pub(crate) enum Emit<T> {
    /// Send the item and keep decoding.
    Send(T),
    /// Send nothing for this item and keep decoding.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    Skip,
    /// Send the item and stop decoding.
    Last(T),
}

/// Why the send loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamEnd {
    /// The engine stream ran out before an [`Emit::Last`].
    Drained,
    /// An [`Emit::Last`] item went out.
    Finished,
    /// The reader dropped its receiver, or took no item within the idle timeout.
    Closed,
}

/// Turns engine tokens into the items a decode channel carries.
pub(crate) trait DecodeOutput: Send + 'static {
    /// The item the channel carries.
    type Item: Send + 'static;

    /// Map one engine item.
    fn token(&mut self, token: Result<GeneratedToken>) -> Emit<Self::Item>;

    /// The loop stopped for `why`. Only a [`StreamEnd::Drained`] stream
    /// sends the returned item.
    fn end(&mut self, why: StreamEnd) -> Option<Self::Item>;

    /// The item that reports a decode that stopped on `message`: a panic,
    /// or a decode thread that did not start. A WebAssembly target starts
    /// no thread, so it never builds one.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    fn failure(message: String) -> Self::Item;
}

/// Engine tokens, forwarded unchanged. An error is the last item.
pub(crate) struct RawTokens;

impl DecodeOutput for RawTokens {
    type Item = Result<GeneratedToken>;

    fn token(&mut self, token: Result<GeneratedToken>) -> Emit<Self::Item> {
        match token {
            Ok(token) => Emit::Send(Ok(token)),
            Err(e) => Emit::Last(Err(e)),
        }
    }

    fn end(&mut self, _why: StreamEnd) -> Option<Self::Item> {
        None
    }

    fn failure(message: String) -> Self::Item {
        Err(anyhow!(message))
    }
}

/// Drive `stream`, map each item through `out`, and send it on `tx`.
///
/// The loop reads `tx` before each step, so a dropped receiver stops the
/// decode within one token. A send that waits longer than `idle` stops it
/// too. `None` waits without limit. `on_step` runs after each engine item
/// with its index.
pub(crate) async fn forward_tokens<S, O>(
    stream: S,
    out: &mut O,
    tx: &mpsc::Sender<O::Item>,
    idle: Option<Duration>,
    mut on_step: impl FnMut(usize),
) where
    S: Stream<Item = Result<GeneratedToken>>,
    O: DecodeOutput,
{
    let mut stream = std::pin::pin!(stream);
    let mut step = 0;
    loop {
        if tx.is_closed() {
            let _ = out.end(StreamEnd::Closed);
            return;
        }
        let Some(token) = stream.next().await else {
            if let Some(close) = out.end(StreamEnd::Drained) {
                send_item(tx, close, idle).await;
            }
            return;
        };
        on_step(step);
        step += 1;
        match out.token(token) {
            Emit::Skip => {}
            Emit::Send(item) => {
                if !send_item(tx, item, idle).await {
                    let _ = out.end(StreamEnd::Closed);
                    return;
                }
            }
            Emit::Last(item) => {
                send_item(tx, item, idle).await;
                let _ = out.end(StreamEnd::Finished);
                return;
            }
        }
    }
}

/// Send `item` on `tx`, waiting at most `idle` for room. Returns whether
/// the reader got it.
pub(super) async fn send_item<T>(tx: &mpsc::Sender<T>, item: T, idle: Option<Duration>) -> bool {
    let Some(idle) = idle else {
        return tx.send(item).await.is_ok();
    };
    match tx.send_timeout(item, idle).await {
        Ok(()) => true,
        Err(SendTimeoutError::Timeout(_)) => {
            tracing::warn!(
                idle_secs = idle.as_secs_f64(),
                "the reader took no token within inference.decode_idle_timeout_secs; the \
                 generation stopped"
            );
            false
        }
        Err(SendTimeoutError::Closed(_)) => false,
    }
}

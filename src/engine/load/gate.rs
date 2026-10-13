//! One load at a time per model name. A request that finds a load of its
//! model in flight waits for it, then finds the model loaded, so two
//! requests that race load it once. Loads of other models run alongside.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Each gated name's lock, with how many callers hold or wait on it.
type GateMap = HashMap<String, (Arc<AsyncMutex<()>>, usize)>;

/// The gate map, shared with every held gate.
type Gates = Arc<Mutex<GateMap>>;

/// A poisoned lock still guards whole entries: each change is one map
/// call, so a panicking holder leaves no half-written entry.
fn lock(gates: &Gates) -> MutexGuard<'_, GateMap> {
    gates.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The per-name load locks of one scheduler.
#[derive(Default)]
pub(crate) struct LoadGates {
    gates: Gates,
}

/// One caller's use of a name's lock. The last user of a name removes it
/// from the map, so the map holds only names in use.
struct Entry {
    gates: Gates,
    name: String,
    gate: Arc<AsyncMutex<()>>,
}

impl Entry {
    fn new(gates: &Gates, name: &str) -> Self {
        let mut map = lock(gates);
        let (gate, users) = map
            .entry(name.to_string())
            .or_insert_with(|| (Arc::new(AsyncMutex::new(())), 0));
        *users += 1;
        Self {
            gates: Arc::clone(gates),
            name: name.to_string(),
            gate: Arc::clone(gate),
        }
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        let mut map = lock(&self.gates);
        if let Some((_, users)) = map.get_mut(&self.name) {
            *users = users.saturating_sub(1);
            if *users == 0 {
                map.remove(&self.name);
            }
        }
    }
}

/// The held load lock of one name. Dropping it lets the next caller of
/// that name in. It is `Send + 'static`, so a blocking load holds it to
/// the end, even after the request that started the load stops waiting.
pub(crate) struct LoadGate {
    // Field order is drop order: the lock frees before the entry leaves.
    _held: OwnedMutexGuard<()>,
    _entry: Entry,
}

impl LoadGates {
    /// Wait until no other caller holds the lock of `name`, then hold it.
    /// A caller that stops waiting leaves no entry behind.
    pub(crate) async fn acquire(&self, name: &str) -> LoadGate {
        let entry = Entry::new(&self.gates, name);
        let held = Arc::clone(&entry.gate).lock_owned().await;
        LoadGate {
            _held: held,
            _entry: entry,
        }
    }

    /// How many names are in use.
    #[cfg(test)]
    fn in_use(&self) -> usize {
        lock(&self.gates).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const BRIEF: Duration = Duration::from_millis(50);

    #[tokio::test]
    async fn a_name_admits_one_holder_and_other_names_pass() {
        let gates = LoadGates::default();
        let first = gates.acquire("m").await;
        let other = tokio::time::timeout(BRIEF, gates.acquire("n")).await;
        assert!(other.is_ok(), "another name waits on no one");
        let same = tokio::time::timeout(BRIEF, gates.acquire("m")).await;
        assert!(same.is_err(), "the same name waits for the holder");
        drop(other);
        assert_eq!(gates.in_use(), 1);

        drop(first);
        let next = tokio::time::timeout(BRIEF, gates.acquire("m")).await;
        assert!(next.is_ok(), "the next caller holds it once it frees");
        drop(next);
        assert_eq!(gates.in_use(), 0);
    }

    #[tokio::test]
    async fn a_waiter_that_stops_leaves_no_entry() {
        let gates = LoadGates::default();
        let holder = gates.acquire("m").await;
        assert!(tokio::time::timeout(BRIEF, gates.acquire("m"))
            .await
            .is_err());
        drop(holder);
        assert_eq!(gates.in_use(), 0);
    }

    #[test]
    fn a_held_gate_crosses_threads() {
        fn send_static<T: Send + 'static>(_: &T) {}
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        let gates = LoadGates::default();
        let gate = runtime.block_on(gates.acquire("m"));
        send_static(&gate);
        std::thread::spawn(move || drop(gate))
            .join()
            .expect("thread");
        assert_eq!(gates.in_use(), 0);
    }
}

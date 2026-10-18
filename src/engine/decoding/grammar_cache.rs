//! `GrammarCache`: compiled constraint DFAs shared across requests.
//!
//! A DFA depends on its constraint and the compile limits alone, not on the
//! model, so one cache serves every loaded model. The cache evicts the least
//! recently used entry past an entry count or a table byte budget.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use boostr::inference::grammar::json_schema::{
    compile_json_schema, json_object_grammar, SchemaOptions,
};
use boostr::inference::grammar::{compile, ByteDfa, CompileLimits, GrammarError};

use crate::config::Constraint;

/// Most entries a default cache holds.
pub const DEFAULT_MAX_ENTRIES: usize = 64;

/// Most DFA table bytes a default cache holds.
pub const DEFAULT_MAX_BYTES: usize = 256 << 20;

/// Compile `constraint` to its DFA under `opts` and `limits`.
///
/// Any input returns a DFA or a `GrammarError`, never a panic.
pub fn compile_constraint(
    constraint: &Constraint,
    opts: &SchemaOptions,
    limits: &CompileLimits,
) -> Result<ByteDfa, GrammarError> {
    match constraint {
        Constraint::Gbnf(text) => compile(text, limits),
        Constraint::JsonSchema(schema) => compile_json_schema(schema, opts, limits),
        Constraint::JsonObject => json_object_grammar(opts)?.compile(limits),
    }
}

/// Memory one DFA holds: four bytes per table entry, one per accepting flag.
fn dfa_bytes(dfa: &ByteDfa) -> usize {
    dfa.transitions()
        .len()
        .saturating_mul(4)
        .saturating_add(dfa.accepting().len())
}

/// The full cache key: constraint kind and text, schema config and limits.
///
/// The map hashes the whole key and compares it in full on a hit, so two
/// constraints never share an entry.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key(Arc<str>);

impl Key {
    fn new(constraint: &Constraint, opts: &SchemaOptions, limits: &CompileLimits) -> Self {
        let body = match constraint {
            Constraint::Gbnf(text) => format!("gbnf\n{text}"),
            // With `preserve_order`, the text keeps the declared property order.
            Constraint::JsonSchema(schema) => format!("json_schema\n{schema}"),
            Constraint::JsonObject => "json_object".to_string(),
        };
        Self(format!("{opts:?}\n{limits:?}\n{body}").into())
    }
}

/// One cached DFA.
struct Entry {
    dfa: Arc<ByteDfa>,
    bytes: usize,
    last_used: u64,
}

/// One compile in flight. Every caller of its key waits on the same cell.
type InFlight = Arc<OnceLock<Result<Arc<ByteDfa>, GrammarError>>>;

#[derive(Default)]
struct State {
    entries: HashMap<Key, Entry>,
    in_flight: HashMap<Key, InFlight>,
    bytes: usize,
    clock: u64,
}

/// A bounded least-recently-used map from constraint to compiled DFA.
///
/// A miss compiles outside the cache lock. Concurrent callers of one key
/// wait for one compile. An error is returned to each waiter and never
/// cached, so the next call compiles again.
pub struct GrammarCache {
    state: Mutex<State>,
    max_entries: usize,
    max_bytes: usize,
    opts: SchemaOptions,
    limits: CompileLimits,
    compiles: AtomicUsize,
}

impl Default for GrammarCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES)
    }
}

impl GrammarCache {
    /// A cache of at most `max_entries` DFAs and `max_bytes` table bytes.
    ///
    /// It compiles with the default `SchemaOptions` and `CompileLimits`.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            max_entries,
            max_bytes,
            opts: SchemaOptions::default(),
            limits: CompileLimits::default(),
            compiles: AtomicUsize::new(0),
        }
    }

    /// The DFA of `constraint`, from the cache or from one new compile.
    ///
    /// This blocks for the length of a compile. An async caller uses
    /// [`Self::get_or_compile_async`].
    pub fn get_or_compile(&self, constraint: &Constraint) -> Result<Arc<ByteDfa>, GrammarError> {
        let key = Key::new(constraint, &self.opts, &self.limits);
        let cell = {
            let mut state = self.lock();
            state.clock = state.clock.wrapping_add(1);
            let now = state.clock;
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.last_used = now;
                return Ok(Arc::clone(&entry.dfa));
            }
            Arc::clone(state.in_flight.entry(key.clone()).or_default())
        };
        let result = cell
            .get_or_init(|| {
                self.compiles.fetch_add(1, Ordering::Relaxed);
                compile_constraint(constraint, &self.opts, &self.limits).map(Arc::new)
            })
            .clone();
        let mut state = self.lock();
        if state
            .in_flight
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, &cell))
        {
            state.in_flight.remove(&key);
        }
        if let Ok(dfa) = &result {
            self.insert(&mut state, key, dfa);
        }
        result
    }

    /// [`Self::get_or_compile`] on a blocking thread, so an async caller
    /// never stalls its runtime for a compile.
    ///
    /// # Errors
    ///
    /// The `GrammarError` of a constraint that does not compile, or an error
    /// when the compile task stops before it returns.
    #[cfg(feature = "server")]
    pub async fn get_or_compile_async(
        self: &Arc<Self>,
        constraint: Constraint,
    ) -> anyhow::Result<Arc<ByteDfa>> {
        let cache = Arc::clone(self);
        let joined = tokio::task::spawn_blocking(move || cache.get_or_compile(&constraint)).await;
        let result =
            joined.map_err(|e| anyhow::anyhow!("the grammar compile task stopped: {e}"))?;
        Ok(result?)
    }

    /// DFAs the cache holds.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// True when the cache holds no DFA.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Table bytes the cache holds.
    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    /// Compiles this cache ran, errors included.
    pub fn compiles(&self) -> usize {
        self.compiles.load(Ordering::Relaxed)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Insert `dfa` under `key`, then evict the least recently used entries
    /// until both bounds hold. A DFA above the byte budget alone stays out.
    fn insert(&self, state: &mut State, key: Key, dfa: &Arc<ByteDfa>) {
        let bytes = dfa_bytes(dfa);
        if bytes > self.max_bytes || self.max_entries == 0 || state.entries.contains_key(&key) {
            return;
        }
        state.clock = state.clock.wrapping_add(1);
        let last_used = state.clock;
        state.bytes = state.bytes.saturating_add(bytes);
        state.entries.insert(
            key,
            Entry {
                dfa: Arc::clone(dfa),
                bytes,
                last_used,
            },
        );
        while state.entries.len() > self.max_entries || state.bytes > self.max_bytes {
            let oldest = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(oldest) = oldest else { break };
            if let Some(gone) = state.entries.remove(&oldest) {
                state.bytes = state.bytes.saturating_sub(gone.bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gbnf(text: &str) -> Constraint {
        Constraint::Gbnf(text.to_string())
    }

    fn get(cache: &GrammarCache, c: &Constraint) -> Arc<ByteDfa> {
        cache
            .get_or_compile(c)
            .unwrap_or_else(|e| panic!("{c:?}: {e}"))
    }

    #[test]
    fn a_hit_returns_the_same_dfa_without_a_compile() {
        let cache = GrammarCache::default();
        let c = gbnf(r#"root ::= "a" | "b""#);
        let first = get(&cache, &c);
        let second = get(&cache, &c);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.compiles(), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), dfa_bytes(&first));
    }

    #[test]
    fn kinds_with_the_same_text_never_share_an_entry() {
        let cache = GrammarCache::default();
        get(&cache, &Constraint::JsonObject);
        get(&cache, &Constraint::JsonSchema(json!({"type": "integer"})));
        get(&cache, &gbnf(r#"root ::= "json_object""#));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.compiles(), 3);
    }

    #[test]
    fn the_least_recently_used_entry_goes_first() {
        let cache = GrammarCache::new(2, DEFAULT_MAX_BYTES);
        let (a, b, c) = (
            gbnf(r#"root ::= "a""#),
            gbnf(r#"root ::= "b""#),
            gbnf(r#"root ::= "c""#),
        );
        let first_a = get(&cache, &a);
        get(&cache, &b);
        // A use of `a` makes `b` the oldest.
        get(&cache, &a);
        get(&cache, &c);
        assert_eq!(cache.len(), 2);
        assert!(Arc::ptr_eq(&first_a, &get(&cache, &a)));
        assert_eq!(cache.compiles(), 3);
        get(&cache, &b);
        assert_eq!(cache.compiles(), 4, "the evicted entry compiles again");
    }

    #[test]
    fn the_byte_budget_evicts_old_entries_and_refuses_oversized_ones() {
        let one = dfa_bytes(&get(&GrammarCache::default(), &gbnf(r#"root ::= "a""#)));
        let cache = GrammarCache::new(DEFAULT_MAX_ENTRIES, one * 2);
        get(&cache, &gbnf(r#"root ::= "a""#));
        get(&cache, &gbnf(r#"root ::= "b""#));
        get(&cache, &gbnf(r#"root ::= "c""#));
        assert_eq!(cache.len(), 2);
        assert!(cache.bytes() <= one * 2);

        let tiny = GrammarCache::new(DEFAULT_MAX_ENTRIES, one - 1);
        let dfa = get(&tiny, &gbnf(r#"root ::= "a""#));
        assert!(dfa.accepts(b"a"), "an oversized DFA still returns");
        assert!(tiny.is_empty());
    }

    #[test]
    fn an_error_is_returned_and_never_cached() {
        let cache = GrammarCache::default();
        let bad = gbnf("root ::= (");
        assert!(cache.get_or_compile(&bad).is_err());
        assert!(cache.get_or_compile(&bad).is_err());
        assert!(cache.is_empty());
        assert_eq!(
            cache.compiles(),
            2,
            "each call compiles the failed key again"
        );
        let unsupported = Constraint::JsonSchema(json!({"type": "integer", "minimum": 0}));
        assert!(cache.get_or_compile(&unsupported).is_err());
        assert!(cache.is_empty());
    }

    #[test]
    fn concurrent_callers_of_one_key_share_one_compile() {
        let cache = Arc::new(GrammarCache::default());
        let schema = Constraint::JsonSchema(json!({
            "type": "object",
            "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
            "required": ["n", "s"],
        }));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let schema = schema.clone();
                std::thread::spawn(move || cache.get_or_compile(&schema))
            })
            .collect();
        let results: Vec<Arc<ByteDfa>> = workers
            .into_iter()
            .map(|w| {
                w.join()
                    .unwrap_or_else(|_| panic!("a worker panicked"))
                    .unwrap_or_else(|e| panic!("{e}"))
            })
            .collect();
        assert!(cache.compiles() <= 2, "{} compiles", cache.compiles());
        for dfa in &results {
            assert_eq!(**dfa, *results[0]);
        }
        assert_eq!(cache.len(), 1);
    }
}

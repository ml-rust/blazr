//! Commands that load a model in this process: `run`, `chat`, and `bench`.

pub(super) mod bench;
pub(super) mod chat;
pub(super) mod run;
pub(super) mod run_load;
pub(super) mod run_repl;

pub use bench::bench;
pub use chat::chat;
pub use run::run;

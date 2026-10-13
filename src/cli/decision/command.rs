//! The `blazr decision` subcommands and their flags.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::cli::DecisionArgs;
use crate::engine::decision::calibration::DEFAULT_MIN_ROWS;

/// `blazr decision <command>`.
#[derive(Subcommand, Debug)]
pub enum DecisionCommand {
    /// Fit one answer temperature per type (choice, noul, score) of a
    /// decision model on labelled data, and write blazr_calibration.json
    /// beside the model's sidecars. `blazr serve` then serves each fitted
    /// type with it over the vendor's temperature. Vendor files are never
    /// written.
    Calibrate(CalibrateArgs),
}

/// `blazr decision calibrate` flags.
#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// Decision model name or path.
    #[arg(long, short)]
    pub model: String,

    /// JSON lines: labelled requests `{"state", "questions", "labels":
    /// {id: gold}}`, or precomputed records `{"type", "logits" |
    /// "level_logits", "gold"}`. A choice gold is the option key, a noul
    /// gold true or false, a score gold the level index or the level as
    /// the request spells it.
    #[arg(long, value_name = "FILE")]
    pub labels: PathBuf,

    /// Fewest labelled answers a type needs to be fitted. A type with
    /// fewer keeps the vendor temperature.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MIN_ROWS)]
    pub min_rows: usize,

    /// Where to write the file. [default: blazr_calibration.json in the
    /// sidecar directory: the model directory, beside a bare .gguf, or
    /// beside --decision-config]
    #[arg(long, value_name = "FILE")]
    pub out: Option<PathBuf>,

    /// Print the fit and write nothing.
    #[arg(long)]
    pub dry_run: bool,

    #[command(flatten)]
    pub decision: DecisionArgs,
}

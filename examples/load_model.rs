//! Load a model on the CPU and exit. Measures load cost alone.
//!
//! ```bash
//! cargo build --release --example load_model
//! /usr/bin/time -v target/release/examples/load_model /path/to/model.gguf
//! ```
//!
//! The model loads without the warmup pass, so peak RSS is the load alone.
//! On Linux it also prints the resident and peak memory from `/proc`.

use std::path::PathBuf;
use std::time::Instant;

use blazr::boostr::{CpuDevice, CpuRuntime};
use blazr::engine::{load_executor, LoadOptions};

fn main() -> anyhow::Result<()> {
    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        anyhow::bail!("usage: load_model <GGUF file or SafeTensors directory>");
    };
    let options = LoadOptions {
        warmup: false,
        ..LoadOptions::default()
    };
    let start = Instant::now();
    let executor = load_executor::<CpuRuntime>(&path, CpuDevice::new(), &options)?;
    println!(
        "loaded {} (vocab {}) in {:.2?}",
        path.display(),
        executor.vocab_size(),
        start.elapsed()
    );
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS") || line.starts_with("VmHWM") {
                println!("{line}");
            }
        }
    }
    Ok(())
}

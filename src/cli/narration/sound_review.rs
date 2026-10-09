//! `blazr sound-review`: the motion-video skill's developer sound review
//! (`scripts/dev/sound-review.py`). voxr measures, ranks and prints; blazr
//! finds and loads the CLAP model the first time the review ranks.

use std::path::Path;

use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use voxr::clap::ClapBundle;
use voxr::sound_review::{run_review, usage, AudioTextScorer};

use super::config::read_config;
use super::hearing::Placement;
use super::paths::{model_search, working_dir};
use crate::loader::hub::{ModelKind, CLAP_ID};

/// The command line the usage text names.
const RUN_LINE: &str = "blazr sound-review";

/// Loads the CLAP checkpoint at `dir` on `at`.
fn load_clap(dir: &Path, at: Placement) -> voxr::Result<Box<dyn AudioTextScorer>> {
    Ok(match at {
        Placement::Cpu => Box::new(ClapBundle::<CpuRuntime>::from_dir_on(
            dir,
            &CpuDevice::new(),
        )?),
        #[cfg(feature = "cuda")]
        Placement::Cuda => {
            use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
            Box::new(ClapBundle::<CudaRuntime>::from_dir_on(
                dir,
                &CudaDevice::new(0),
            )?)
        }
    })
}

/// Finds the CLAP model, downloading it on first use, and loads it on the
/// first GPU when there is one.
fn scorer() -> Result<Box<dyn AudioTextScorer>, String> {
    let cwd = working_dir().map_err(|e| e.line())?;
    let config = read_config().map_err(|e| e.line())?;
    let dir = model_search(&config, &cwd).obtain("sound-review", CLAP_ID, ModelKind::Clap, None)?;
    load_clap(&dir, Placement::pick(None)).map_err(|e| format!("sound-review: {e}"))
}

/// Runs `blazr sound-review` on `argv`, the arguments after the command,
/// and returns its exit code: 0 when every sound passes, 1 when one fails
/// a limit or the run stops on an error, 2 with no directory.
pub fn sound_review(argv: &[String]) -> i32 {
    let out = run_review(argv, &usage(RUN_LINE), &mut scorer);
    print!("{}", out.stdout);
    eprint!("{}", out.stderr);
    out.code
}

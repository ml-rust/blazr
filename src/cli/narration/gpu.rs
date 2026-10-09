//! Whether VoxCPM2 runs here, as the skill's `voxcpmReady()` decides it: an
//! NVIDIA GPU with 8 GB of memory (7.5 GB or more counts), here read from
//! the CUDA driver. A missing checkpoint downloads on the first run.

/// GPU memory, in GiB, below which VoxCPM2 is not ready.
const MIN_GPU_GB: f64 = 7.5;

/// VoxCPM2's readiness, and why, in the skill's words: the reason follows
/// "VoxCPM2" or "voxcpm" in a message.
#[derive(Debug, Clone, PartialEq)]
pub struct Readiness {
    /// VoxCPM2 runs here.
    pub ok: bool,
    /// Why not, such as `needs an NVIDIA GPU with 8 GB of memory (none
    /// found)`; when ready, the GPU and its memory.
    pub why: String,
}

/// The first CUDA GPU: its name and memory in GiB. `None` when there is
/// none, or this binary is built without CUDA.
pub fn gpu() -> Option<(String, f64)> {
    #[cfg(feature = "cuda")]
    {
        use boostr::runtime::cuda::CudaDevice;
        if CudaDevice::count().ok()? == 0 {
            return None;
        }
        let device = CudaDevice::new(0);
        let name = device.product_name().ok()?;
        let bytes = device.total_memory().ok()?;
        Some((name, bytes as f64 / (1024.0 * 1024.0 * 1024.0)))
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// `x.toFixed(0)` for a GPU's memory in GiB.
fn whole_gb(gb: f64) -> String {
    format!("{:.0}", gb.round())
}

/// VoxCPM2's readiness from the GPU.
pub fn voxcpm_readiness(gpu: Option<(String, f64)>) -> Readiness {
    let Some((name, gb)) = gpu else {
        let why = if cfg!(feature = "cuda") {
            "needs an NVIDIA GPU with 8 GB of memory (none found)".to_string()
        } else {
            "needs an NVIDIA GPU with 8 GB of memory, and this blazr is built without CUDA (use the CUDA build of blazr, or build with --features audio,cuda)".to_string()
        };
        return Readiness { ok: false, why };
    };
    if gb < MIN_GPU_GB {
        return Readiness {
            ok: false,
            why: format!("needs 8 GB of GPU memory ({name} has {} GB)", whole_gb(gb)),
        };
    }
    Readiness {
        ok: true,
        why: format!("{name}, {} GB", whole_gb(gb)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_reads_as_the_skill_words_it() {
        assert_eq!(
            voxcpm_readiness(Some(("GPU".to_string(), 6.0))),
            Readiness {
                ok: false,
                why: "needs 8 GB of GPU memory (GPU has 6 GB)".to_string()
            }
        );
        assert_eq!(
            voxcpm_readiness(Some(("GPU".to_string(), 11.6))),
            Readiness {
                ok: true,
                why: "GPU, 12 GB".to_string()
            }
        );
        let none = voxcpm_readiness(None);
        assert!(!none.ok);
        assert!(none
            .why
            .starts_with("needs an NVIDIA GPU with 8 GB of memory"));
    }
}

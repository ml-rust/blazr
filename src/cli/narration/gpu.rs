//! Whether VoxCPM2 runs here, as the skill's `voxcpmReady()` decides it: an
//! NVIDIA GPU with 8 GB of memory (7.5 GB or more counts), here read from
//! the CUDA driver in a CUDA build and from `nvidia-smi` in any other. A
//! missing checkpoint downloads on the first run.

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
    /// VoxCPM2 runs on the GPU in this binary: it is built with CUDA.
    pub native: bool,
}

/// The first GPU: its name and memory in GiB. A CUDA build reads the CUDA
/// driver. Any other build reads `nvidia-smi`, as the skill does. `None`
/// when there is no GPU or the reader fails.
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
    {
        use std::process::{Command, Stdio};
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .stdin(Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
    }
}

/// The first line of `nvidia-smi --query-gpu=name,memory.total
/// --format=csv,noheader,nounits`: the GPU's name and its memory in GiB.
/// `None` when the line has no name or no numeric memory in MiB.
#[cfg(any(not(feature = "cuda"), test))]
fn parse_nvidia_smi(text: &str) -> Option<(String, f64)> {
    let line = text.lines().next()?;
    let mut fields = line.split(',').map(str::trim);
    let name = fields.next().filter(|n| !n.is_empty())?;
    let mib: f64 = fields.next()?.parse().ok()?;
    Some((name.to_string(), mib / 1024.0))
}

/// `x.toFixed(0)` for a GPU's memory in GiB.
fn whole_gb(gb: f64) -> String {
    format!("{:.0}", gb.round())
}

/// VoxCPM2's readiness from the GPU.
pub fn voxcpm_readiness(gpu: Option<(String, f64)>) -> Readiness {
    let native = cfg!(feature = "cuda");
    let Some((name, gb)) = gpu else {
        return Readiness {
            ok: false,
            why: "needs an NVIDIA GPU with 8 GB of memory (none found)".to_string(),
            native,
        };
    };
    if gb < MIN_GPU_GB {
        return Readiness {
            ok: false,
            why: format!("needs 8 GB of GPU memory ({name} has {} GB)", whole_gb(gb)),
            native,
        };
    }
    Readiness {
        ok: true,
        why: format!("{name}, {} GB", whole_gb(gb)),
        native,
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
                why: "needs 8 GB of GPU memory (GPU has 6 GB)".to_string(),
                native: cfg!(feature = "cuda")
            }
        );
        assert_eq!(
            voxcpm_readiness(Some(("GPU".to_string(), 11.6))),
            Readiness {
                ok: true,
                why: "GPU, 12 GB".to_string(),
                native: cfg!(feature = "cuda")
            }
        );
        let none = voxcpm_readiness(None);
        assert!(!none.ok);
        assert_eq!(
            none.why,
            "needs an NVIDIA GPU with 8 GB of memory (none found)"
        );
        assert_eq!(none.native, cfg!(feature = "cuda"));
    }

    #[test]
    fn nvidia_smi_line_parses_to_name_and_gib() {
        assert_eq!(
            parse_nvidia_smi("GPU A, 12288\n"),
            Some(("GPU A".to_string(), 12.0))
        );
    }

    #[test]
    fn nvidia_smi_multi_gpu_takes_the_first_line() {
        assert_eq!(
            parse_nvidia_smi("GPU A, 8192\nGPU B, 24576\n"),
            Some(("GPU A".to_string(), 8.0))
        );
    }

    #[test]
    fn nvidia_smi_unreadable_output_is_none() {
        assert_eq!(parse_nvidia_smi(""), None);
        assert_eq!(parse_nvidia_smi("garbage"), None);
        assert_eq!(parse_nvidia_smi("GPU, lots"), None);
        assert_eq!(parse_nvidia_smi(", 8192"), None);
    }
}

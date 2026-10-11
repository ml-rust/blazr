//! The `device` key: the device a command runs on.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::runtime::GpuBackend;

/// A device written as `cpu`, `cuda`, `cuda:N`, `wgpu`, or `wgpu:N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum DeviceChoice {
    /// Run on the CPU.
    Cpu,
    /// Run on the CUDA device with this index.
    Cuda(usize),
    /// Run on the WebGPU adapter with this index.
    Wgpu(usize),
}

impl DeviceChoice {
    /// Parse `cpu`, `cuda`, `cuda:N`, `wgpu`, or `wgpu:N`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let indexed = |prefix: &str| -> Option<usize> {
            text.strip_prefix(prefix)
                .and_then(|index| index.parse::<usize>().ok())
        };
        match text {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            "wgpu" => Ok(Self::Wgpu(0)),
            other => indexed("cuda:")
                .map(Self::Cuda)
                .or_else(|| indexed("wgpu:").map(Self::Wgpu))
                .ok_or_else(|| {
                    format!("device '{other}' is not valid. Use cpu, cuda, cuda:N, wgpu, or wgpu:N")
                }),
        }
    }

    /// The device a command runs on when neither a flag nor the `device`
    /// key names one: GPU 0 of the build, or the CPU.
    pub const fn build_default() -> Self {
        match GpuBackend::of_build() {
            Some(GpuBackend::Cuda) => Self::Cuda(0),
            Some(GpuBackend::Wgpu) => Self::Wgpu(0),
            None => Self::Cpu,
        }
    }

    /// The GPU index for a command whose runtime the build fixes.
    ///
    /// `serve`, `chat` and `bench` run on the build's GPU runtime (CUDA,
    /// then wgpu), else on the CPU. A device the build cannot run on is an
    /// error.
    pub fn build_gpu_index(choice: Option<Self>) -> Result<usize, String> {
        let build = GpuBackend::of_build();
        match (choice, build) {
            (None, _) | (Some(Self::Cpu), None) => Ok(0),
            (Some(Self::Cuda(index)), Some(GpuBackend::Cuda))
            | (Some(Self::Wgpu(index)), Some(GpuBackend::Wgpu)) => Ok(index),
            (Some(Self::Cpu), Some(gpu)) => Err(format!(
                "device: cpu does not apply to this command in a {} build. Set device: {}, or \
                 use `blazr run --cpu`",
                gpu.name(),
                gpu.name()
            )),
            (Some(wanted), _) => {
                let wanted_name = wanted.backend_name();
                let runs = build.map_or("cpu", GpuBackend::name);
                Err(format!(
                    "device: {wanted} needs a build with --features {wanted_name}. This build \
                     runs {runs}. Set device: {runs}, or rebuild with --features {wanted_name}"
                ))
            }
        }
    }

    /// `cpu`, `cuda` or `wgpu`.
    pub const fn backend_name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda(_) => "cuda",
            Self::Wgpu(_) => "wgpu",
        }
    }
}

impl TryFrom<String> for DeviceChoice {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::parse(&text)
    }
}

impl From<DeviceChoice> for String {
    fn from(choice: DeviceChoice) -> Self {
        choice.to_string()
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(index) => write!(f, "cuda:{index}"),
            Self::Wgpu(index) => write!(f, "wgpu:{index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_form() {
        assert_eq!(DeviceChoice::parse("cpu"), Ok(DeviceChoice::Cpu));
        assert_eq!(DeviceChoice::parse("cuda"), Ok(DeviceChoice::Cuda(0)));
        assert_eq!(DeviceChoice::parse("cuda:3"), Ok(DeviceChoice::Cuda(3)));
        assert_eq!(DeviceChoice::parse("wgpu"), Ok(DeviceChoice::Wgpu(0)));
        assert_eq!(DeviceChoice::parse(" wgpu:2 "), Ok(DeviceChoice::Wgpu(2)));
        assert!(DeviceChoice::parse("gpu").is_err());
        assert!(DeviceChoice::parse("cuda:x").is_err());
        assert!(DeviceChoice::parse("wgpu:").is_err());
        assert!(DeviceChoice::parse("vulkan").is_err());
    }

    #[test]
    fn display_round_trips() {
        for choice in [
            DeviceChoice::Cpu,
            DeviceChoice::Cuda(1),
            DeviceChoice::Wgpu(4),
        ] {
            assert_eq!(DeviceChoice::parse(&choice.to_string()), Ok(choice));
        }
    }

    #[test]
    fn default_is_the_build_gpu() {
        let expected = match GpuBackend::of_build() {
            Some(GpuBackend::Cuda) => DeviceChoice::Cuda(0),
            Some(GpuBackend::Wgpu) => DeviceChoice::Wgpu(0),
            None => DeviceChoice::Cpu,
        };
        assert_eq!(DeviceChoice::build_default(), expected);
    }

    #[test]
    fn build_index_follows_the_build() {
        assert_eq!(DeviceChoice::build_gpu_index(None), Ok(0));
        let cuda = DeviceChoice::build_gpu_index(Some(DeviceChoice::Cuda(2)));
        let wgpu = DeviceChoice::build_gpu_index(Some(DeviceChoice::Wgpu(3)));
        let cpu = DeviceChoice::build_gpu_index(Some(DeviceChoice::Cpu));
        match GpuBackend::of_build() {
            Some(GpuBackend::Cuda) => {
                assert_eq!(cuda, Ok(2));
                assert!(wgpu.is_err());
                assert!(cpu.is_err());
            }
            Some(GpuBackend::Wgpu) => {
                assert!(cuda.is_err());
                assert_eq!(wgpu, Ok(3));
                assert!(cpu.is_err());
            }
            None => {
                assert!(cuda.is_err());
                assert!(wgpu.is_err());
                assert_eq!(cpu, Ok(0));
            }
        }
    }

    #[test]
    fn wrong_backend_error_names_the_fix() {
        let wanted = if GpuBackend::of_build() == Some(GpuBackend::Wgpu) {
            DeviceChoice::Cuda(0)
        } else {
            DeviceChoice::Wgpu(0)
        };
        let err = DeviceChoice::build_gpu_index(Some(wanted)).unwrap_err();
        assert!(err.contains("--features"), "{err}");
        assert!(err.contains(wanted.backend_name()), "{err}");
    }
}

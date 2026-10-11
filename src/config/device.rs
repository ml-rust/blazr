//! The `device` key: the device a command runs on.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A device written as `cpu`, `cuda`, or `cuda:N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum DeviceChoice {
    /// Run on the CPU.
    Cpu,
    /// Run on the CUDA device with this index.
    Cuda(usize),
}

impl DeviceChoice {
    /// Parse `cpu`, `cuda`, or `cuda:N`.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text.trim() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            other => other
                .strip_prefix("cuda:")
                .and_then(|index| index.parse::<usize>().ok())
                .map(Self::Cuda)
                .ok_or_else(|| format!("device '{other}' is not valid. Use cpu, cuda, or cuda:N")),
        }
    }

    /// The CUDA device index for a command whose runtime the build fixes.
    ///
    /// `serve`, `chat` and `bench` run on CUDA in a `cuda` build and on the
    /// CPU otherwise. A device the build cannot run on is an error.
    pub fn build_cuda_index(choice: Option<Self>) -> Result<usize, String> {
        match (choice, cfg!(feature = "cuda")) {
            (None, _) | (Some(Self::Cpu), false) => Ok(0),
            (Some(Self::Cuda(index)), true) => Ok(index),
            (Some(Self::Cpu), true) => Err("device: cpu does not apply to this command in a \
                 CUDA build. Set device: cuda, or use `blazr run --cpu`"
                .to_string()),
            (Some(Self::Cuda(_)), false) => Err("device: cuda needs a build with --features \
                 cuda. Set device: cpu, or rebuild with CUDA"
                .to_string()),
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
        assert!(DeviceChoice::parse("gpu").is_err());
        assert!(DeviceChoice::parse("cuda:x").is_err());
    }

    #[test]
    fn build_index_follows_the_build() {
        assert_eq!(DeviceChoice::build_cuda_index(None), Ok(0));
        let cuda = DeviceChoice::build_cuda_index(Some(DeviceChoice::Cuda(2)));
        let cpu = DeviceChoice::build_cuda_index(Some(DeviceChoice::Cpu));
        if cfg!(feature = "cuda") {
            assert_eq!(cuda, Ok(2));
            assert!(cpu.is_err());
        } else {
            assert!(cuda.is_err());
            assert_eq!(cpu, Ok(0));
        }
    }
}

//! The `--version` text.

/// The optional features this binary is built with, as `--version` lists
/// them after the version: `0.1.1 (audio, cuda)`.
const FEATURES: [(&str, bool); 5] = [
    ("audio", cfg!(feature = "audio")),
    ("cuda", cfg!(feature = "cuda")),
    ("wgpu", cfg!(feature = "wgpu")),
    ("nccl", cfg!(feature = "nccl")),
    ("distributed", cfg!(feature = "distributed")),
];

/// The `--version` text: the crate version, then the built features.
pub(super) static VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let built: Vec<&str> = FEATURES
        .iter()
        .filter(|(_, on)| *on)
        .map(|(name, _)| *name)
        .collect();
    if built.is_empty() {
        env!("CARGO_PKG_VERSION").to_string()
    } else {
        format!("{} ({})", env!("CARGO_PKG_VERSION"), built.join(", "))
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_names_each_built_feature() {
        assert!(VERSION.starts_with(env!("CARGO_PKG_VERSION")));
        assert_eq!(VERSION.contains("(audio"), cfg!(feature = "audio"));
        assert_eq!(VERSION.contains("cuda"), cfg!(feature = "cuda"));
        assert_eq!(VERSION.contains("wgpu"), cfg!(feature = "wgpu"));
        assert_eq!(VERSION.contains('('), FEATURES.iter().any(|(_, on)| *on));
    }
}

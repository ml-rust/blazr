//! Precedence for `blazr swarm` and `blazr disagg`: flag, then environment,
//! then the user config file, then the built-in default.
//!
//! The swarm leader and the disaggregated router serve HTTP from the
//! `server` section. Every role loads its model from `model_dir` on the
//! `device` key's CUDA index.

use std::path::PathBuf;

use super::resolve::{pick, resolve_model, resolve_model_dir};
use super::resolve_serve::check_tls_pair;
use super::{DeviceChoice, EnvVars, FileConfig, ServerConfig};

/// The `swarm` and `disagg` flags that a key or variable also sets. `None`
/// means not given.
#[derive(Debug, Clone, Default)]
pub struct ClusterFlags {
    pub model: Option<String>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub api_key: Option<String>,
}

/// Every value a `swarm` or `disagg` role starts with.
#[derive(Debug, Clone)]
pub struct ClusterSettings {
    /// `--model`, then `default_model`.
    pub model: Option<String>,
    /// `BLAZR_MODEL_DIR`, then `model_dir`, then `./models`.
    pub model_dir: PathBuf,
    /// GPU index from `device`, for the build's runtime. 0 on a CPU build.
    pub gpu_index: usize,
    /// The `server` section with `--port` and `--host` written over it.
    pub server: ServerConfig,
    /// `--api-key`, then `BLAZR_API_KEY`.
    pub api_key: Option<String>,
}

/// Resolve every `swarm` or `disagg` value.
///
/// # Errors
///
/// An error when the device does not fit the build, or when only one of
/// the TLS certificate and key is set.
pub fn resolve_cluster(
    flags: ClusterFlags,
    env: &EnvVars,
    file: &FileConfig,
) -> Result<ClusterSettings, String> {
    let mut server = file.server.clone();
    server.port = pick(flags.port, None, None, server.port);
    server.host = pick(flags.host, None, None, server.host);
    check_tls_pair(&server)?;
    Ok(ClusterSettings {
        model: resolve_model(flags.model, file),
        model_dir: resolve_model_dir(None, env, file),
        gpu_index: DeviceChoice::build_gpu_index(file.device)?,
        server,
        api_key: flags.api_key.or_else(|| env.api_key.clone()),
    })
}

impl ClusterSettings {
    /// The model a worker role loads.
    ///
    /// # Errors
    ///
    /// An error naming `role` when neither `--model` nor `default_model` is set.
    pub fn require_model(&self, role: &str) -> Result<String, String> {
        self.model.clone().ok_or_else(|| {
            format!(
                "the {role} role needs a model. Pass --model, or set default_model in \
                 ~/.blazr/config.yaml"
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn file(text: &str) -> FileConfig {
        FileConfig::parse(text, Path::new("config.yaml"))
            .unwrap()
            .config
    }

    #[test]
    fn port_and_host_flag_then_file_then_default() {
        let from_file = file("server:\n  port: 9100\n  host: 10.0.0.5\n");
        let env = EnvVars::default();
        let flags = ClusterFlags {
            port: Some(7000),
            host: Some("127.0.0.1".into()),
            ..Default::default()
        };
        let s = resolve_cluster(flags, &env, &from_file).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (7000, "127.0.0.1"));

        let s = resolve_cluster(ClusterFlags::default(), &env, &from_file).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (9100, "10.0.0.5"));

        let s = resolve_cluster(ClusterFlags::default(), &env, &file("")).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (8080, "0.0.0.0"));
    }

    #[test]
    fn the_old_top_level_port_key_still_applies() {
        let s = resolve_cluster(
            ClusterFlags::default(),
            &EnvVars::default(),
            &file("port: 9300\n"),
        )
        .unwrap();
        assert_eq!(s.server.port, 9300);
    }

    #[test]
    fn server_limits_and_slo_come_from_the_file() {
        let text = "server:\n  max_concurrent_requests: 2\n  slo:\n    ttft_p95_ms: 300\n";
        let s = resolve_cluster(ClusterFlags::default(), &EnvVars::default(), &file(text)).unwrap();
        assert_eq!(s.server.max_concurrent_requests, 2);
        assert_eq!(s.server.slo.and_then(|slo| slo.ttft_p95_ms), Some(300));
    }

    #[test]
    fn model_device_model_dir_and_api_key() {
        let from_file = file("default_model: file-model\nmodel_dir: /file\n");
        let env = EnvVars {
            model_dir: Some(PathBuf::from("/env")),
            api_key: Some("env-key".into()),
            ..Default::default()
        };
        let s = resolve_cluster(ClusterFlags::default(), &env, &from_file).unwrap();
        assert_eq!(s.model.as_deref(), Some("file-model"));
        assert_eq!(s.model_dir, PathBuf::from("/env"));
        assert_eq!(s.api_key.as_deref(), Some("env-key"));
        assert_eq!(s.gpu_index, 0);

        let flags = ClusterFlags {
            model: Some("flag-model".into()),
            api_key: Some("flag-key".into()),
            ..Default::default()
        };
        let s = resolve_cluster(flags, &env, &from_file).unwrap();
        assert_eq!(s.require_model("prefill").unwrap(), "flag-model");
        assert_eq!(s.api_key.as_deref(), Some("flag-key"));
    }

    #[test]
    fn a_worker_without_a_model_names_its_role() {
        let s = resolve_cluster(ClusterFlags::default(), &EnvVars::default(), &file("")).unwrap();
        let err = s.require_model("decode").unwrap_err();
        assert!(err.contains("decode role"), "{err}");
        assert!(err.contains("default_model"), "{err}");
    }

    #[test]
    fn a_lone_tls_key_is_an_error() {
        let err = resolve_cluster(
            ClusterFlags::default(),
            &EnvVars::default(),
            &file("server:\n  tls_key: /k.pem\n"),
        )
        .unwrap_err();
        assert!(err.contains("tls_cert"), "{err}");
    }
}

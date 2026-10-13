//! Precedence for `blazr serve`: flag, then environment, then the user
//! config file, then the built-in default.

use std::path::PathBuf;

use super::resolve::{
    pick, resolve_dtype, resolve_model, resolve_model_dir, resolve_num_ctx,
    resolve_reasoning_format, resolve_tp_size,
};
use super::{
    resolve_decision, DecisionConfig, DecisionFlags, DeviceChoice, DtypeRequest, EnvVars,
    FileConfig, ReasoningFormat, ServerConfig,
};

/// The `serve` flags that a key or variable also sets. `None` means not given.
#[derive(Debug, Clone, Default)]
pub struct ServeFlags {
    pub model: Option<String>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub api_key: Option<String>,
    pub reasoning_format: Option<ReasoningFormat>,
    pub voice_dir: Option<PathBuf>,
    pub data_parallel: Option<usize>,
    pub dtype: Option<boostr::DType>,
    pub decision: DecisionFlags,
}

/// Every value `blazr serve` starts with.
#[derive(Debug, Clone)]
pub struct ServeSettings {
    /// Model to load at start: `--model`, then `default_model`.
    pub model: Option<String>,
    /// `BLAZR_MODEL_DIR`, then `model_dir`, then `./models`.
    pub model_dir: PathBuf,
    /// GPU index from `device`, for the build's runtime. 0 on a CPU build.
    pub gpu_index: usize,
    /// `num_ctx`, then the default.
    pub num_ctx: usize,
    /// `BLAZR_TP_SIZE`, then `inference.tensor_parallel_size`. `None`
    /// leaves each model's own value.
    pub tensor_parallel_size: Option<usize>,
    /// `--data-parallel`, then `inference.data_parallel`, then 1.
    pub data_parallel: usize,
    /// The `server` section with each given flag written over it.
    pub server: ServerConfig,
    /// `--reasoning-format`, kept apart so a reload of the file never
    /// overrides it.
    pub reasoning_format_flag: Option<ReasoningFormat>,
    /// `--api-key`, then `BLAZR_API_KEY`.
    pub api_key: Option<String>,
    /// `--voice-dir`, then `BLAZR_VOICE_DIR`.
    pub voice_dir: Option<PathBuf>,
    /// `--dtype`, then `inference.dtype`. `None` takes each model's
    /// `torch_dtype`, else f32.
    pub dtype: Option<DtypeRequest>,
    /// `--decision-*` flags, then `inference.decision`.
    pub decision: DecisionConfig,
}

/// Resolve every `serve` value.
///
/// # Errors
///
/// An error when the device does not fit the build, when only one of the
/// TLS certificate and key is set, or when the data-parallel degree does
/// not fit the other settings.
pub fn resolve_serve(
    flags: ServeFlags,
    env: &EnvVars,
    file: &FileConfig,
) -> Result<ServeSettings, String> {
    let mut server = file.server.clone();
    server.port = pick(flags.port, None, None, server.port);
    server.host = pick(flags.host, None, None, server.host);
    server.tls_cert = flags.tls_cert.or(server.tls_cert);
    server.tls_key = flags.tls_key.or(server.tls_key);
    server.reasoning_format = resolve_reasoning_format(flags.reasoning_format, &server);
    check_tls_pair(&server)?;

    let model = resolve_model(flags.model, file);
    let tensor_parallel_size = resolve_tp_size(env, file);
    let data_parallel = pick(flags.data_parallel, None, file.inference.data_parallel(), 1);
    check_data_parallel(data_parallel, tensor_parallel_size, model.is_some())?;
    let decision = resolve_decision(flags.decision, file, model.clone())?;

    Ok(ServeSettings {
        model,
        model_dir: resolve_model_dir(None, env, file),
        gpu_index: DeviceChoice::build_gpu_index(file.device)?,
        num_ctx: resolve_num_ctx(None, file),
        tensor_parallel_size,
        data_parallel,
        server,
        reasoning_format_flag: flags.reasoning_format,
        api_key: flags.api_key.or_else(|| env.api_key.clone()),
        voice_dir: flags.voice_dir.or_else(|| env.voice_dir.clone()),
        dtype: resolve_dtype(flags.dtype, file),
        decision,
    })
}

/// An error when only one of the TLS certificate and key is set.
pub fn check_tls_pair(server: &ServerConfig) -> Result<(), String> {
    if server.tls_cert.is_some() != server.tls_key.is_some() {
        return Err(
            "HTTPS needs both a certificate and a key. Set --tls-cert and --tls-key, \
             or server.tls_cert and server.tls_key"
                .to_string(),
        );
    }
    Ok(())
}

/// Check the data-parallel degree against tensor parallelism and the model.
///
/// # Errors
///
/// An error when `replicas` is 0, when both degrees are above 1, or when
/// replicas above 1 have no start model to copy.
pub fn check_data_parallel(
    replicas: usize,
    tensor_parallel: Option<usize>,
    has_model: bool,
) -> Result<(), String> {
    if replicas == 0 {
        return Err(
            "inference.data_parallel is 0. Set 1 or more, in the file or with --data-parallel"
                .to_string(),
        );
    }
    if replicas > 1 {
        if let Some(tp) = tensor_parallel.filter(|&tp| tp > 1) {
            return Err(tp_conflict(replicas, tp));
        }
        if !has_model {
            return Err(format!(
                "inference.data_parallel is {replicas} and no model loads at start. Pass \
                 --model or set default_model, or set inference.data_parallel to 1"
            ));
        }
    }
    Ok(())
}

/// The error for data and tensor parallelism above 1 together.
pub fn tp_conflict(replicas: usize, tp: usize) -> String {
    format!(
        "inference.data_parallel is {replicas} and inference.tensor_parallel_size is {tp}. \
         blazr serve runs one of the two. Set inference.data_parallel or \
         inference.tensor_parallel_size to 1. --data-parallel and BLAZR_TP_SIZE also set them"
    )
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
        let from_file = file("server:\n  port: 9000\n  host: 127.0.0.1\n");
        let env = EnvVars::default();

        let flags = ServeFlags {
            port: Some(7000),
            host: Some("::1".into()),
            ..Default::default()
        };
        let s = resolve_serve(flags, &env, &from_file).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (7000, "::1"));

        let s = resolve_serve(ServeFlags::default(), &env, &from_file).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (9000, "127.0.0.1"));

        let s = resolve_serve(ServeFlags::default(), &env, &FileConfig::default()).unwrap();
        assert_eq!((s.server.port, s.server.host.as_str()), (8080, "0.0.0.0"));
    }

    #[test]
    fn server_limits_come_from_the_file() {
        let text = "\
server:
  max_concurrent_requests: 3
  request_timeout_secs: 30
  max_body_size: 1024
  max_inflight_tokens: 5000
  slo:
    e2e_p99_ms: 900
";
        let s = resolve_serve(ServeFlags::default(), &EnvVars::default(), &file(text)).unwrap();
        assert_eq!(s.server.max_concurrent_requests, 3);
        assert_eq!(s.server.request_timeout_secs, 30);
        assert_eq!(s.server.max_body_size, 1024);
        assert_eq!(s.server.max_inflight_tokens, 5000);
        assert_eq!(s.server.slo.and_then(|slo| slo.e2e_p99_ms), Some(900));

        let d = resolve_serve(ServeFlags::default(), &EnvVars::default(), &file("")).unwrap();
        let default = ServerConfig::default();
        assert_eq!(
            d.server.max_concurrent_requests,
            default.max_concurrent_requests
        );
        assert!(d.server.slo.is_none());
    }

    #[test]
    fn api_key_and_voice_dir_flag_then_env() {
        let env = EnvVars {
            api_key: Some("env-key".into()),
            voice_dir: Some(PathBuf::from("/env-voices")),
            ..Default::default()
        };
        let flags = ServeFlags {
            api_key: Some("flag-key".into()),
            voice_dir: Some(PathBuf::from("/flag-voices")),
            ..Default::default()
        };
        let s = resolve_serve(flags, &env, &FileConfig::default()).unwrap();
        assert_eq!(s.api_key.as_deref(), Some("flag-key"));
        assert_eq!(s.voice_dir, Some(PathBuf::from("/flag-voices")));

        let s = resolve_serve(ServeFlags::default(), &env, &FileConfig::default()).unwrap();
        assert_eq!(s.api_key.as_deref(), Some("env-key"));
        assert_eq!(s.voice_dir, Some(PathBuf::from("/env-voices")));

        let none = EnvVars::default();
        let s = resolve_serve(ServeFlags::default(), &none, &FileConfig::default()).unwrap();
        assert!(s.api_key.is_none() && s.voice_dir.is_none());
    }

    #[test]
    fn model_and_model_dir_and_num_ctx() {
        let from_file = file("default_model: m\nmodel_dir: /file\nnum_ctx: 8192\n");
        let env = EnvVars {
            model_dir: Some(PathBuf::from("/env")),
            ..Default::default()
        };
        let flags = ServeFlags {
            model: Some("flag-model".into()),
            ..Default::default()
        };
        let s = resolve_serve(flags, &env, &from_file).unwrap();
        assert_eq!(s.model.as_deref(), Some("flag-model"));
        assert_eq!(s.model_dir, PathBuf::from("/env"));
        assert_eq!(s.num_ctx, 8192);

        let s = resolve_serve(ServeFlags::default(), &EnvVars::default(), &from_file).unwrap();
        assert_eq!(s.model.as_deref(), Some("m"));
        assert_eq!(s.model_dir, PathBuf::from("/file"));
    }

    #[test]
    fn reasoning_flag_wins_and_is_kept_apart() {
        let from_file = file("server:\n  reasoning_format: none\n");
        let flags = ServeFlags {
            reasoning_format: Some(ReasoningFormat::Deepseek),
            ..Default::default()
        };
        let s = resolve_serve(flags, &EnvVars::default(), &from_file).unwrap();
        assert_eq!(s.server.reasoning_format, ReasoningFormat::Deepseek);
        assert_eq!(s.reasoning_format_flag, Some(ReasoningFormat::Deepseek));
    }

    #[test]
    fn data_parallel_flag_then_file_then_one() {
        let from_file = file("default_model: m\ninference:\n  data_parallel: 3\n");
        let env = EnvVars::default();
        let flags = ServeFlags {
            data_parallel: Some(2),
            ..Default::default()
        };
        assert_eq!(
            resolve_serve(flags, &env, &from_file)
                .unwrap()
                .data_parallel,
            2
        );
        let s = resolve_serve(ServeFlags::default(), &env, &from_file).unwrap();
        assert_eq!(s.data_parallel, 3);
        let s = resolve_serve(ServeFlags::default(), &env, &file("")).unwrap();
        assert_eq!(s.data_parallel, 1);
    }

    #[test]
    fn data_parallel_with_tensor_parallel_names_both_keys() {
        let text = "default_model: m\ninference:\n  data_parallel: 2\n  tensor_parallel_size: 2\n";
        let err =
            resolve_serve(ServeFlags::default(), &EnvVars::default(), &file(text)).unwrap_err();
        assert!(err.contains("inference.data_parallel"), "{err}");
        assert!(err.contains("inference.tensor_parallel_size"), "{err}");

        // BLAZR_TP_SIZE sets the same key, so the flag and the variable clash too.
        let env = EnvVars {
            tp_size: Some(4),
            ..Default::default()
        };
        let flags = ServeFlags {
            data_parallel: Some(2),
            ..Default::default()
        };
        let err = resolve_serve(flags, &env, &file("default_model: m\n")).unwrap_err();
        assert!(err.contains("tensor_parallel_size is 4"), "{err}");

        // Either degree at 1 is fine.
        let env_one = EnvVars {
            tp_size: Some(1),
            ..Default::default()
        };
        let flags = ServeFlags {
            data_parallel: Some(2),
            ..Default::default()
        };
        assert!(resolve_serve(flags, &env_one, &file("default_model: m\n")).is_ok());
    }

    #[test]
    fn data_parallel_needs_a_start_model_and_at_least_one_replica() {
        let flags = ServeFlags {
            data_parallel: Some(2),
            ..Default::default()
        };
        let err = resolve_serve(flags, &EnvVars::default(), &file("")).unwrap_err();
        assert!(err.contains("--model"), "{err}");

        let zero = file("inference:\n  data_parallel: 0\n");
        let err = resolve_serve(ServeFlags::default(), &EnvVars::default(), &zero).unwrap_err();
        assert!(err.contains("data_parallel is 0"), "{err}");
    }

    #[test]
    fn tls_from_flag_and_file_must_pair() {
        let from_file = file("server:\n  tls_key: /k.pem\n");
        let flags = ServeFlags {
            tls_cert: Some(PathBuf::from("/c.pem")),
            ..Default::default()
        };
        let s = resolve_serve(flags, &EnvVars::default(), &from_file).unwrap();
        assert!(s.server.tls_enabled());

        let err =
            resolve_serve(ServeFlags::default(), &EnvVars::default(), &from_file).unwrap_err();
        assert!(err.contains("tls_cert"), "{err}");
    }

    #[test]
    fn dtype_flag_then_file_then_model() {
        use crate::config::DtypeSource;
        use boostr::DType;

        let from_file = file("inference:\n  dtype: f32\n");
        let env = EnvVars::default();
        let flags = ServeFlags {
            dtype: Some(DType::BF16),
            ..Default::default()
        };
        let s = resolve_serve(flags, &env, &from_file).unwrap();
        let request = s.dtype.unwrap();
        assert_eq!(
            (request.dtype, request.source),
            (DType::BF16, DtypeSource::Flag)
        );

        let s = resolve_serve(ServeFlags::default(), &env, &from_file).unwrap();
        let request = s.dtype.unwrap();
        assert_eq!(
            (request.dtype, request.source),
            (DType::F32, DtypeSource::File)
        );

        let s = resolve_serve(ServeFlags::default(), &env, &file("")).unwrap();
        assert!(s.dtype.is_none());
    }
}

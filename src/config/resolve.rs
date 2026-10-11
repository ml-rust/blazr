//! Precedence for each value that more than one source sets.
//!
//! Order: CLI flag, then environment variable, then the user config file,
//! then the built-in default. Each function here is pure. `EnvVars` and
//! `FileConfig` carry the values read from the process and the disk.

use std::path::PathBuf;

use super::{
    DeviceChoice, EnvVars, FileConfig, GenerationConfig, ReasoningFormat, ServerConfig,
    DEFAULT_MODEL_DIR, DEFAULT_NUM_CTX,
};

/// The first set value in precedence order, else `default`.
pub fn pick<T>(flag: Option<T>, env: Option<T>, file: Option<T>, default: T) -> T {
    flag.or(env).or(file).unwrap_or(default)
}

/// Model directory: `flag` (only `pull --output`), `BLAZR_MODEL_DIR`,
/// `model_dir`, then `./models`.
pub fn resolve_model_dir(flag: Option<PathBuf>, env: &EnvVars, file: &FileConfig) -> PathBuf {
    pick(
        flag,
        env.model_dir.clone(),
        file.model_dir.clone(),
        PathBuf::from(DEFAULT_MODEL_DIR),
    )
}

/// Initial KV cache size: `--num-ctx`, then `num_ctx`, then the default.
pub fn resolve_num_ctx(flag: Option<usize>, file: &FileConfig) -> usize {
    pick(flag, None, file.num_ctx, DEFAULT_NUM_CTX)
}

/// The model a command loads: the flag or argument, then `default_model`.
pub fn resolve_model(flag: Option<String>, file: &FileConfig) -> Option<String> {
    flag.or_else(|| file.default_model.clone())
}

/// [`resolve_model`] for a command that needs a model.
///
/// # Errors
///
/// An error when neither source names a model.
pub fn require_model(flag: Option<String>, file: &FileConfig) -> Result<String, String> {
    resolve_model(flag, file).ok_or_else(|| {
        "no model given. Pass a model name or path, or set default_model in \
         ~/.blazr/config.yaml"
            .to_string()
    })
}

/// Device for `run` and `generate`: `--cpu`, then `device`, then CUDA 0.
pub fn resolve_run_device(cpu_flag: bool, file: &FileConfig) -> DeviceChoice {
    pick(
        cpu_flag.then_some(DeviceChoice::Cpu),
        None,
        file.device,
        DeviceChoice::Cuda(0),
    )
}

/// Tensor parallel degree: `BLAZR_TP_SIZE`, then
/// `inference.tensor_parallel_size`. `None` leaves each model's own value.
pub fn resolve_tp_size(env: &EnvVars, file: &FileConfig) -> Option<usize> {
    env.tp_size
        .or_else(|| file.inference.tensor_parallel_size())
}

/// Default reasoning split: `--reasoning-format`, then
/// `server.reasoning_format`, which holds the built-in default when unset.
pub fn resolve_reasoning_format(
    flag: Option<ReasoningFormat>,
    server: &ServerConfig,
) -> ReasoningFormat {
    flag.unwrap_or(server.reasoning_format)
}

/// Sampling flags of `run`, `generate` and `chat`. `None` means not given.
#[derive(Debug, Clone, Default)]
pub struct GenerationFlags {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    /// `--verbose-prompt` or `-v`. A switch: absent leaves the file value.
    pub verbose_prompt: bool,
}

/// Sampling settings: each given flag over the file's `generation`
/// section, which holds the built-in default for each key it omits.
pub fn resolve_generation(flags: &GenerationFlags, file: &GenerationConfig) -> GenerationConfig {
    let mut config = file.clone();
    config.max_tokens = flags.max_tokens.unwrap_or(config.max_tokens);
    config.temperature = flags.temperature.unwrap_or(config.temperature);
    config.top_p = flags.top_p.unwrap_or(config.top_p);
    config.top_k = flags.top_k.unwrap_or(config.top_k);
    config.min_p = flags.min_p.unwrap_or(config.min_p);
    config.repeat_penalty = flags.repeat_penalty.unwrap_or(config.repeat_penalty);
    config.repeat_last_n = flags.repeat_last_n.unwrap_or(config.repeat_last_n);
    config.verbose_prompt = flags.verbose_prompt || config.verbose_prompt;
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InferenceConfig;
    use crate::config::InferenceOverlay;

    #[test]
    fn pick_takes_each_step_in_order() {
        assert_eq!(pick(Some(1), Some(2), Some(3), 4), 1);
        assert_eq!(pick(None, Some(2), Some(3), 4), 2);
        assert_eq!(pick(None, None, Some(3), 4), 3);
        assert_eq!(pick(None::<i32>, None, None, 4), 4);
    }

    fn file_with_model_dir(dir: Option<&str>) -> FileConfig {
        FileConfig {
            model_dir: dir.map(PathBuf::from),
            ..Default::default()
        }
    }

    #[test]
    fn model_dir_flag_then_env_then_file_then_default() {
        let env = EnvVars {
            model_dir: Some(PathBuf::from("/env")),
            ..Default::default()
        };
        let file = file_with_model_dir(Some("/file"));
        let flag = Some(PathBuf::from("/flag"));
        assert_eq!(resolve_model_dir(flag, &env, &file), PathBuf::from("/flag"));
        assert_eq!(resolve_model_dir(None, &env, &file), PathBuf::from("/env"));
        let no_env = EnvVars::default();
        assert_eq!(
            resolve_model_dir(None, &no_env, &file),
            PathBuf::from("/file")
        );
        assert_eq!(
            resolve_model_dir(None, &no_env, &file_with_model_dir(None)),
            PathBuf::from(DEFAULT_MODEL_DIR)
        );
    }

    #[test]
    fn num_ctx_flag_then_file_then_default() {
        let file = FileConfig {
            num_ctx: Some(4096),
            ..Default::default()
        };
        assert_eq!(resolve_num_ctx(Some(512), &file), 512);
        assert_eq!(resolve_num_ctx(None, &file), 4096);
        assert_eq!(
            resolve_num_ctx(None, &FileConfig::default()),
            DEFAULT_NUM_CTX
        );
    }

    #[test]
    fn model_argument_then_default_model() {
        let file = FileConfig {
            default_model: Some("from-file".into()),
            ..Default::default()
        };
        assert_eq!(require_model(Some("arg".into()), &file).unwrap(), "arg");
        assert_eq!(require_model(None, &file).unwrap(), "from-file");
        let err = require_model(None, &FileConfig::default()).unwrap_err();
        assert!(err.contains("default_model"), "{err}");
    }

    #[test]
    fn run_device_cpu_flag_then_file_then_cuda() {
        let file = FileConfig {
            device: Some(DeviceChoice::Cuda(1)),
            ..Default::default()
        };
        assert_eq!(resolve_run_device(true, &file), DeviceChoice::Cpu);
        assert_eq!(resolve_run_device(false, &file), DeviceChoice::Cuda(1));
        assert_eq!(
            resolve_run_device(false, &FileConfig::default()),
            DeviceChoice::Cuda(0)
        );
    }

    #[test]
    fn tp_size_env_then_file_then_unset() {
        let section = InferenceConfig {
            tensor_parallel_size: 4,
            ..Default::default()
        };
        let keys = ["tensor_parallel_size".to_string()].into_iter().collect();
        let file = FileConfig {
            inference: InferenceOverlay::from_section(&section, &keys).unwrap(),
            ..Default::default()
        };
        let env = EnvVars {
            tp_size: Some(2),
            ..Default::default()
        };
        assert_eq!(resolve_tp_size(&env, &file), Some(2));
        assert_eq!(resolve_tp_size(&EnvVars::default(), &file), Some(4));
        assert_eq!(
            resolve_tp_size(&EnvVars::default(), &FileConfig::default()),
            None
        );
    }

    #[test]
    fn reasoning_format_flag_then_file_then_default() {
        let server = ServerConfig {
            reasoning_format: ReasoningFormat::None,
            ..Default::default()
        };
        let flag = Some(ReasoningFormat::Deepseek);
        assert_eq!(
            resolve_reasoning_format(flag, &server),
            ReasoningFormat::Deepseek
        );
        assert_eq!(
            resolve_reasoning_format(None, &server),
            ReasoningFormat::None
        );
        assert_eq!(
            resolve_reasoning_format(None, &ServerConfig::default()),
            ReasoningFormat::Deepseek
        );
    }

    #[test]
    fn generation_flags_then_file_then_default() {
        let file = GenerationConfig {
            temperature: 0.3,
            top_p: 0.8,
            ..Default::default()
        };
        let flags = GenerationFlags {
            temperature: Some(0.0),
            ..Default::default()
        };
        let resolved = resolve_generation(&flags, &file);
        assert_eq!(resolved.temperature, 0.0);
        assert_eq!(resolved.top_p, 0.8);
        let default = GenerationConfig::default();
        assert_eq!(resolved.max_tokens, default.max_tokens);
        assert_eq!(resolved.repeat_penalty, default.repeat_penalty);
        assert!(!resolved.verbose_prompt);

        let verbose = GenerationFlags {
            verbose_prompt: true,
            ..Default::default()
        };
        assert!(resolve_generation(&verbose, &file).verbose_prompt);
    }
}

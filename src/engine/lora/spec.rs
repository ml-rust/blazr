//! Adapter specs: the `<dir>[=<name>]` arguments of `--lora` and
//! `inference.lora`, adapter names, and the `<model>:<name>` ids of the
//! variants they make.

use std::path::{Path, PathBuf};

use super::error::LoraError;

/// The character between a base model name and an adapter name in a
/// variant id.
pub const VARIANT_SEPARATOR: char = ':';

/// The longest adapter name.
const MAX_NAME_LEN: usize = 64;

/// One PEFT LoRA adapter, to apply to a base model as a variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterSpec {
    /// The adapter name: the `<name>` of the variant id `<model>:<name>`.
    pub name: String,
    /// The PEFT adapter directory, or its `.safetensors` file with
    /// `adapter_config.json` beside it.
    pub path: PathBuf,
}

impl AdapterSpec {
    /// The adapter at `path` under `name`.
    ///
    /// # Errors
    ///
    /// [`LoraError::Invalid`] for an empty path or a name that is not
    /// 1 to 64 letters, digits, `-`, `_` or `.`.
    pub fn new(name: &str, path: impl Into<PathBuf>) -> Result<Self, LoraError> {
        let path = path.into();
        if path.as_os_str().is_empty() {
            return Err(LoraError::Invalid(format!(
                "LoRA adapter {name:?} names no adapter path"
            )));
        }
        check_name(name)?;
        Ok(Self {
            name: name.to_string(),
            path,
        })
    }

    /// Parse `<dir>[=<name>]`. Without `=<name>`, the name is the adapter
    /// directory's name; for a `.safetensors` file, the name of the
    /// directory holding it. A text after the last `=` that holds a path
    /// separator is part of the path.
    ///
    /// # Errors
    ///
    /// [`LoraError::Invalid`] when the path is empty, no name follows from
    /// it, or the name is not valid.
    pub fn parse(arg: &str) -> Result<Self, LoraError> {
        let (path, name) = match arg.rsplit_once('=') {
            Some((path, name)) if !name.contains(['/', '\\']) => (path, Some(name)),
            _ => (arg, None),
        };
        let path = PathBuf::from(path);
        let name = match name {
            Some(name) => name.to_string(),
            None => default_name(&path).ok_or_else(|| {
                LoraError::Invalid(format!(
                    "--lora {arg:?}: no adapter name follows from the path. Pass <dir>=<name>"
                ))
            })?,
        };
        Self::new(&name, path).map_err(|e| LoraError::Invalid(format!("--lora {arg:?}: {e}")))
    }

    /// The id of the variant this adapter makes of `base`.
    pub fn variant_id(&self, base: &str) -> String {
        variant_id(base, &self.name)
    }
}

/// The variant id of adapter `adapter` on base model `base`:
/// `<base>:<adapter>`.
pub fn variant_id(base: &str, adapter: &str) -> String {
    format!("{base}{VARIANT_SEPARATOR}{adapter}")
}

/// `(base, adapter)` when `model` has the shape `<base>:<adapter>` with a
/// valid adapter name. The shape alone does not make a variant: a model
/// directory may hold a `:` in its name.
pub fn split_variant(model: &str) -> Option<(&str, &str)> {
    model
        .rsplit_once(VARIANT_SEPARATOR)
        .filter(|(base, adapter)| !base.is_empty() && check_name(adapter).is_ok())
}

/// Check an adapter name: 1 to 64 letters, digits, `-`, `_` or `.`.
pub fn check_name(name: &str) -> Result<(), LoraError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if valid {
        Ok(())
    } else {
        Err(LoraError::Invalid(format!(
            "LoRA adapter name {name:?} is not valid. Use 1 to {MAX_NAME_LEN} letters, digits, \
             '-', '_' or '.'"
        )))
    }
}

/// The adapters to register at start: each `--lora` flag, else the user
/// config file's `inference.lora` list. Flags replace the file list whole.
///
/// # Errors
///
/// [`LoraError::Invalid`] when the file value is not a list of strings,
/// an argument does not parse, or two adapters share a name.
pub fn configured_adapters(
    flags: &[String],
    file: Option<&serde_json::Value>,
) -> Result<Vec<AdapterSpec>, LoraError> {
    let file_args: Vec<String> = match (flags.is_empty(), file) {
        (true, Some(value)) => serde_json::from_value(value.clone()).map_err(|e| {
            LoraError::Invalid(format!(
                "inference.lora: {e}. Set a list of \"<dir>[=<name>]\" strings"
            ))
        })?,
        _ => Vec::new(),
    };
    let args = if flags.is_empty() {
        file_args.as_slice()
    } else {
        flags
    };
    let mut specs: Vec<AdapterSpec> = Vec::with_capacity(args.len());
    for arg in args {
        let spec = AdapterSpec::parse(arg)?;
        if let Some(first) = specs.iter().find(|s| s.name == spec.name) {
            return Err(LoraError::Invalid(format!(
                "two LoRA adapters are named {:?}: {} and {}. Name one with <dir>=<name>",
                spec.name,
                first.path.display(),
                spec.path.display()
            )));
        }
        specs.push(spec);
    }
    Ok(specs)
}

/// The default adapter name of `path`: its directory's name.
fn default_name(path: &Path) -> Option<String> {
    let dir = if path.extension().is_some_and(|e| e == "safetensors") {
        path.parent()?
    } else {
        path
    };
    dir.file_name()?.to_str().map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parsed(arg: &str) -> (String, PathBuf) {
        let spec = AdapterSpec::parse(arg).unwrap_or_else(|e| panic!("{arg}: {e}"));
        (spec.name, spec.path)
    }

    #[test]
    fn parses_dir_with_and_without_a_name() {
        assert_eq!(parsed("/m/kev-4b=kev"), ("kev".into(), "/m/kev-4b".into()));
        assert_eq!(parsed("/m/kev-4b"), ("kev-4b".into(), "/m/kev-4b".into()));
        assert_eq!(parsed("/m/kev-4b/"), ("kev-4b".into(), "/m/kev-4b/".into()));
        assert_eq!(
            parsed("/m/kev/adapter_model.safetensors"),
            ("kev".into(), "/m/kev/adapter_model.safetensors".into())
        );
        // A `=` inside a directory name stays in the path.
        assert_eq!(parsed("/m/a=b/kev"), ("kev".into(), "/m/a=b/kev".into()));
        assert_eq!(parsed("/m/a=b/kev=k2"), ("k2".into(), "/m/a=b/kev".into()));
    }

    #[test]
    fn refuses_bad_names_and_paths() {
        for arg in [
            "=kev",
            "/m/kev=",
            "/m/kev=a:b",
            "/m/kev=a b",
            ".",
            "/m/kev=x?",
        ] {
            let err = AdapterSpec::parse(arg).expect_err(arg);
            assert!(matches!(err, LoraError::Invalid(_)), "{arg}: {err:?}");
        }
        assert!(check_name(&"a".repeat(MAX_NAME_LEN)).is_ok());
        assert!(check_name(&"a".repeat(MAX_NAME_LEN + 1)).is_err());
    }

    #[test]
    fn variant_ids_split_back() {
        let id = variant_id("qwen3.5-4b-base", "kev");
        assert_eq!(id, "qwen3.5-4b-base:kev");
        assert_eq!(split_variant(&id), Some(("qwen3.5-4b-base", "kev")));
        assert_eq!(split_variant("a:b:kev"), Some(("a:b", "kev")));
        assert_eq!(split_variant("plain"), None);
        assert_eq!(split_variant(":kev"), None);
        assert_eq!(split_variant("C:\\models\\m"), None);
    }

    #[test]
    fn flags_replace_the_file_list() {
        let file = json!(["/m/a", "/m/b=bee"]);
        let names = |specs: Vec<AdapterSpec>| specs.into_iter().map(|s| s.name).collect::<Vec<_>>();
        assert_eq!(
            names(configured_adapters(&[], Some(&file)).unwrap()),
            ["a", "bee"]
        );
        let flags = vec!["/m/c".to_string()];
        assert_eq!(
            names(configured_adapters(&flags, Some(&file)).unwrap()),
            ["c"]
        );
        assert!(configured_adapters(&[], None).unwrap().is_empty());
        let err = configured_adapters(&[], Some(&json!("/m/a"))).unwrap_err();
        assert!(err.message().contains("inference.lora"), "{err}");
        let twice = vec!["/x/kev".to_string(), "/y/kev".to_string()];
        let err = configured_adapters(&twice, None).unwrap_err();
        assert!(err.message().contains("two LoRA adapters"), "{err}");
    }
}

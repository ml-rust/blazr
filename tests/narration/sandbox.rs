//! A sandbox for the narration binary: its own config, caches and model
//! folder (links to the real models), a project directory as the working
//! directory, and the binary run in it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use crate::common::{model_fixture, skip_notice};

/// The built binary.
const BLAZR: &str = env!("CARGO_BIN_EXE_blazr");

/// A sandbox, removed when the test ends.
pub struct Sandbox {
    root: tempfile::TempDir,
}

/// What a run printed and returned.
pub struct Run {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Run {
    /// Standard output parsed as the one JSON object `--json` prints.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(self.stdout.trim()).unwrap_or_else(|e| {
            panic!(
                "stdout is not one JSON object ({e}):\n{}\nstderr:\n{}",
                self.stdout, self.stderr
            )
        })
    }
}

impl From<Output> for Run {
    fn from(o: Output) -> Self {
        Self {
            code: o.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
        }
    }
}

/// The Whisper checkpoints: base, and small (the base one when no small
/// one is on disk). `None` with a skip notice when base is missing.
pub fn whisper() -> Option<(PathBuf, PathBuf)> {
    let Some(base) = model_fixture("WHISPER_BASE_DIR", "whisper-base") else {
        skip_notice("Whisper base checkpoint", "WHISPER_BASE_DIR");
        return None;
    };
    let small = model_fixture("WHISPER_SMALL_DIR", "whisper-small").unwrap_or_else(|| {
        eprintln!("WHISPER_SMALL_DIR unset and no whisper-small under BOOSTR_MODELS_DIR: the base checkpoint serves as the small recognizer");
        base.clone()
    });
    Some((base, small))
}

impl Sandbox {
    /// A sandbox whose model folder links `whisper-base`, `whisper-small`
    /// and each of `models` (name, target).
    pub fn new(base: &Path, small: &Path, models: &[(&str, &Path)]) -> Self {
        let root = tempfile::tempdir().unwrap_or_else(|e| panic!("sandbox: {e}"));
        let dir = root.path();
        for sub in ["models", "config/motion-video", "cache", "hf", "project"] {
            std::fs::create_dir_all(dir.join(sub)).unwrap_or_else(|e| panic!("{e}"));
        }
        let link = |name: &str, target: &Path| {
            std::os::unix::fs::symlink(target, dir.join("models").join(name))
                .unwrap_or_else(|e| panic!("linking {name}: {e}"));
        };
        link("whisper-base", base);
        link("whisper-small", small);
        for (name, target) in models {
            link(name, target);
        }
        Self { root }
    }

    /// The working directory of every run.
    pub fn project(&self) -> PathBuf {
        self.root.path().join("project")
    }

    /// Writes the narration config.
    pub fn config(&self, value: &serde_json::Value) {
        let file = self.root.path().join("config/motion-video/config.json");
        std::fs::write(file, value.to_string()).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Writes `text` to `path` under the project.
    pub fn write(&self, path: &str, text: &str) {
        let p = self.project().join(path);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("{e}"));
        }
        std::fs::write(p, text).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Checks the project directory is still empty: a refused run made no
    /// directory and wrote no file in it.
    pub fn assert_untouched(&self) {
        let left: Vec<String> = std::fs::read_dir(self.project())
            .unwrap_or_else(|e| panic!("{e}"))
            .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<Result<_, _>>()
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            left.is_empty(),
            "the refused run left {left:?} in the project"
        );
    }

    /// Reads `path` under the project as text.
    pub fn read(&self, path: &str) -> String {
        std::fs::read_to_string(self.project().join(path))
            .unwrap_or_else(|e| panic!("reading {path}: {e}"))
    }

    /// Runs `blazr` with `args` in the project.
    pub fn run(&self, args: &[&str]) -> Run {
        self.run_with(args, &[])
    }

    /// Runs `blazr` with `args` in the project, with `env` set as well.
    pub fn run_with(&self, args: &[&str], env: &[(&str, &str)]) -> Run {
        let dir = self.root.path();
        Command::new(BLAZR)
            .args(args)
            .envs(env.iter().copied())
            .current_dir(self.project())
            .env("BOOSTR_MODELS_DIR", dir.join("models"))
            .env("XDG_CONFIG_HOME", dir.join("config"))
            .env("XDG_CACHE_HOME", dir.join("cache"))
            .env("HF_HOME", dir.join("hf"))
            .env_remove("HF_HUB_CACHE")
            .env_remove("MOTION_VIDEO_MODELS")
            .output()
            .unwrap_or_else(|e| panic!("running {BLAZR}: {e}"))
            .into()
    }
}

/// The WAV header of `bytes`: channels, rate and bits per sample.
pub fn wav_format(bytes: &[u8]) -> (u16, u32, u16) {
    assert!(bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32_at =
        |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    (u16_at(22), u32_at(24), u16_at(34))
}

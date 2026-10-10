//! A byte progress bar on standard error for one hub file download.

use std::sync::Mutex;

use hf_hub::progress::{DownloadEvent, FileStatus, Progress, ProgressEvent, ProgressHandler};
use indicatif::{ProgressBar, ProgressStyle};

/// Longest file name the bar shows. A longer one keeps its tail.
const NAME_WIDTH: usize = 30;

/// Draws one bar for one file: its name, elapsed time, bytes, rate and
/// time left. The bar appears when the download starts and stays drawn
/// when it ends. A cached file draws nothing.
pub struct BarProgress {
    name: String,
    bar: Mutex<Option<ProgressBar>>,
}

impl BarProgress {
    /// A handler for the repo file `name`, ready for a download builder.
    pub fn progress(name: &str) -> Progress {
        Progress::new(Self {
            name: short_name(name),
            bar: Mutex::new(None),
        })
    }

    fn with_bar(&self, f: impl FnOnce(&mut Option<ProgressBar>)) {
        // A poisoned lock only means an earlier draw panicked. The bar is
        // display state, so keep drawing.
        let mut bar = self.bar.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut bar);
    }
}

impl ProgressHandler for BarProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => self.with_bar(|bar| {
                *bar = Some(new_bar(&self.name, *total_bytes));
            }),
            DownloadEvent::Progress { files } => self.with_bar(|bar| {
                let Some(bar) = bar else { return };
                for file in files {
                    if file.total_bytes > 0 {
                        bar.set_length(file.total_bytes);
                    }
                    bar.set_position(file.bytes_completed);
                    if file.status == FileStatus::Complete {
                        bar.finish();
                    }
                }
            }),
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => self.with_bar(|bar| {
                let Some(bar) = bar else { return };
                if *total_bytes > 0 {
                    bar.set_length(*total_bytes);
                }
                bar.set_position(*bytes_completed);
            }),
            DownloadEvent::Complete => self.with_bar(|bar| {
                if let Some(bar) = bar {
                    bar.finish();
                }
            }),
        }
    }
}

/// A bar of `len` bytes labelled `name`.
fn new_bar(name: &str, len: u64) -> ProgressBar {
    let bar = ProgressBar::new(len);
    if let Ok(style) = ProgressStyle::with_template(
        "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
    ) {
        bar.set_style(style);
    }
    bar.set_message(name.to_string());
    bar
}

/// `name`, or `..` and its last [`NAME_WIDTH`] characters when longer.
fn short_name(name: &str) -> String {
    let count = name.chars().count();
    if count <= NAME_WIDTH {
        return name.to_string();
    }
    let tail: String = name.chars().skip(count - NAME_WIDTH).collect();
    format!("..{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_name_keeps_its_tail() {
        assert_eq!(short_name("config.json"), "config.json");
        let long = "model-00001-of-00004.safetensors";
        assert_eq!(short_name(long), "..del-00001-of-00004.safetensors");
        assert_eq!(short_name(long).chars().count(), NAME_WIDTH + 2);
    }

    #[test]
    fn a_long_name_is_cut_on_a_character_boundary() {
        let name = "é".repeat(40);
        assert_eq!(short_name(&name), format!("..{}", "é".repeat(NAME_WIDTH)));
    }
}

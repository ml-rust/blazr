//! Printing a narration command's lines: each to its stream, or every one
//! to standard error when `--json` keeps standard output for the result.

use voxr_narrate::{Line, Stream};

/// Where a command's lines go.
#[derive(Debug, Clone, Copy)]
pub struct Printer {
    /// `--json`: standard output holds only the JSON object.
    pub json: bool,
}

impl Printer {
    /// Prints each line on its stream.
    pub fn lines(&self, lines: &[Line]) {
        for line in lines {
            self.line(line);
        }
    }

    /// Prints one line on its stream.
    pub fn line(&self, line: &Line) {
        match line.stream {
            Stream::Out if !self.json => println!("{}", line.text),
            _ => eprintln!("{}", line.text),
        }
    }

    /// Prints a report line: standard output, or standard error with
    /// `--json`.
    pub fn out(&self, text: &str) {
        self.line(&Line::out(text));
    }

    /// Prints the one JSON object on standard output.
    pub fn json(&self, value: &serde_json::Value) {
        println!("{value}");
    }
}

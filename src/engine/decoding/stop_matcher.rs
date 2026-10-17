//! Stop sequence matching over a stream of text pieces.
//!
//! Text logic only. A stop can span several pieces or sit inside one piece.
//! The matcher holds back text that can still grow into a stop, so no part
//! of a stop reaches the caller.

/// Result of pushing one piece into a [`StopMatcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopStep {
    /// Text that is safe to send. It never contains part of a stop.
    pub emit: String,
    /// True once a stop matched. Later pushes emit nothing.
    pub stopped: bool,
}

/// Cuts a text stream at the earliest stop sequence.
#[derive(Debug, Clone)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopMatcher {
    /// Create a matcher. Empty stop strings are ignored.
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
            stopped: false,
        }
    }

    /// Append `piece` and return the text that is now safe to send.
    ///
    /// The earliest stop wins. On equal start, the longest stop wins.
    /// Without a stop, the longest suffix that is a proper prefix of a stop
    /// stays held back.
    pub fn push(&mut self, piece: &str) -> StopStep {
        if self.stopped {
            return StopStep {
                emit: String::new(),
                stopped: true,
            };
        }
        self.pending.push_str(piece);

        if let Some(start) = self.earliest_stop() {
            self.pending.truncate(start);
            let emit = std::mem::take(&mut self.pending);
            self.stopped = true;
            return StopStep {
                emit,
                stopped: true,
            };
        }

        let hold = self.held_len();
        let cut = self.pending.len() - hold;
        let emit: String = self.pending.drain(..cut).collect();
        StopStep {
            emit,
            stopped: false,
        }
    }

    /// Return the held-back text when generation ends without a stop.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Start index of the earliest stop in `pending`, longest stop on a tie.
    fn earliest_stop(&self) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for stop in &self.stops {
            if let Some(pos) = self.pending.find(stop.as_str()) {
                let better = match best {
                    None => true,
                    Some((p, len)) => pos < p || (pos == p && stop.len() > len),
                };
                if better {
                    best = Some((pos, stop.len()));
                }
            }
        }
        best.map(|(pos, _)| pos)
    }

    /// Byte length of the longest suffix of `pending` that is a proper
    /// prefix of at least one stop.
    fn held_len(&self) -> usize {
        let mut hold = 0;
        for stop in &self.stops {
            let max = (stop.len() - 1).min(self.pending.len());
            for k in (hold + 1..=max).rev() {
                if !stop.is_char_boundary(k) {
                    continue;
                }
                let start = self.pending.len() - k;
                if !self.pending.is_char_boundary(start) {
                    continue;
                }
                if self.pending[start..] == stop[..k] {
                    hold = k;
                    break;
                }
            }
        }
        hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        StopMatcher::new(&owned)
    }

    /// Feed `pieces`, return the joined emit text and whether a stop hit.
    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = matcher(stops);
        let mut out = String::new();
        let mut stopped = false;
        for p in pieces {
            let step = m.push(p);
            out.push_str(&step.emit);
            if step.stopped {
                stopped = true;
                break;
            }
        }
        if !stopped {
            out.push_str(&m.finish());
        }
        (out, stopped)
    }

    #[test]
    fn stop_split_across_two_tokens() {
        let mut m = matcher(&["END"]);
        let a = m.push("EN");
        assert_eq!((a.emit.as_str(), a.stopped), ("", false));
        let b = m.push("D");
        assert_eq!((b.emit.as_str(), b.stopped), ("", true));
    }

    #[test]
    fn stop_split_across_three_tokens() {
        let (text, stopped) = run(&["END"], &["hi E", "N", "D tail"]);
        assert_eq!(text, "hi ");
        assert!(stopped);
    }

    #[test]
    fn stop_inside_a_token_with_trailing_text() {
        let (text, stopped) = run(&["END"], &["abcENDxyz"]);
        assert_eq!(text, "abc");
        assert!(stopped);
    }

    #[test]
    fn earliest_stop_wins() {
        let (text, stopped) = run(&["two", "one"], &["zero one two"]);
        assert_eq!(text, "zero ");
        assert!(stopped);
    }

    #[test]
    fn longest_stop_wins_on_equal_start() {
        let mut m = matcher(&["ab", "abc"]);
        let step = m.push("xabcd");
        assert_eq!(step.emit, "x");
        assert!(step.stopped);
        let (text, _) = run(&["ab", "abcd"], &["xabcd"]);
        assert_eq!(text, "x");
    }

    #[test]
    fn no_stops_pass_through_byte_for_byte() {
        let mut m = matcher(&[]);
        for piece in ["", "a", " b\n", "世界", "🦀"] {
            let step = m.push(piece);
            assert_eq!(step.emit, piece);
            assert!(!step.stopped);
        }
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn unfinished_prefix_is_flushed_by_finish() {
        let mut m = matcher(&["END"]);
        assert_eq!(m.push("go EN").emit, "go ");
        assert_eq!(m.finish(), "EN");
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn multibyte_stop_split_across_tokens() {
        let (text, stopped) = run(&["世界"], &["你好世", "界!"]);
        assert_eq!(text, "你好");
        assert!(stopped);
    }

    #[test]
    fn multibyte_text_is_never_cut_inside_a_char() {
        let mut m = matcher(&["🦀🦀", "世界"]);
        assert_eq!(m.push("a🦀").emit, "a");
        assert_eq!(m.push("世").emit, "🦀");
        assert_eq!(m.finish(), "世");
        let (text, stopped) = run(&["界x"], &["é世界", "y"]);
        assert_eq!(text, "é世界y");
        assert!(!stopped);
    }

    #[test]
    fn empty_stop_is_ignored() {
        let (text, stopped) = run(&["", "END"], &["a", "bEND"]);
        assert_eq!(text, "ab");
        assert!(stopped);
        let (text, stopped) = run(&[""], &["abc"]);
        assert_eq!(text, "abc");
        assert!(!stopped);
    }

    #[test]
    fn push_after_stop_emits_nothing() {
        let mut m = matcher(&["END"]);
        assert!(m.push("END").stopped);
        let step = m.push("more");
        assert_eq!(step.emit, "");
        assert!(step.stopped);
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn held_back_text_is_the_longest_prefix_suffix() {
        let mut m = matcher(&["abc"]);
        assert_eq!(m.push("xab").emit, "x");
        let step = m.push("d");
        assert_eq!(step.emit, "abd");
        assert!(!step.stopped);
    }

    #[test]
    fn repeated_prefix_characters_hold_correctly() {
        let mut m = matcher(&["aab"]);
        assert_eq!(m.push("aaa").emit, "a");
        // "aaab" cuts at index 1; the first push already emitted that "a".
        let step = m.push("b");
        assert_eq!(step.emit, "");
        assert!(step.stopped);
    }
}

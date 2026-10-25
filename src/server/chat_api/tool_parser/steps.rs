//! The scan states of [`ToolCallParser`]: plain text, a `<tool_call>` block and a bare JSON object.

use super::parser::{Candidate, State, ToolCallParser, ToolEvents, TAG_CLOSE, TAG_OPEN};
use crate::server::json_scan::{first_key, FirstKey};
use crate::server::tag_prefix::partial_prefix_len;
use crate::server::tool_calls::{is_declared, parse_tool_call_json, parse_tool_call_xml};

impl ToolCallParser {
    /// Release text up to the next trigger and enter its candidate state.
    /// Returns true when a candidate starts.
    pub(super) fn scan_text(&mut self, out: &mut ToolEvents, end: bool) -> bool {
        let buf = std::mem::take(&mut self.held);
        let mut line_start = self.line_start;
        let mut space_from = None;
        for (i, ch) in buf.char_indices() {
            let trigger = if ch == '<' && buf[i..].starts_with(TAG_OPEN) {
                Some(State::InTag)
            } else if ch == '{' && line_start {
                Some(State::InBare)
            } else {
                None
            };
            if let Some(state) = trigger {
                let gap_from = space_from.unwrap_or(i);
                self.release(&buf[..gap_from], out);
                self.candidate = Candidate {
                    gap: buf[gap_from..i].to_string(),
                    ..Candidate::default()
                };
                self.held = buf[i..].to_string();
                self.state = state;
                return true;
            }
            if ch.is_whitespace() {
                space_from.get_or_insert(i);
                line_start |= ch == '\n';
            } else {
                space_from = None;
                line_start = false;
            }
        }
        if end {
            self.release(&buf, out);
            return false;
        }
        let prefix_from = buf.len() - partial_prefix_len(&buf, TAG_OPEN);
        let keep_from = buf[..prefix_from].trim_end().len();
        self.release(&buf[..keep_from], out);
        self.held = buf[keep_from..].to_string();
        false
    }

    /// Step a `<tool_call>` block. Returns true when it resolves.
    pub(super) fn step_tag(&mut self, out: &mut ToolEvents, end: bool) -> bool {
        let open = TAG_OPEN.len();
        let body = self.held[open..].trim_start();
        let body_at = self.held.len() - body.len();
        let search_from = if body.starts_with('{') {
            // A JSON body can carry `</tool_call>` inside a string, so the
            // close tag counts only after the object ends.
            match self.candidate.scan.advance(&self.held[body_at..]) {
                Some(len) => body_at + len,
                None => return self.unresolved(open, out, end),
            }
        } else if body.is_empty() {
            return self.unresolved(open, out, end);
        } else {
            open
        };
        let from = search_from.max(self.candidate.searched);
        let Some(at) = self.held[from..].find(TAG_CLOSE) else {
            let mut next = self
                .held
                .len()
                .saturating_sub(TAG_CLOSE.len() - 1)
                .max(from);
            while !self.held.is_char_boundary(next) {
                next -= 1;
            }
            self.candidate.searched = next;
            return self.unresolved(open, out, end);
        };
        let close = from + at;
        let block_end = close + TAG_CLOSE.len();
        let inner = self.held[open..close].trim();
        let call = parse_tool_call_json(inner).or_else(|| parse_tool_call_xml(inner));
        match call.filter(|call| is_declared(&self.declared, call)) {
            Some(call) => self.emit(call, block_end, out),
            None => self.reject(block_end, out),
        }
        true
    }

    /// Step a bare JSON object. Returns true when it resolves.
    pub(super) fn step_bare(&mut self, out: &mut ToolEvents, end: bool) -> bool {
        if !self.candidate.key_ok {
            match first_key(&self.held) {
                FirstKey::Other => {
                    self.reject(1, out);
                    return true;
                }
                FirstKey::Pending => return self.unresolved(1, out, end),
                FirstKey::CallKey => self.candidate.key_ok = true,
            }
        }
        let Some(len) = self.candidate.scan.advance(&self.held) else {
            return self.unresolved(1, out, end);
        };
        let call = parse_tool_call_json(&self.held[..len]);
        match call.filter(|call| is_declared(&self.declared, call)) {
            Some(call) => self.emit(call, len, out),
            // Only the `{` goes out, so a call on a later line still parses.
            None => self.reject(1, out),
        }
        true
    }

    /// Hold a candidate that needs more input. At the end of the output,
    /// release its first `trigger_len` bytes and scan on. Returns true when
    /// the candidate resolves.
    fn unresolved(&mut self, trigger_len: usize, out: &mut ToolEvents, end: bool) -> bool {
        if end {
            self.reject(trigger_len, out);
        }
        end
    }
}

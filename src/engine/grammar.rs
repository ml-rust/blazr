//! Grammar-constrained generation (GBNF + JSON Schema)
//!
//! Compiles GBNF grammars to a DFA that masks logits during sampling,
//! ensuring generated text conforms to the grammar.
//! Also supports converting JSON schemas to GBNF.

use std::collections::{HashMap, HashSet, VecDeque};

use boostr::ops::traits::inference::grammar::INVALID_STATE;
use boostr::{DeviceGrammarDfa, Runtime, Tensor};

pub use super::grammar_json::json_schema_to_gbnf;
pub use super::grammar_parser::{
    parse_gbnf, CharRange, GbnfElement, GbnfRule, GbnfSequence, RepeatKind,
};

use super::grammar_parser::GbnfElement as Element;

/// A compiled DFA for grammar-constrained generation
#[derive(Debug, Clone)]
pub struct GrammarDfa {
    /// State transitions: state_id -> [(byte, next_state)]
    transitions: Vec<HashMap<u8, usize>>,
    /// Which states are accepting
    accepting: HashSet<usize>,
    /// Current state
    current_state: usize,
}

impl GrammarDfa {
    /// Get the set of valid next bytes from the current state
    pub fn valid_next_bytes(&self) -> &HashMap<u8, usize> {
        &self.transitions[self.current_state]
    }

    /// Advance the DFA by one byte. Returns true if the transition was valid.
    pub fn advance(&mut self, byte: u8) -> bool {
        if let Some(&next) = self.transitions[self.current_state].get(&byte) {
            self.current_state = next;
            true
        } else {
            false
        }
    }

    /// Check if the current state is accepting
    pub fn is_accepting(&self) -> bool {
        self.accepting.contains(&self.current_state)
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.current_state = 0;
    }

    /// Get the current DFA state
    pub fn current_state(&self) -> usize {
        self.current_state
    }

    /// Get the number of states
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }

    /// Create a vocabulary mask: for each token in the vocabulary, check if
    /// any of its byte representations lead to a valid DFA transition.
    /// Returns a boolean mask (true = allowed).
    pub fn compute_token_mask(&self, vocab: &[Vec<u8>]) -> Vec<bool> {
        vocab
            .iter()
            .map(|token_bytes| {
                let mut state = self.current_state;
                for &byte in token_bytes {
                    if let Some(&next) = self.transitions[state].get(&byte) {
                        state = next;
                    } else {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Convert this CPU-side DFA into a device-resident DFA for on-device masking.
    ///
    /// Flattens the transition table to `[num_states * 256]` i32 and builds
    /// concatenated vocab byte buffers with offsets.
    pub fn to_device<R: Runtime<DType = boostr::DType>>(
        &self,
        vocab_bytes_list: &[Vec<u8>],
        device: &R::Device,
    ) -> DeviceGrammarDfa<R> {
        let num_states = self.transitions.len();
        let vocab_size = vocab_bytes_list.len();

        // Flatten transition table: [num_states * 256], INVALID_STATE for missing
        let mut table = vec![INVALID_STATE as f32; num_states * 256];
        for (state, trans_map) in self.transitions.iter().enumerate() {
            for (&byte, &next_state) in trans_map {
                table[state * 256 + byte as usize] = next_state as f32;
            }
        }

        // Accepting mask: [num_states]
        let mut accepting = vec![0.0f32; num_states];
        for &state in &self.accepting {
            if state < num_states {
                accepting[state] = 1.0;
            }
        }

        // Concatenate vocab bytes and build offsets
        let mut all_bytes: Vec<f32> = Vec::new();
        let mut offsets: Vec<f32> = Vec::with_capacity(vocab_size + 1);
        for token_bytes in vocab_bytes_list {
            offsets.push(all_bytes.len() as f32);
            for &b in token_bytes {
                all_bytes.push(b as f32);
            }
        }
        offsets.push(all_bytes.len() as f32);

        // Handle empty vocab_bytes (need at least one element for tensor)
        if all_bytes.is_empty() {
            all_bytes.push(0.0);
        }

        DeviceGrammarDfa {
            transition_table: Tensor::from_slice(&table, &[num_states * 256], device),
            accepting_mask: Tensor::from_slice(&accepting, &[num_states], device),
            vocab_bytes: Tensor::from_slice(&all_bytes, &[all_bytes.len()], device),
            vocab_offsets: Tensor::from_slice(&offsets, &[offsets.len()], device),
            current_state: self.current_state as u32,
            num_states,
            vocab_size,
        }
    }

    /// Apply grammar mask to logits: set disallowed tokens to -inf
    pub fn mask_logits(&self, logits: &mut [f32], vocab: &[Vec<u8>]) {
        for (i, token_bytes) in vocab.iter().enumerate() {
            let mut state = self.current_state;
            let mut valid = true;
            for &byte in token_bytes {
                if let Some(&next) = self.transitions[state].get(&byte) {
                    state = next;
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Compile a GBNF grammar into a DFA for token masking.
///
/// This is a simplified NFA→DFA compiler that handles the common
/// grammar patterns used in structured output (JSON, function calls).
pub fn compile_grammar_to_dfa(gbnf: &str) -> Result<GrammarDfa, String> {
    let rules = parse_gbnf(gbnf)?;

    // Build NFA from rules
    let mut nfa_transitions: Vec<HashMap<u8, Vec<usize>>> = Vec::new();
    let mut nfa_accepting = HashSet::new();

    // State 0 is the start state
    nfa_transitions.push(HashMap::new());

    // For the simplified compiler, expand the root rule's literals into NFA states
    if let Some(root) = rules.iter().find(|r| r.name == "root") {
        let state = 0;
        for alt in &root.alternatives {
            let mut current_state = state;
            for element in alt {
                match element {
                    Element::Literal(s) => {
                        for byte in s.bytes() {
                            let next_state = nfa_transitions.len();
                            nfa_transitions.push(HashMap::new());
                            nfa_transitions[current_state]
                                .entry(byte)
                                .or_default()
                                .push(next_state);
                            current_state = next_state;
                        }
                    }
                    Element::CharClass(ranges) => {
                        let next_state = nfa_transitions.len();
                        nfa_transitions.push(HashMap::new());
                        for range in ranges {
                            for byte in range.start..=range.end {
                                nfa_transitions[current_state]
                                    .entry(byte)
                                    .or_default()
                                    .push(next_state);
                            }
                        }
                        current_state = next_state;
                    }
                    _ => {
                        // For rule references and complex elements, allow all bytes
                        // (full grammar support would require recursive expansion)
                        let next_state = nfa_transitions.len();
                        nfa_transitions.push(HashMap::new());
                        for byte in 0u8..=127 {
                            nfa_transitions[current_state]
                                .entry(byte)
                                .or_default()
                                .push(next_state);
                        }
                        current_state = next_state;
                    }
                }
            }
            nfa_accepting.insert(current_state);
        }
    }

    // Convert NFA to DFA via subset construction
    let mut dfa_transitions: Vec<HashMap<u8, usize>> = Vec::new();
    let mut dfa_accepting = HashSet::new();
    let mut state_map: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut work_queue: VecDeque<Vec<usize>> = VecDeque::new();

    let initial = vec![0usize];
    state_map.insert(initial.clone(), 0);
    dfa_transitions.push(HashMap::new());
    work_queue.push_back(initial);

    while let Some(nfa_states) = work_queue.pop_front() {
        let dfa_state = state_map[&nfa_states];

        // Check if this DFA state is accepting
        if nfa_states.iter().any(|s| nfa_accepting.contains(s)) {
            dfa_accepting.insert(dfa_state);
        }

        // Collect all possible transitions
        let mut byte_targets: HashMap<u8, Vec<usize>> = HashMap::new();
        for &nfa_state in &nfa_states {
            if nfa_state < nfa_transitions.len() {
                for (&byte, targets) in &nfa_transitions[nfa_state] {
                    byte_targets.entry(byte).or_default().extend(targets);
                }
            }
        }

        for (byte, mut targets) in byte_targets {
            targets.sort_unstable();
            targets.dedup();

            let target_dfa_state = if let Some(&existing) = state_map.get(&targets) {
                existing
            } else {
                let new_state = dfa_transitions.len();
                dfa_transitions.push(HashMap::new());
                state_map.insert(targets.clone(), new_state);
                work_queue.push_back(targets);
                new_state
            };

            dfa_transitions[dfa_state].insert(byte, target_dfa_state);
        }
    }

    Ok(GrammarDfa {
        transitions: dfa_transitions,
        accepting: dfa_accepting,
        current_state: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_gbnf() {
        let grammar = r#"root ::= "hello" | "world""#;
        let rules = parse_gbnf(grammar).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "root");
        assert_eq!(rules[0].alternatives.len(), 2);
    }

    #[test]
    fn test_compile_literal_dfa() {
        let grammar = r#"root ::= "yes" | "no""#;
        let dfa = compile_grammar_to_dfa(grammar).unwrap();
        assert!(dfa.num_states() > 0);
    }

    #[test]
    fn test_json_schema_to_gbnf() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        let gbnf = json_schema_to_gbnf(&schema).unwrap();
        assert!(gbnf.contains("root"));
        assert!(gbnf.contains("object"));
    }
}

//! The ids that end generation for one checkpoint.
//!
//! A `tokenizer.json` names no EOS. splintr guesses one by spelling. That guess
//! is often not the id a chat turn ends on:
//!
//! - Qwen3 ends a turn on `<|im_end|>`. The guess is `<|endoftext|>`.
//! - Phi-3.5 ends a turn on `<|end|>`. The guess is `</s>`, which it never emits.
//! - Gemma 2 ends on `<eos>` or `<end_of_turn>`. The guess is `</s>`, which is
//!   its HTML strikethrough tag, id 213.
//!
//! A checkpoint states the truth in two places:
//!
//! - `tokenizer_config.json` `eos_token`
//! - `eos_token_id` of `generation_config.json` and `config.json`, which can
//!   list several ids
//!
//! A template's end-of-turn marker can be absent from all of them. `<|eot_id|>`
//! is absent in Llama 3.2 1B. So a marker the vocabulary names also stops
//! generation. llama.cpp keeps the same list.

use serde_json::Value;
use splintr::AnyTokenizer;

use super::bos::{token_content, CheckpointTokenizerFiles};

/// End-of-turn markers that stop generation when the vocabulary names them.
const TURN_END_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end|>",
    "<end_of_turn>",
    "<|END_OF_TURN_TOKEN|>",
];

/// The ids an `eos_token_id` field states: one id or a list.
fn stated_ids(config: Option<&Value>) -> Vec<u32> {
    let Some(config) = config else {
        return Vec::new();
    };
    let field = config
        .get("eos_token_id")
        .or_else(|| config.get("text_config")?.get("eos_token_id"));
    let values = match field {
        Some(Value::Array(ids)) => ids.iter().collect(),
        Some(id) => vec![id],
        None => Vec::new(),
    };
    values
        .into_iter()
        .filter_map(Value::as_u64)
        .filter_map(|id| u32::try_from(id).ok())
        .collect()
}

/// Every id that ends generation under `files`, the primary EOS first:
///
/// 1. the token `eos_token` names
/// 2. every id `generation_config.json` and `config.json` state
/// 3. the vocabulary's own EOS, only when steps 1 and 2 state none
/// 4. every end-of-turn marker the vocabulary names
///
/// A stated id counts whether or not it names an added token, as in
/// `transformers`. Falcon-RW 1B states id 2, the ordinary token `#`, so its text
/// ends at the first `#`, and `transformers` ends there too.
pub fn stop_ids(tokenizer: &AnyTokenizer, files: &CheckpointTokenizerFiles) -> Vec<u32> {
    let mut ids: Vec<u32> = files
        .tokenizer_config
        .as_ref()
        .and_then(|c| token_content(c.get("eos_token")))
        .and_then(|name| tokenizer.special_token_id(&name))
        .into_iter()
        .collect();
    let add = |ids: &mut Vec<u32>, id: u32| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };
    let stated = stated_ids(files.generation_config.as_ref())
        .into_iter()
        .chain(stated_ids(files.config.as_ref()));
    for id in stated {
        add(&mut ids, id);
    }
    // A bundled vocabulary or a GGUF knows its EOS. A `tokenizer.json` guess is
    // a spelling, so it counts only when no file states an id.
    if ids.is_empty() {
        for id in tokenizer.eos_token_ids() {
            add(&mut ids, *id);
        }
    }
    let turn_ends = TURN_END_TOKENS
        .iter()
        .filter_map(|name| tokenizer.special_token_id(name));
    for id in turn_ends {
        add(&mut ids, id);
    }
    ids
}

/// `tokenizer` with the stop ids that `files` state.
pub fn apply_stop_files(tokenizer: AnyTokenizer, files: &CheckpointTokenizerFiles) -> AnyTokenizer {
    let ids = stop_ids(&tokenizer, files);
    tokenizer.with_eos_ids(ids)
}

/// `tokenizer` with the end-of-turn markers its vocabulary names added to its
/// EOS. For a source with no checkpoint files, such as a GGUF.
pub fn with_turn_end_stops(tokenizer: AnyTokenizer) -> AnyTokenizer {
    apply_stop_files(tokenizer, &CheckpointTokenizerFiles::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A byte-level tokenizer with the added tokens `names`, ids from 10.
    fn tokenizer(names: &[&str]) -> AnyTokenizer {
        let added: Vec<Value> = names
            .iter()
            .enumerate()
            .map(|(i, name)| json!({"id": 10 + i, "content": name, "special": true}))
            .collect();
        let file = json!({
            "added_tokens": added,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        });
        splintr::from_json_bytes(file.to_string().as_bytes()).expect("tokenizer loads")
    }

    fn files(
        tokenizer_config: Value,
        generation_config: Value,
        config: Value,
    ) -> CheckpointTokenizerFiles {
        CheckpointTokenizerFiles {
            tokenizer_config: Some(tokenizer_config),
            tokenizer_json: None,
            config: Some(config),
            generation_config: Some(generation_config),
        }
    }

    /// Qwen3: `eos_token` is `<|im_end|>`, and `generation_config.json` lists
    /// both. splintr's guess is `<|endoftext|>` alone.
    #[test]
    fn qwen3_stops_on_im_end_and_endoftext() {
        let tok = tokenizer(&["<|endoftext|>", "<|im_start|>", "<|im_end|>"]);
        assert_eq!(tok.eos_token_id(), Some(10), "the guess");
        let f = files(
            json!({"eos_token": "<|im_end|>"}),
            json!({"eos_token_id": [12, 10]}),
            json!({"eos_token_id": 12}),
        );
        assert_eq!(stop_ids(&tok, &f), [12, 10]);
        let tok = apply_stop_files(tok, &f);
        assert_eq!(tok.eos_token_id(), Some(12));
        assert!(tok.is_eos(12) && tok.is_eos(10));
        assert!(!tok.is_eos(11));
    }

    /// Phi-3.5 lists three ids. The `</s>` that wins the guess is dropped.
    #[test]
    fn phi3_stops_on_every_listed_id() {
        let tok = tokenizer(&["</s>", "<|endoftext|>", "<|assistant|>", "<|end|>"]);
        let f = files(
            json!({"eos_token": "<|endoftext|>"}),
            json!({"eos_token_id": [13, 12, 11]}),
            json!({"eos_token_id": 11}),
        );
        assert_eq!(stop_ids(&tok, &f), [11, 13, 12]);
    }

    /// Gemma 2 states `<eos>` only. Its template ends a turn on `<end_of_turn>`.
    #[test]
    fn a_turn_marker_stops_when_no_file_states_it() {
        let tok = tokenizer(&["<eos>", "<start_of_turn>", "<end_of_turn>", "</s>"]);
        assert_eq!(tok.eos_token_id(), Some(13), "the guess is the HTML tag");
        let f = files(
            json!({"eos_token": "<eos>"}),
            json!({"eos_token_id": 10}),
            json!({"eos_token_id": 10}),
        );
        assert_eq!(stop_ids(&tok, &f), [10, 12]);
    }

    /// Falcon-RW 1B states the id of a plain token, and `transformers` stops on
    /// it. The token `eos_token` names stays primary.
    #[test]
    fn a_stated_id_that_names_no_added_token_stops_beside_a_named_one() {
        let tok = tokenizer(&["<|endoftext|>"]);
        let f = files(
            json!({"eos_token": "<|endoftext|>"}),
            json!({"eos_token_id": 2}),
            json!({"eos_token_id": 2}),
        );
        assert_eq!(stop_ids(&tok, &f), [10, 2]);
    }

    /// A stated id serves when no name states an EOS.
    #[test]
    fn a_stated_id_serves_when_nothing_names_an_eos() {
        let tok = tokenizer(&[]);
        let f = files(json!({}), json!({"eos_token_id": 2}), json!({}));
        assert_eq!(stop_ids(&tok, &f), [2]);
    }

    /// A stated id beats splintr's guess, even when it names no added token.
    #[test]
    fn a_stated_id_beats_the_guess() {
        let tok = tokenizer(&["</s>"]);
        assert_eq!(tok.eos_token_id(), Some(10), "the guess");
        let f = files(json!({}), json!({"eos_token_id": 2}), json!({}));
        assert_eq!(stop_ids(&tok, &f), [2]);
    }

    /// A turn marker in the vocabulary joins a stated id and never replaces it.
    #[test]
    fn a_stated_id_survives_a_turn_marker() {
        let tok = tokenizer(&["<|im_end|>"]);
        let f = files(json!({}), json!({"eos_token_id": 2}), json!({}));
        assert_eq!(stop_ids(&tok, &f), [2, 10]);
    }

    /// A multimodal `config.json` nests the id under `text_config`.
    #[test]
    fn a_nested_config_id_counts() {
        let tok = tokenizer(&["<custom_end>"]);
        let f = files(
            json!({}),
            json!({}),
            json!({"text_config": {"eos_token_id": 10}}),
        );
        assert_eq!(stop_ids(&tok, &f), [10]);
    }

    /// No checkpoint file, as for a GGUF: the vocabulary's own EOS stays
    /// primary, and its turn marker joins it.
    #[test]
    fn with_no_files_the_turn_marker_joins_the_vocabulary_eos() {
        let tok = with_turn_end_stops(tokenizer(&["</s>", "<|im_end|>"]));
        assert_eq!(tok.eos_token_ids(), [10, 11]);
    }
}

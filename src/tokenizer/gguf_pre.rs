//! `tokenizer.ggml.pre` names splintr does not accept, mapped to ones it does.

use splintr::GgufVocab;

/// `tokenizer.ggml.pre` values rewritten before `splintr::from_gguf_vocab`.
///
/// `qwen35` (Bonsai 2) names a byte-level BPE split that splintr has no
/// constant for, so it refuses the vocabulary. Qwen 3.5 ships Qwen2's
/// vocabulary and merge list, and its split differs from `qwen2` only in
/// digit-run handling, so `qwen2` is the closest accepted pattern.
const PRE_ALIASES: &[(&str, &str)] = &[("qwen35", "qwen2")];

/// Rewrite `vocab.pre` through [`PRE_ALIASES`]. Other values pass through.
pub fn map_gguf_pre_tokenizer(vocab: &mut GgufVocab) {
    let Some(pre) = vocab.pre.as_deref() else {
        return;
    };
    if let Some((_, target)) = PRE_ALIASES.iter().find(|(from, _)| *from == pre) {
        tracing::info!(from = pre, to = *target, "Mapped tokenizer.ggml.pre");
        vocab.pre = Some((*target).to_owned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(pre: Option<&str>) -> GgufVocab {
        GgufVocab {
            model: "gpt2".to_owned(),
            pre: pre.map(str::to_owned),
            ..GgufVocab::default()
        }
    }

    #[test]
    fn qwen35_maps_to_qwen2() {
        let mut v = vocab(Some("qwen35"));
        map_gguf_pre_tokenizer(&mut v);
        assert_eq!(v.pre.as_deref(), Some("qwen2"));
    }

    #[test]
    fn other_values_pass_through() {
        let mut v = vocab(Some("llama3"));
        map_gguf_pre_tokenizer(&mut v);
        assert_eq!(v.pre.as_deref(), Some("llama3"));

        let mut v = vocab(None);
        map_gguf_pre_tokenizer(&mut v);
        assert!(v.pre.is_none());
    }
}

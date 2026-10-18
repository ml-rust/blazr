//! The fill-in-the-middle prompt that `/v1/infill` and `/v1/completions` share.

use std::sync::LazyLock;

/// FIM token names used across different model families.
pub(super) const FIM_PREFIX_TOKEN: &str = "<|fim_prefix|>";
pub(super) const FIM_MIDDLE_TOKEN: &str = "<|fim_middle|>";
pub(super) const FIM_SUFFIX_TOKEN: &str = "<|fim_suffix|>";

/// The only special tokens a FIM prompt inserts itself.
///
/// A caller that spells any other control token in its text is refused,
/// not promoted to that token's id.
pub(super) static FIM_MARKERS: LazyLock<splintr::FxHashSet<String>> = LazyLock::new(|| {
    [FIM_PREFIX_TOKEN, FIM_MIDDLE_TOKEN, FIM_SUFFIX_TOKEN]
        .iter()
        .map(|t| (*t).to_string())
        .collect()
});

/// The FIM markers `tokenizer` does not define, in prefix, suffix, middle order.
fn missing_markers(tokenizer: &splintr::AnyTokenizer) -> Vec<&'static str> {
    [FIM_PREFIX_TOKEN, FIM_SUFFIX_TOKEN, FIM_MIDDLE_TOKEN]
        .into_iter()
        .filter(|t| tokenizer.special_token_id(t).is_none())
        .collect()
}

/// Build a FIM prompt from prefix and suffix using the model's FIM markers.
///
/// PSM order (prefix-suffix-middle): `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>`.
/// StarCoder, CodeLlama, and OpenAI models use this order.
///
/// The markers go in as their literal spellings, which is exactly what the
/// allow-list names, so the encode turns them into the ids `special_token_id` reports.
///
/// # Errors
///
/// The tokenizer lacks a marker. The error names every missing marker.
pub(super) fn build_fim_prompt(
    tokenizer: &splintr::AnyTokenizer,
    prefix: &str,
    suffix: &str,
) -> Result<String, String> {
    let missing = missing_markers(tokenizer);
    if !missing.is_empty() {
        return Err(format!(
            "Model tokenizer does not support FIM tokens. Missing: {}",
            missing.join(", ")
        ));
    }
    Ok(format!(
        "{FIM_PREFIX_TOKEN}{prefix}{FIM_SUFFIX_TOKEN}{suffix}{FIM_MIDDLE_TOKEN}"
    ))
}

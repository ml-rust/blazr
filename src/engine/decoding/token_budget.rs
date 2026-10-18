//! How many tokens one request decodes.
//!
//! The single-sequence loop and the batching scheduler both cap a request's
//! `max_tokens` here, so a request ends at the same token on either path.

/// The tokens a request with `prompt_len` prompt ids decodes.
///
/// The answer is `min(max_tokens, context_len - prompt_len)`, and 0 when the
/// prompt fills the context. `context_len` is the most tokens one sequence
/// can hold: prompt plus output.
pub fn token_budget(max_tokens: usize, context_len: usize, prompt_len: usize) -> usize {
    max_tokens.min(context_len.saturating_sub(prompt_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_the_request_or_the_room_left() {
        assert_eq!(token_budget(8, 16, 4), 8);
        assert_eq!(token_budget(100, 16, 12), 4);
        assert_eq!(token_budget(5, 16, 16), 0);
        assert_eq!(token_budget(5, 16, 20), 0);
        assert_eq!(token_budget(0, 16, 1), 0);
    }
}

//! Non-streaming completions: run every prompt `n` times and collect the choices.

use super::completion_prompt::ResolvedPrompt;
use super::completion_types::CompletionChoice;
use crate::config::GenerationConfig;
use crate::server::generation::{choice_config, convert_logprobs};
use crate::server::handlers::Route;
use crate::server::scheduled::generate_text_routed;

/// The `index` of choice `choice` of prompt `prompt`, with `n` choices per prompt.
pub(super) fn choice_index(prompt: usize, n: usize, choice: usize) -> usize {
    prompt * n + choice
}

/// The choices of a request and the totals behind its `usage`.
pub(super) struct Generated {
    pub choices: Vec<CompletionChoice>,
    /// Prompt tokens, counted once per prompt.
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prefill time of the first generation.
    pub first_prompt_eval_ms: u64,
}

/// Run each prompt `n` times in request order.
///
/// Choice `i` samples with `seed + i` when the request sets a seed.
/// `echo` puts the prompt text before each choice.
///
/// # Errors
///
/// The first generation error ends the request.
pub(super) async fn generate_choices(
    route: &Route,
    prompts: &[ResolvedPrompt],
    n: usize,
    echo: bool,
    config: &GenerationConfig,
) -> anyhow::Result<Generated> {
    let mut out = Generated {
        choices: Vec::with_capacity(prompts.len() * n),
        prompt_tokens: 0,
        completion_tokens: 0,
        first_prompt_eval_ms: 0,
    };
    for (p, prompt) in prompts.iter().enumerate() {
        for j in 0..n {
            let index = choice_index(p, n, j);
            let iter_config = choice_config(config, index);
            // The route's batching scheduler when it has one, else its executor.
            let result = generate_text_routed(route, &prompt.tokens, &iter_config).await?;
            if index == 0 {
                out.first_prompt_eval_ms = result.prompt_eval_duration_ms;
            }
            if j == 0 {
                out.prompt_tokens += result.prompt_tokens;
            }
            out.completion_tokens += result.completion_tokens;
            let text = if echo {
                format!("{}{}", prompt.echo, result.text)
            } else {
                result.text
            };
            out.choices.push(CompletionChoice {
                text,
                index,
                finish_reason: result.finish_reason.as_str().to_string(),
                logprobs: result
                    .token_logprobs
                    .as_ref()
                    .map(|tl| convert_logprobs(tl)),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_index_is_prompt_times_n_plus_choice() {
        assert_eq!(choice_index(0, 1, 0), 0);
        assert_eq!(choice_index(2, 1, 0), 2);
        assert_eq!(choice_index(0, 3, 2), 2);
        assert_eq!(choice_index(1, 3, 0), 3);
        assert_eq!(choice_index(2, 3, 2), 8);
    }

    #[test]
    fn indexes_cover_zero_to_total_once() {
        let (prompts, n) = (3, 4);
        let mut seen: Vec<usize> = (0..prompts)
            .flat_map(|p| (0..n).map(move |j| choice_index(p, n, j)))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..prompts * n).collect::<Vec<_>>());
    }
}

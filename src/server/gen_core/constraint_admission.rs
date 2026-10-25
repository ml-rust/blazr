//! Request admission: compile the constraint once and resolve DRY, before generation.

use splintr::AnyTokenizer;

use crate::config::{CompiledConstraint, Constraint, GenerationConfig};
use crate::engine::dry_params;
use crate::server::handlers::AppState;

/// Set `gen_config`'s constraint and compile it through the shared cache.
/// Then check the request's DRY fields against `tokenizer`.
///
/// `requested` is the request's own constraint from `resolve_constraint`.
/// It wins over the live `generation.grammar` default already in
/// `gen_config`. The compiled DFA goes into `gen_config.compiled`, so the
/// sampler compiles nothing and the request compiles at most once.
///
/// # Errors
///
/// The error string of a request the handler answers 400: an ambiguous
/// constraint, a grammar or schema that does not compile, or an invalid DRY
/// breaker or field.
pub async fn admit_generation(
    state: &AppState,
    gen_config: &mut GenerationConfig,
    requested: Result<Option<Constraint>, String>,
    tokenizer: &AnyTokenizer,
) -> Result<(), String> {
    admit_constraint(state, gen_config, requested).await?;
    dry_params(tokenizer, gen_config).map(drop)
}

async fn admit_constraint(
    state: &AppState,
    gen_config: &mut GenerationConfig,
    requested: Result<Option<Constraint>, String>,
) -> Result<(), String> {
    if let Some(constraint) = requested? {
        gen_config.constraint = Some(constraint);
    }
    gen_config.compiled = None;
    let Some(constraint) = gen_config.constraint.clone() else {
        return Ok(());
    };
    let dfa = state
        .grammar_cache
        .get_or_compile_async(constraint.clone())
        .await
        .map_err(|e| request_error(&format!("{e:#}")))?;
    gen_config.compiled = Some(CompiledConstraint::new(constraint, dfa));
    Ok(())
}

/// The 400 text for a compile error: one period, then the next action.
fn request_error(message: &str) -> String {
    let stop = if message.ends_with('.') { "" } else { "." };
    format!("{message}{stop} Correct the grammar or schema")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_without_a_period_gets_one() {
        assert_eq!(
            request_error("rule cycle: a -> a"),
            "rule cycle: a -> a. Correct the grammar or schema"
        );
    }

    #[test]
    fn a_message_with_a_period_keeps_a_single_one() {
        let text = request_error("group has no closing `)`.");
        assert_eq!(
            text,
            "group has no closing `)`. Correct the grammar or schema"
        );
        assert!(!text.contains(".."));
    }
}

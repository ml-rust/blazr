//! AINode's question translation and prompt text, ported from the copy the
//! Jebadiah models were trained against: `scripts/ainode_prompt_verbatim.py`
//! (AINode `ainode/api/decide.py` and `ainode/api/systemone.py`, commit
//! [`PROMPT_SOURCE_COMMIT`]), plus the option checks of AINode's
//! `normalize_questions` that `/v1/systemone` runs on every translated
//! question.
//!
//! Error messages are AINode's, word for word.

use serde_json::{Map, Value};

use super::encode::PromptText;
use super::labels::option_label;
use super::py_text::{py_repr, py_repr_str, py_split, py_strip};

/// The AINode commit the renderer was copied from.
pub const PROMPT_SOURCE_COMMIT: &str = "e5c089386e0239c9eb270eeb490d181722b8da5b";
/// `prompt_source_sha256` of the copied definitions, as a model's
/// `prompt_contract.json` names it.
pub const PROMPT_SOURCE_SHA256: &str =
    "d2660ebec28bd3f1704235bda88d24a397c1c62475e740519cb8ef2d08f25fdd";

/// The system message.
pub const SYSTEM_PROMPT: &str =
    "You are a decision function. Answer with the single letter of the best option and nothing else.";
/// The last line of the user message.
pub const ANSWER_INSTRUCTION: &str = "Answer with the label of one option and nothing else.";
/// AINode's widest option set.
pub const MAX_OPTIONS: usize = 255;
/// The labels one engine call reports a probability for.
pub const TOP_LOGPROBS: usize = 20;
/// AINode's widest choice question.
pub const MAX_CRITERIA: usize = TOP_LOGPROBS;
/// The fewest levels of a score question.
pub const MIN_SCORE_LEVELS: usize = 2;
/// The most levels of a score question.
pub const MAX_SCORE_LEVELS: usize = 10;
/// A noul's two options, always in this order.
pub const NOUL_OPTIONS: [&str; 2] = ["true", "false"];

/// A Jev question type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// One of the criteria keys.
    Choice,
    /// True or false.
    Noul,
    /// One level of an ordered scale.
    Score,
}

/// One question as the decision core sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translated {
    /// The question type.
    pub kind: Kind,
    /// The question line.
    pub question: String,
    /// One option line each.
    pub options: Vec<PromptText>,
    /// What each option answers to on the wire, stripped.
    pub names: Vec<String>,
}

/// `option_text`: the name, then its description flattened to one line.
pub fn option_text(name: &str, description: Option<&str>) -> PromptText {
    let mut text = PromptText::caller(name);
    if let Some(d) = description.filter(|d| !py_strip(d).is_empty()) {
        text.push_template(": ").push_caller(py_split(d).join(" "));
    }
    text
}

/// A criteria description: `None` for null, the string otherwise.
type Pair = (String, Option<String>);

/// `criteria_pairs`: the `(name, description)` pairs of an object
/// `criteria`, names stripped, in order.
pub fn criteria_pairs(
    key: &str,
    criteria: Option<&Value>,
    kind: &str,
) -> Result<Vec<Pair>, String> {
    let map = match criteria {
        Some(Value::Object(map)) if !map.is_empty() => map,
        _ => {
            return Err(format!(
                "question '{key}': a {kind} question needs a non-empty 'criteria' object of \
                 {{name: description}}"
            ))
        }
    };
    let mut pairs = Vec::with_capacity(map.len());
    for (name, description) in map {
        if py_strip(name).is_empty() {
            return Err(format!(
                "question '{key}': every 'criteria' name must be a non-empty string (got {})",
                py_repr_str(name)
            ));
        }
        let description = match description {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            _ => {
                return Err(format!(
                    "question '{key}': the 'criteria' description for '{name}' must be a string"
                ))
            }
        };
        pairs.push((py_strip(name).to_string(), description));
    }
    Ok(pairs)
}

/// Option lines and the names they answer to.
type Options = (Vec<PromptText>, Vec<String>);

fn pairs_to_options(pairs: &[Pair]) -> Options {
    let options = pairs
        .iter()
        .map(|(n, d)| option_text(n, d.as_deref()))
        .collect();
    (options, pairs.iter().map(|(n, _)| n.clone()).collect())
}

/// `choice_options`.
fn choice_options(key: &str, criteria: Option<&Value>) -> Result<Options, String> {
    let pairs = criteria_pairs(key, criteria, "choice")?;
    if pairs.len() < 2 {
        return Err(format!(
            "question '{key}': a choice needs at least 2 'criteria' options, got {}",
            pairs.len()
        ));
    }
    if pairs.len() > MAX_CRITERIA {
        return Err(format!(
            "question '{key}': {} 'criteria' options is more than the {MAX_CRITERIA} this node \
             can report a probability for. One engine call carries back the top {TOP_LOGPROBS} \
             labels, so a wider option set would answer with a distribution missing its tail",
            pairs.len()
        ));
    }
    Ok(pairs_to_options(&pairs))
}

/// `noul_options`: always `true` then `false`, each with its description.
fn noul_options(key: &str, criteria: Option<&Value>) -> Result<Options, String> {
    let mut described: Map<String, Value> = Map::new();
    match criteria {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (name, description) in map {
                let flat = py_strip(name);
                if !NOUL_OPTIONS.contains(&flat) {
                    return Err(format!(
                        "question '{key}': a noul's 'criteria' names only 'true' and 'false' \
                         (got {})",
                        py_repr_str(name)
                    ));
                }
                if !matches!(description, Value::Null | Value::String(_)) {
                    return Err(format!(
                        "question '{key}': the 'criteria' description for '{flat}' must be a \
                         string"
                    ));
                }
                described.insert(flat.to_string(), description.clone());
            }
        }
        Some(_) => {
            return Err(format!(
                "question '{key}': 'criteria' must be an object of {{true: description, false: \
                 description}}"
            ))
        }
    }
    let options = NOUL_OPTIONS
        .iter()
        .map(|name| option_text(name, described.get(*name).and_then(Value::as_str)))
        .collect();
    Ok((
        options,
        NOUL_OPTIONS.iter().map(|s| s.to_string()).collect(),
    ))
}

/// `score_options`: levels in rubric order, a list by position or an
/// object by insertion.
fn score_options(key: &str, criteria: Option<&Value>) -> Result<Options, String> {
    let (options, names) = match criteria {
        Some(Value::Array(levels)) => {
            let mut names = Vec::with_capacity(levels.len());
            for level in levels {
                match level.as_str().map(py_strip) {
                    Some(name) if !name.is_empty() => names.push(name.to_string()),
                    _ => {
                        return Err(format!(
                            "question '{key}': every 'criteria' level must be a non-empty \
                             string (got {})",
                            py_repr(level)
                        ))
                    }
                }
            }
            let options = names.iter().map(PromptText::caller).collect();
            (options, names)
        }
        Some(Value::Object(_)) => pairs_to_options(&criteria_pairs(key, criteria, "score")?),
        _ => {
            return Err(format!(
                "question '{key}': a score question needs 'criteria', either an ordered list of \
                 levels or an object of {{level: description}}"
            ))
        }
    };
    if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&names.len()) {
        return Err(format!(
            "question '{key}': a score's 'criteria' needs {MIN_SCORE_LEVELS} to \
             {MAX_SCORE_LEVELS} ordered levels, got {}",
            names.len()
        ));
    }
    let mut seen = names.clone();
    seen.sort();
    seen.dedup();
    if seen.len() != names.len() {
        return Err(format!(
            "question '{key}': 'criteria' repeats a level name. Every level must be distinct so \
             a score names one of them"
        ));
    }
    Ok((options, names))
}

/// The question type of `spec`, or AINode's error for it.
pub fn question_kind(key: &str, spec: &Map<String, Value>) -> Result<Kind, String> {
    match spec.get("type").and_then(Value::as_str) {
        Some("choice") => Ok(Kind::Choice),
        Some("noul") => Ok(Kind::Noul),
        Some("score") => Ok(Kind::Score),
        _ => Err(format!(
            "question '{key}': 'type' must be one of choice, noul, score (got {})",
            spec.get("type").map_or_else(|| "None".to_string(), py_repr)
        )),
    }
}

/// `translate_one`: one question off the wire, from its `type`,
/// `instructions` and `criteria`.
pub fn translate_one(key: &str, spec: &Value) -> Result<Translated, String> {
    let Value::Object(spec) = spec else {
        return Err(format!("question '{key}' must be an object"));
    };
    let kind = question_kind(key, spec)?;
    let instructions = match spec.get("instructions") {
        Some(Value::String(s)) if !py_strip(s).is_empty() => py_strip(s).to_string(),
        _ => {
            return Err(format!(
                "question '{key}' needs a non-empty 'instructions' string"
            ))
        }
    };
    let criteria = spec.get("criteria");
    let (options, names) = match kind {
        Kind::Noul => noul_options(key, criteria)?,
        Kind::Score => score_options(key, criteria)?,
        Kind::Choice => choice_options(key, criteria)?,
    };
    Ok(Translated {
        kind,
        question: instructions,
        options,
        names,
    })
}

/// The option checks of `normalize_questions`, which `/v1/systemone` runs
/// on every translated question: 2 to 255 options, each distinct.
pub fn check_options(key: &str, options: &[PromptText]) -> Result<(), String> {
    let texts: Vec<String> = options.iter().map(PromptText::text).collect();
    if texts.len() < 2 {
        return Err(format!(
            "question '{key}': needs at least 2 options, got {}",
            texts.len()
        ));
    }
    if texts.len() > MAX_OPTIONS {
        return Err(format!(
            "question '{key}': {} options is more than the {MAX_OPTIONS} allowed",
            texts.len()
        ));
    }
    let mut sorted: Vec<&String> = texts.iter().collect();
    sorted.sort();
    let mut dupes: Vec<&String> = sorted
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0])
        .collect();
    if !dupes.is_empty() {
        dupes.dedup();
        let listed: Vec<String> = dupes.iter().map(|d| py_repr_str(d)).collect();
        return Err(format!(
            "question '{key}': duplicate options [{}]. Every option must be distinct so an \
             answer is unambiguous",
            listed.join(", ")
        ));
    }
    Ok(())
}

/// `build_messages`' user message for `labels` (AINode's `option_label`
/// order unless the caller swaps the alphabet).
pub fn user_message(
    state: &str,
    question: &str,
    options: &[PromptText],
    labels: &[String],
) -> PromptText {
    let mut text = PromptText::template("STATE:\n");
    text.push_caller(state)
        .push_template("\n\nQUESTION: ")
        .push_caller(question)
        .push_template("\n\nOPTIONS:");
    for (label, option) in labels.iter().zip(options) {
        text.push_template(format!("\n{label}. ")).push_text(option);
    }
    text.push_template(format!("\n\n{ANSWER_INSTRUCTION}"));
    text
}

/// AINode's labels for `n` options.
pub fn ainode_labels(n: usize) -> Vec<String> {
    (0..n).map(option_label).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options(t: &Translated) -> Vec<String> {
        t.options.iter().map(PromptText::text).collect()
    }

    #[test]
    fn choice_options_keep_names_and_flatten_descriptions() {
        let spec = json!({"type": "choice", "instructions": "  Pick one ",
            "criteria": {" billing ": "an invoice,\n a  charge", "support": null, "sales": "  "}});
        let t = translate_one("route", &spec).expect("translates");
        assert_eq!(t.question, "Pick one");
        assert_eq!(
            options(&t),
            ["billing: an invoice, a charge", "support", "sales"]
        );
        assert_eq!(t.names, ["billing", "support", "sales"]);
    }

    #[test]
    fn noul_is_true_then_false() {
        let spec = json!({"type": "noul", "instructions": "Blocked?",
            "criteria": {"false": "it can wait", " true": "stopped"}});
        let t = translate_one("b", &spec).expect("translates");
        assert_eq!(options(&t), ["true: stopped", "false: it can wait"]);
        let bad = json!({"type": "noul", "instructions": "x", "criteria": {"yes": null}});
        assert_eq!(
            translate_one("b", &bad),
            Err(
                "question 'b': a noul's 'criteria' names only 'true' and 'false' (got 'yes')"
                    .into()
            )
        );
    }

    #[test]
    fn errors_are_ainodes() {
        assert_eq!(
            translate_one("q", &json!({"type": "maybe", "instructions": "x"})),
            Err("question 'q': 'type' must be one of choice, noul, score (got 'maybe')".into())
        );
        assert_eq!(
            translate_one("q", &json!({"instructions": "x"})),
            Err("question 'q': 'type' must be one of choice, noul, score (got None)".into())
        );
        assert_eq!(
            translate_one("q", &json!({"type": "choice", "instructions": " "})),
            Err("question 'q' needs a non-empty 'instructions' string".into())
        );
        assert_eq!(
            translate_one(
                "q",
                &json!({"type": "choice", "instructions": "x", "criteria": {}})
            ),
            Err(
                "question 'q': a choice question needs a non-empty 'criteria' object of \
                 {name: description}"
                    .into()
            )
        );
        assert_eq!(
            translate_one(
                "q",
                &json!({"type": "score", "instructions": "x", "criteria": ["a", 1]})
            ),
            Err("question 'q': every 'criteria' level must be a non-empty string (got 1)".into())
        );
        assert_eq!(
            translate_one(
                "q",
                &json!({"type": "score", "instructions": "x", "criteria": ["a", "a "]})
            ),
            Err(
                "question 'q': 'criteria' repeats a level name. Every level must be distinct so \
                 a score names one of them"
                    .into()
            )
        );
    }

    #[test]
    fn duplicate_options_are_refused() {
        let opts = [
            PromptText::caller("a"),
            PromptText::caller("b"),
            PromptText::caller("a"),
        ];
        assert_eq!(
            check_options("k", &opts),
            Err(
                "question 'k': duplicate options ['a']. Every option must be distinct so an \
                 answer is unambiguous"
                    .into()
            )
        );
    }

    #[test]
    fn user_message_matches_build_messages() {
        let opts = [PromptText::caller("o1"), PromptText::caller("o2")];
        let text = user_message("x", "q?", &opts, &ainode_labels(2)).text();
        assert_eq!(
            text,
            "STATE:\nx\n\nQUESTION: q?\n\nOPTIONS:\nA. o1\nB. o2\n\nAnswer with the label of one \
             option and nothing else."
        );
    }
}

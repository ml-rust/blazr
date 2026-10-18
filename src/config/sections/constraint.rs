//! `Constraint`: the language a request's output must match, and its compiled DFA.

use std::fmt;
use std::sync::Arc;

use boostr::inference::grammar::ByteDfa;
use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Map key of a GBNF constraint.
const GBNF: &str = "gbnf";
/// Map key of a JSON-schema constraint.
const JSON_SCHEMA: &str = "json_schema";
/// Map key of the any-JSON-object constraint.
const JSON_OBJECT: &str = "json_object";

/// The language a constrained request's output must belong to.
///
/// The config key `generation.grammar` holds one in one of these forms:
/// - a string is GBNF text: `grammar: 'root ::= "yes" | "no"'`.
/// - `grammar: {gbnf: <text>}` is GBNF text too.
/// - `grammar: {json_schema: <schema>}` is a JSON schema, not an OpenAI wrapper.
/// - `grammar: {json_object: true}` is any JSON object.
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// GBNF grammar text.
    Gbnf(String),
    /// A JSON schema. Object properties keep the key order of the value.
    JsonSchema(Value),
    /// Any JSON object, the `json_object` response format.
    JsonObject,
}

impl Serialize for Constraint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Gbnf(text) => serializer.serialize_str(text),
            Self::JsonSchema(schema) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(JSON_SCHEMA, schema)?;
                map.end()
            }
            Self::JsonObject => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(JSON_OBJECT, &true)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Constraint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ConstraintVisitor)
    }
}

/// Reads the string form or the one-key map form of a [`Constraint`].
struct ConstraintVisitor;

impl<'de> Visitor<'de> for ConstraintVisitor {
    type Value = Constraint;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GBNF text, or a map with one key: gbnf, json_schema or json_object")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Constraint, E> {
        Ok(Constraint::Gbnf(text.to_string()))
    }

    fn visit_string<E: de::Error>(self, text: String) -> Result<Constraint, E> {
        Ok(Constraint::Gbnf(text))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Constraint, A::Error> {
        let key: String = map.next_key()?.ok_or_else(|| {
            <A::Error as de::Error>::custom(
                "the grammar map is empty. Set gbnf, json_schema or json_object",
            )
        })?;
        let constraint = match key.as_str() {
            GBNF => Constraint::Gbnf(map.next_value()?),
            JSON_SCHEMA => Constraint::JsonSchema(map.next_value()?),
            JSON_OBJECT => {
                if !map.next_value::<bool>()? {
                    return Err(de::Error::custom(
                        "grammar json_object is false. Set it to true, or remove the grammar key",
                    ));
                }
                Constraint::JsonObject
            }
            other => {
                return Err(de::Error::unknown_field(
                    other,
                    &[GBNF, JSON_SCHEMA, JSON_OBJECT],
                ))
            }
        };
        if map.next_key::<IgnoredAny>()?.is_some() {
            return Err(de::Error::custom(
                "the grammar map holds more than one key. Keep one of gbnf, json_schema or \
                 json_object",
            ));
        }
        Ok(constraint)
    }
}

/// A constraint and the DFA compiled from it.
///
/// Request admission fills it, so the sampler compiles nothing. A clone
/// shares the DFA.
#[derive(Clone)]
pub struct CompiledConstraint {
    source: Constraint,
    dfa: Arc<ByteDfa>,
}

impl CompiledConstraint {
    /// Pair `dfa` with the `source` it was compiled from.
    pub fn new(source: Constraint, dfa: Arc<ByteDfa>) -> Self {
        Self { source, dfa }
    }

    /// The constraint the DFA was compiled from.
    pub fn source(&self) -> &Constraint {
        &self.source
    }

    /// The compiled DFA.
    pub fn dfa(&self) -> &Arc<ByteDfa> {
        &self.dfa
    }
}

impl fmt::Debug for CompiledConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledConstraint")
            .field("source", &self.source)
            .field("num_states", &self.dfa.num_states())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Deserialize, Serialize)]
    struct Section {
        #[serde(default)]
        grammar: Option<Constraint>,
    }

    fn yaml(text: &str) -> Option<Constraint> {
        serde_saphyr::from_str::<Section>(text)
            .unwrap_or_else(|e| panic!("{text}: {e}"))
            .grammar
    }

    #[test]
    fn a_grammar_string_is_gbnf_text() {
        let text = "grammar: 'root ::= \"yes\" | \"no\"'\n";
        assert_eq!(
            yaml(text),
            Some(Constraint::Gbnf(r#"root ::= "yes" | "no""#.to_string()))
        );
        assert_eq!(yaml("{}\n"), None);
        assert_eq!(yaml("grammar: null\n"), None);
    }

    #[test]
    fn map_forms_name_their_kind() {
        let schema = yaml("grammar:\n  json_schema:\n    type: integer\n");
        assert_eq!(
            schema,
            Some(Constraint::JsonSchema(json!({"type": "integer"})))
        );
        assert_eq!(
            yaml("grammar:\n  json_object: true\n"),
            Some(Constraint::JsonObject)
        );
        assert_eq!(
            yaml("grammar:\n  gbnf: 'root ::= \"a\"'\n"),
            Some(Constraint::Gbnf(r#"root ::= "a""#.to_string()))
        );
    }

    #[test]
    fn bad_map_forms_are_errors() {
        for text in [
            "grammar: {}\n",
            "grammar:\n  json_object: false\n",
            "grammar:\n  regex: a+\n",
            "grammar:\n  json_object: true\n  gbnf: x\n",
        ] {
            assert!(
                serde_saphyr::from_str::<Section>(text).is_err(),
                "{text} must fail"
            );
        }
    }

    #[test]
    fn every_kind_round_trips_through_json() {
        for constraint in [
            Constraint::Gbnf(r#"root ::= "a""#.to_string()),
            Constraint::JsonSchema(json!({"type": "object", "properties": {"b": {}, "a": {}}})),
            Constraint::JsonObject,
        ] {
            let section = Section {
                grammar: Some(constraint.clone()),
            };
            let text = serde_json::to_string(&section).unwrap_or_default();
            let back: Section = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(back.grammar, Some(constraint), "{text}");
        }
    }
}

//! The constraint that forces a tool call: bare JSON `{"name":...,"arguments":...}`.

use serde_json::{json, Value};

use super::tool_choice::ToolChoice;
use super::tools::Tool;
use crate::config::Constraint;
use crate::server::generation::ResponseFormat;

/// The schema of one call to `tool`.
///
/// `name` is the tool's own name. `arguments` follows the tool's parameters
/// schema. A tool with no parameters takes the empty object.
fn call_schema(tool: &Tool) -> Value {
    let arguments = tool
        .function
        .parameters
        .clone()
        .unwrap_or_else(|| json!({"type": "object"}));
    json!({
        "type": "object",
        "properties": {
            "name": {"const": tool.function.name},
            "arguments": arguments,
        },
        "required": ["name", "arguments"],
    })
}

/// The constraint `choice` puts on the output, or `None` when it puts none.
///
/// - `required` allows a call to any tool in `tools`.
/// - A named choice allows a call to that tool only.
///
/// # Errors
///
/// A named tool that `tools` lacks. `ToolChoice::parse` already refuses it.
pub fn tool_call_constraint(
    choice: &ToolChoice,
    tools: &[Tool],
) -> Result<Option<Constraint>, String> {
    if !choice.forces_call() {
        return Ok(None);
    }
    let schemas: Vec<Value> = match choice {
        ToolChoice::Named(name) => {
            let tool = tools
                .iter()
                .find(|t| &t.function.name == name)
                .ok_or_else(|| {
                    format!(
                        "tool_choice names function {name:?} and no declared tool has it. \
                         Declare the tool"
                    )
                })?;
            vec![call_schema(tool)]
        }
        ToolChoice::Mode(_) => tools.iter().map(call_schema).collect(),
    };
    let schema = match <[Value; 1]>::try_from(schemas) {
        Ok([only]) => only,
        Err(many) => json!({"anyOf": many}),
    };
    Ok(Some(Constraint::JsonSchema(schema)))
}

/// The constraint of a chat request.
///
/// A forced tool call constrains the output. `grammar` and a `response_format`
/// other than `text` constrain it too, so each is an error beside a forced call.
/// Without a forced call, `request` is the constraint `grammar` and
/// `response_format` ask for.
///
/// # Errors
///
/// The error string of a request the handler answers 400.
pub fn chat_constraint(
    choice: &ToolChoice,
    tools: &[Tool],
    grammar: Option<&str>,
    response_format: Option<&ResponseFormat>,
    request: Result<Option<Constraint>, String>,
) -> Result<Option<Constraint>, String> {
    let Some(forced) = tool_call_constraint(choice, tools)? else {
        return request;
    };
    let has_grammar = grammar.is_some_and(|text| !text.trim().is_empty());
    let has_format = response_format.is_some_and(|rf| rf.r#type != "text");
    if has_grammar || has_format {
        let field = if has_grammar {
            "grammar"
        } else {
            "response_format"
        };
        return Err(format!(
            "tool_choice {} forces a tool call and {field} also constrains the output. \
             Remove {field} or set tool_choice to \"auto\"",
            choice_text(choice)
        ));
    }
    Ok(Some(forced))
}

/// The `tool_choice` as the client wrote it, for an error text.
fn choice_text(choice: &ToolChoice) -> String {
    match choice {
        ToolChoice::Named(name) => format!("{{\"function\":{{\"name\":{name:?}}}}}"),
        ToolChoice::Mode(_) => "\"required\"".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::tool_choice::ToolMode;
    use super::super::tools::ToolFunction;
    use super::*;
    use crate::engine::compile_constraint;
    use boostr::inference::grammar::json_schema::SchemaOptions;
    use boostr::inference::grammar::CompileLimits;

    fn tool(name: &str, parameters: Option<Value>) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: name.to_string(),
                description: None,
                parameters,
            },
        }
    }

    fn weather() -> Tool {
        tool(
            "weather",
            Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            })),
        )
    }

    fn dfa(constraint: &Constraint) -> boostr::inference::grammar::ByteDfa {
        compile_constraint(
            constraint,
            &SchemaOptions::default(),
            &CompileLimits::default(),
        )
        .unwrap_or_else(|e| panic!("{e}"))
    }

    fn format(kind: &str) -> ResponseFormat {
        ResponseFormat {
            r#type: kind.to_string(),
            json_schema: None,
        }
    }

    #[test]
    fn a_named_constraint_accepts_only_that_tool() {
        let tools = [weather(), tool("ping", None)];
        let choice = ToolChoice::Named("weather".to_string());
        let constraint = tool_call_constraint(&choice, &tools).unwrap().unwrap();
        let dfa = dfa(&constraint);
        assert!(dfa.accepts(br#"{"name":"weather","arguments":{"city":"Paris"}}"#));
        assert!(!dfa.accepts(br#"{"name":"ping","arguments":{}}"#));
        assert!(!dfa.accepts(br#"{"name":"weather","arguments":{}}"#));
        assert!(!dfa.accepts(b"hello"));
    }

    #[test]
    fn a_required_constraint_accepts_every_tool() {
        let tools = [weather(), tool("ping", None)];
        let choice = ToolChoice::Mode(ToolMode::Required);
        let constraint = tool_call_constraint(&choice, &tools).unwrap().unwrap();
        let dfa = dfa(&constraint);
        assert!(dfa.accepts(br#"{"name":"weather","arguments":{"city":"Paris"}}"#));
        assert!(dfa.accepts(br#"{"name":"ping","arguments":{}}"#));
        assert!(!dfa.accepts(br#"{"name":"other","arguments":{}}"#));
    }

    #[test]
    fn auto_and_none_add_no_constraint() {
        for mode in [ToolMode::Auto, ToolMode::None] {
            let got = tool_call_constraint(&ToolChoice::Mode(mode), &[weather()]);
            assert_eq!(got, Ok(None));
        }
    }

    #[test]
    fn a_forced_call_beside_grammar_or_a_format_is_an_error() {
        let tools = [weather()];
        let choice = ToolChoice::Mode(ToolMode::Required);
        let by_grammar = chat_constraint(&choice, &tools, Some("root ::= \"a\""), None, Ok(None));
        let message = by_grammar.unwrap_err();
        assert!(message.contains("\"required\"") && message.contains("grammar"));
        assert!(message.contains("Remove grammar"), "{message}");

        let json = format("json_object");
        let by_format = chat_constraint(&choice, &tools, None, Some(&json), Ok(None));
        assert!(by_format.unwrap_err().contains("Remove response_format"));

        let text = format("text");
        let plain = chat_constraint(&choice, &tools, Some("  "), Some(&text), Ok(None));
        assert!(matches!(plain, Ok(Some(Constraint::JsonSchema(_)))));
    }

    #[test]
    fn without_a_forced_call_the_request_constraint_stands() {
        let choice = ToolChoice::Mode(ToolMode::Auto);
        let request = Ok(Some(Constraint::JsonObject));
        let got = chat_constraint(&choice, &[weather()], None, None, request);
        assert_eq!(got, Ok(Some(Constraint::JsonObject)));
    }
}

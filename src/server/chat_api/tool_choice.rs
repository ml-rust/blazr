//! The OpenAI `tool_choice` field: parsing and what each form asks for.

use serde_json::Value;

use super::tools::Tool;

/// What a string `tool_choice` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    /// The model gets no tools and calls none.
    None,
    /// The model decides whether to call a tool. The default.
    Auto,
    /// The model must call at least one declared tool.
    Required,
}

/// A parsed `tool_choice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    /// One of the string forms.
    Mode(ToolMode),
    /// `{"type":"function","function":{"name":...}}`: the model must call this tool.
    Named(String),
}

impl ToolChoice {
    /// Parse a request's `tool_choice` against its declared `tools`.
    ///
    /// A missing `tool_choice` is `auto`.
    ///
    /// # Errors
    ///
    /// The error string of a request the handler answers 400: a value
    /// without declared tools, a shape that is not a `tool_choice`, or a
    /// named function that no declared tool has.
    pub fn parse(value: Option<&Value>, tools: Option<&[Tool]>) -> Result<Self, String> {
        let Some(value) = value else {
            return Ok(Self::Mode(ToolMode::Auto));
        };
        let tools = tools.unwrap_or(&[]);
        if tools.is_empty() {
            return Err(format!(
                "tool_choice {value} is set and the request declares no tools. \
                 Add tools or remove tool_choice"
            ));
        }
        let choice = Self::from_value(value)?;
        if let Self::Named(name) = &choice {
            if !tools.iter().any(|t| &t.function.name == name) {
                let declared: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
                return Err(format!(
                    "tool_choice names function {name:?} and no declared tool has that name. \
                     Use one of: {}",
                    declared.join(", ")
                ));
            }
        }
        Ok(choice)
    }

    /// The typed form of one `tool_choice` value.
    fn from_value(value: &Value) -> Result<Self, String> {
        let invalid = || {
            format!(
                "tool_choice {value} is not valid. Use \"none\", \"auto\", \"required\" or \
                 {{\"type\":\"function\",\"function\":{{\"name\":\"<tool name>\"}}}}"
            )
        };
        match value {
            Value::String(text) => match text.as_str() {
                "none" => Ok(Self::Mode(ToolMode::None)),
                "auto" => Ok(Self::Mode(ToolMode::Auto)),
                "required" => Ok(Self::Mode(ToolMode::Required)),
                _ => Err(invalid()),
            },
            Value::Object(map) => {
                let is_function = map.get("type").and_then(Value::as_str) == Some("function");
                let name = map
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty());
                match (is_function, name) {
                    (true, Some(name)) => Ok(Self::Named(name.to_string())),
                    _ => Err(invalid()),
                }
            }
            _ => Err(invalid()),
        }
    }

    /// The tools the model sees: none under `none`, else all of `tools`.
    pub fn visible<'a>(&self, tools: &'a [Tool]) -> &'a [Tool] {
        match self {
            Self::Mode(ToolMode::None) => &[],
            _ => tools,
        }
    }

    /// True when the output must be a tool call.
    pub fn forces_call(&self) -> bool {
        matches!(self, Self::Mode(ToolMode::Required) | Self::Named(_))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tools::{build_tools_system_prompt, ToolFunction};
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: name.to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                })),
            },
        }
    }

    fn parse(value: Value) -> Result<ToolChoice, String> {
        ToolChoice::parse(Some(&value), Some(&[tool("weather"), tool("time")]))
    }

    #[test]
    fn a_missing_value_is_auto() {
        let got = ToolChoice::parse(None, None);
        assert_eq!(got, Ok(ToolChoice::Mode(ToolMode::Auto)));
    }

    #[test]
    fn each_string_form_parses() {
        assert_eq!(parse(json!("none")), Ok(ToolChoice::Mode(ToolMode::None)));
        assert_eq!(parse(json!("auto")), Ok(ToolChoice::Mode(ToolMode::Auto)));
        assert_eq!(
            parse(json!("required")),
            Ok(ToolChoice::Mode(ToolMode::Required))
        );
    }

    #[test]
    fn the_named_form_parses() {
        let value = json!({"type": "function", "function": {"name": "time"}});
        assert_eq!(parse(value), Ok(ToolChoice::Named("time".to_string())));
    }

    #[test]
    fn an_invalid_shape_names_the_input_and_the_next_action() {
        for bad in [
            json!("sometimes"),
            json!(3),
            json!(true),
            json!([]),
            json!({}),
            json!({"type": "function"}),
            json!({"type": "function", "function": {"name": ""}}),
            json!({"type": "tool", "function": {"name": "time"}}),
        ] {
            let message = parse(bad.clone()).unwrap_err();
            assert!(message.contains(&bad.to_string()), "{message}");
            assert!(message.contains("Use \"none\""), "{message}");
        }
    }

    #[test]
    fn an_unknown_named_function_lists_the_declared_names() {
        let value = json!({"type": "function", "function": {"name": "stocks"}});
        let message = parse(value).unwrap_err();
        assert!(message.contains("\"stocks\""), "{message}");
        assert!(message.contains("Use one of: weather, time"), "{message}");
    }

    #[test]
    fn a_value_without_tools_is_an_error() {
        let value = json!("auto");
        for tools in [None, Some(&[][..])] {
            let message = ToolChoice::parse(Some(&value), tools).unwrap_err();
            assert!(message.contains("\"auto\""), "{message}");
            assert!(message.contains("Add tools or remove tool_choice"));
        }
    }

    #[test]
    fn none_leaves_the_prompt_without_tools() {
        let tools = [tool("weather")];
        let none = ToolChoice::Mode(ToolMode::None);
        assert!(none.visible(&tools).is_empty());
        assert_eq!(
            build_tools_system_prompt(none.visible(&tools), false),
            Ok(None)
        );
        let auto = ToolChoice::Mode(ToolMode::Auto);
        assert_eq!(auto.visible(&tools).len(), 1);
        let prompt = build_tools_system_prompt(auto.visible(&tools), false);
        assert!(prompt.unwrap().is_some_and(|p| p.contains("weather")));
    }

    #[test]
    fn only_required_and_named_force_a_call() {
        assert!(!ToolChoice::Mode(ToolMode::None).forces_call());
        assert!(!ToolChoice::Mode(ToolMode::Auto).forces_call());
        assert!(ToolChoice::Mode(ToolMode::Required).forces_call());
        assert!(ToolChoice::Named("x".to_string()).forces_call());
    }
}

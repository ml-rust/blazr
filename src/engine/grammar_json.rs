//! JSON Schema to GBNF conversion

use std::collections::HashSet;

/// Convert a JSON schema to a GBNF grammar string
pub fn json_schema_to_gbnf(schema: &serde_json::Value) -> Result<String, String> {
    let mut rules = vec![
        "root ::= ws value ws".to_string(),
        "ws ::= [ \\t\\n]*".to_string(),
        r#"value ::= string | number | "true" | "false" | "null" | object | array"#.to_string(),
        r#"string ::= "\"" [^"\\]* "\""#.to_string(),
        r#"number ::= "-"? [0-9]+ ("." [0-9]+)?"#.to_string(),
    ];

    // If the schema specifies a type, constrain the root rule
    if let Some(schema_type) = schema.get("type").and_then(|t| t.as_str()) {
        match schema_type {
            "object" => {
                let mut obj_rule = String::from(r#"object ::= "{" ws"#);
                if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
                    let required: HashSet<&str> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();

                    let mut first = true;
                    for (key, _val) in properties {
                        if !first {
                            obj_rule.push_str(r#" "," ws"#);
                        }
                        if required.contains(key.as_str()) || first {
                            obj_rule.push_str(&format!(r#" "\"{}\"" ws ":" ws value"#, key));
                        }
                        first = false;
                    }
                } else {
                    obj_rule
                        .push_str(r#" (string ws ":" ws value ("," ws string ws ":" ws value)*)?"#);
                }
                obj_rule.push_str(r#" ws "}""#);
                rules.push(obj_rule);
                rules.push(r#"root ::= ws object ws"#.to_string());
            }
            "array" => {
                rules.push(r#"array ::= "[" ws (value ("," ws value)*)? ws "]""#.to_string());
                rules.push(r#"root ::= ws array ws"#.to_string());
            }
            "string" => {
                rules.push(r#"root ::= ws string ws"#.to_string());
            }
            "number" | "integer" => {
                rules.push(r#"root ::= ws number ws"#.to_string());
            }
            "boolean" => {
                rules.push(r#"root ::= ws ("true" | "false") ws"#.to_string());
            }
            _ => {}
        }
    } else {
        rules.push(
            r#"object ::= "{" ws (string ws ":" ws value ("," ws string ws ":" ws value)*)? ws "}""#
                .to_string(),
        );
        rules.push(r#"array ::= "[" ws (value ("," ws value)*)? ws "]""#.to_string());
    }

    Ok(rules.join("\n"))
}

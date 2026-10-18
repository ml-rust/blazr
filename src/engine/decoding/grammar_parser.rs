//! GBNF grammar parser
//!
//! Parses GBNF grammar strings into structured rule representations.

/// GBNF grammar rule
#[derive(Debug, Clone)]
pub struct GbnfRule {
    pub name: String,
    pub alternatives: Vec<GbnfSequence>,
}

/// A sequence of elements in a GBNF rule
pub type GbnfSequence = Vec<GbnfElement>;

/// Elements in a GBNF grammar
#[derive(Debug, Clone)]
pub enum GbnfElement {
    /// Literal string
    Literal(String),
    /// Reference to another rule
    RuleRef(String),
    /// Character class `[a-z0-9]`
    CharClass(Vec<CharRange>),
    /// Negated character class `[^...]`
    NegCharClass(Vec<CharRange>),
    /// Repetition: element*
    Repeat(Box<GbnfElement>, RepeatKind),
    /// Optional: element?
    Optional(Box<GbnfElement>),
}

#[derive(Debug, Clone, Copy)]
pub struct CharRange {
    pub start: u8,
    pub end: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum RepeatKind {
    /// element* (zero or more)
    Star,
    /// element+ (one or more)
    Plus,
}

/// Parse a GBNF grammar string into rules
pub fn parse_gbnf(input: &str) -> Result<Vec<GbnfRule>, String> {
    let mut rules = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Format: rule-name ::= alternatives
        let parts: Vec<&str> = line.splitn(2, "::=").collect();
        if parts.len() != 2 {
            return Err(format!("Invalid GBNF rule: {}", line));
        }

        let name = parts[0].trim().to_string();
        let body = parts[1].trim();

        let alternatives: Vec<GbnfSequence> = split_alternatives(body)
            .into_iter()
            .map(|alt| parse_gbnf_sequence(alt.trim()))
            .collect::<Result<Vec<_>, _>>()?;

        rules.push(GbnfRule { name, alternatives });
    }

    if rules.is_empty() {
        return Err("No rules found in GBNF grammar".to_string());
    }

    Ok(rules)
}

/// Split a rule body at each `|` outside a literal or character class.
fn split_alternatives(body: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_literal = false;
    let mut in_class = false;
    let mut escaped = false;
    for (idx, c) in body.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_literal => escaped = true,
            '"' if !in_class => in_literal = !in_literal,
            '[' if !in_literal => in_class = true,
            ']' if in_class => in_class = false,
            '|' if !in_literal && !in_class => {
                parts.push(&body[start..idx]);
                start = idx + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&body[start..]);
    parts
}

/// Build the error for an unterminated literal or class.
fn unterminated(what: &str, pos: usize, input: &str) -> String {
    format!(
        "Unterminated {what} starting at position {pos} in `{input}`. \
         Close it with {close}.",
        close = if what == "literal" { "`\"`" } else { "`]`" }
    )
}

/// Parse one alternative into elements.
///
/// Every loop iteration consumes at least one character or returns an error.
pub(crate) fn parse_gbnf_sequence(input: &str) -> Result<GbnfSequence, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut elements = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' => {
                let open = i;
                i += 1;
                let mut literal = String::new();
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    i += 1;
                    if ch == '"' {
                        closed = true;
                        break;
                    }
                    if ch != '\\' {
                        literal.push(ch);
                        continue;
                    }
                    let Some(&escaped) = chars.get(i) else {
                        break;
                    };
                    i += 1;
                    match escaped {
                        'n' => literal.push('\n'),
                        't' => literal.push('\t'),
                        '"' => literal.push('"'),
                        '\\' => literal.push('\\'),
                        _ => {
                            literal.push('\\');
                            literal.push(escaped);
                        }
                    }
                }
                if !closed {
                    return Err(unterminated("literal", open, input));
                }
                elements.push(GbnfElement::Literal(literal));
            }
            '[' => {
                let open = i;
                i += 1;
                let negated = chars.get(i) == Some(&'^');
                if negated {
                    i += 1;
                }
                let mut ranges = Vec::new();
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    i += 1;
                    if ch == ']' {
                        closed = true;
                        break;
                    }
                    let start = class_byte(ch, i - 1, input)?;
                    if chars.get(i) == Some(&'-') {
                        i += 1;
                        let Some(&end_ch) = chars.get(i) else {
                            break;
                        };
                        let end = class_byte(end_ch, i, input)?;
                        i += 1;
                        ranges.push(CharRange { start, end });
                    } else {
                        ranges.push(CharRange { start, end: start });
                    }
                }
                if !closed {
                    return Err(unterminated("character class", open, input));
                }
                if negated {
                    elements.push(GbnfElement::NegCharClass(ranges));
                } else {
                    elements.push(GbnfElement::CharClass(ranges));
                }
            }
            ' ' | '\t' => {
                i += 1;
            }
            _ => {
                let begin = i;
                let mut name = String::new();
                while let Some(&ch) = chars.get(i) {
                    if ch.is_alphanumeric() || ch == '-' || ch == '_' {
                        name.push(ch);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if name.is_empty() {
                    return Err(format!(
                        "Unsupported GBNF syntax: unexpected character `{c}` at position {begin} \
                         in `{input}`. The parser supports literals (\"text\"), \
                         character classes ([a-z], [^x]) and rule references with *, + or ?."
                    ));
                }
                let element = GbnfElement::RuleRef(name);
                match chars.get(i) {
                    Some('*') => {
                        i += 1;
                        elements.push(GbnfElement::Repeat(Box::new(element), RepeatKind::Star));
                    }
                    Some('+') => {
                        i += 1;
                        elements.push(GbnfElement::Repeat(Box::new(element), RepeatKind::Plus));
                    }
                    Some('?') => {
                        i += 1;
                        elements.push(GbnfElement::Optional(Box::new(element)));
                    }
                    _ => elements.push(element),
                }
            }
        }
    }

    Ok(elements)
}

/// Convert a character-class character to its byte.
fn class_byte(ch: char, pos: usize, input: &str) -> Result<u8, String> {
    u8::try_from(ch).ok().filter(u8::is_ascii).ok_or_else(|| {
        format!(
            "Unsupported GBNF syntax: non-ASCII character `{ch}` at position {pos} in \
             `{input}`. Character classes support ASCII only."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_err(input: &str) -> bool {
        parse_gbnf(input).is_err()
    }

    #[test]
    fn unsupported_syntax_is_an_error() {
        for g in [
            r#"root ::= ("a" | "b")"#,
            "root ::= a )",
            "root ::= .",
            "root ::= a{2}",
            "root ::= * a",
        ] {
            assert!(is_err(g), "{g} must be an error");
        }
    }

    #[test]
    fn error_names_the_character_and_position() {
        let Err(msg) = parse_gbnf_sequence("a (") else {
            panic!("expected an error");
        };
        assert!(msg.contains("`(`") && msg.contains("position 2"), "{msg}");
    }

    #[test]
    fn unterminated_input_is_an_error() {
        for g in [
            r#"root ::= "unterminated"#,
            r#"root ::= "esc\"#,
            "root ::= [a-z",
            "root ::= [a-",
            "root ::= [^",
        ] {
            assert!(is_err(g), "{g} must be an error");
        }
    }

    #[test]
    fn non_ascii_class_is_an_error() {
        assert!(is_err("root ::= [é]"));
    }

    #[test]
    fn every_ascii_punctuation_element_returns() {
        for p in (0u8..128)
            .map(char::from)
            .filter(char::is_ascii_punctuation)
        {
            let result = parse_gbnf_sequence(&p.to_string());
            if p == '-' || p == '_' {
                assert!(result.is_ok(), "`{p}` names a rule");
            } else {
                assert!(result.is_err(), "`{p}` must be an error");
            }
        }
    }

    #[test]
    fn valid_elements_parse() {
        let lit = parse_gbnf_sequence(r#""ab\n""#).unwrap();
        assert!(matches!(&lit[0], GbnfElement::Literal(s) if s == "ab\n"));
        let class = parse_gbnf_sequence("[a-z0-9]").unwrap();
        assert!(matches!(&class[0], GbnfElement::CharClass(r) if r.len() == 2));
        let neg = parse_gbnf_sequence("[^x]").unwrap();
        assert!(matches!(&neg[0], GbnfElement::NegCharClass(r) if r.len() == 1));
        let star = parse_gbnf_sequence("rule-name*").unwrap();
        assert!(matches!(&star[0], GbnfElement::Repeat(_, RepeatKind::Star)));
        let plus = parse_gbnf_sequence("rule+").unwrap();
        assert!(matches!(&plus[0], GbnfElement::Repeat(_, RepeatKind::Plus)));
        let opt = parse_gbnf_sequence("rule?").unwrap();
        assert!(matches!(&opt[0], GbnfElement::Optional(_)));
    }

    #[test]
    fn multi_alternative_rule_parses() {
        let rules = parse_gbnf("root ::= \"a\" b | [0-9] | c?").unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].alternatives.len(), 3);
        assert_eq!(rules[0].alternatives[0].len(), 2);
    }

    #[test]
    fn bar_inside_a_literal_or_class_is_not_a_separator() {
        let rules = parse_gbnf(r#"root ::= "a|b" | [|x]"#).unwrap();
        assert_eq!(rules[0].alternatives.len(), 2);
    }
}

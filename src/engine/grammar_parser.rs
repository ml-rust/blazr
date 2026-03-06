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
    /// Character class [a-z0-9]
    CharClass(Vec<CharRange>),
    /// Negated character class [^...]
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

        // Split by | for alternatives
        let alternatives: Vec<GbnfSequence> = body
            .split('|')
            .map(|alt| parse_gbnf_sequence(alt.trim()))
            .collect::<Result<Vec<_>, _>>()?;

        rules.push(GbnfRule { name, alternatives });
    }

    if rules.is_empty() {
        return Err("No rules found in GBNF grammar".to_string());
    }

    Ok(rules)
}

pub(crate) fn parse_gbnf_sequence(input: &str) -> Result<GbnfSequence, String> {
    let mut elements = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            '"' => {
                chars.next();
                let mut literal = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == '"' {
                        chars.next();
                        break;
                    }
                    if ch == '\\' {
                        chars.next();
                        if let Some(&escaped) = chars.peek() {
                            chars.next();
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
                    } else {
                        literal.push(ch);
                        chars.next();
                    }
                }
                elements.push(GbnfElement::Literal(literal));
            }
            '[' => {
                chars.next();
                let negated = chars.peek() == Some(&'^');
                if negated {
                    chars.next();
                }
                let mut ranges = Vec::new();
                while let Some(&ch) = chars.peek() {
                    if ch == ']' {
                        chars.next();
                        break;
                    }
                    let start = ch as u8;
                    chars.next();
                    if chars.peek() == Some(&'-') {
                        chars.next();
                        if let Some(&end_ch) = chars.peek() {
                            chars.next();
                            ranges.push(CharRange {
                                start,
                                end: end_ch as u8,
                            });
                        }
                    } else {
                        ranges.push(CharRange { start, end: start });
                    }
                }
                if negated {
                    elements.push(GbnfElement::NegCharClass(ranges));
                } else {
                    elements.push(GbnfElement::CharClass(ranges));
                }
            }
            ' ' | '\t' => {
                chars.next();
            }
            _ => {
                // Rule reference
                let mut name = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '-' || ch == '_' {
                        name.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if !name.is_empty() {
                    // Check for repetition modifiers
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        elements.push(GbnfElement::Repeat(
                            Box::new(GbnfElement::RuleRef(name)),
                            RepeatKind::Star,
                        ));
                    } else if chars.peek() == Some(&'+') {
                        chars.next();
                        elements.push(GbnfElement::Repeat(
                            Box::new(GbnfElement::RuleRef(name)),
                            RepeatKind::Plus,
                        ));
                    } else if chars.peek() == Some(&'?') {
                        chars.next();
                        elements.push(GbnfElement::Optional(Box::new(GbnfElement::RuleRef(name))));
                    } else {
                        elements.push(GbnfElement::RuleRef(name));
                    }
                }
            }
        }
    }

    Ok(elements)
}

//! The Bonsai 2 (`qwen35`) chat template rendered through blazr, compared
//! byte for byte with Python jinja2.
//!
//! # Fixtures
//!
//! `tests/fixtures/chat_templates/qwen35_bonsai2.jinja` is the GGUF's
//! `tokenizer.chat_template`. Each `qwen35_bonsai2.<case>.<gen|nogen>.txt`
//! beside it is that template rendered by Python jinja2 the way transformers
//! renders it (`ImmutableSandboxedEnvironment(trim_blocks=True,
//! lstrip_blocks=True)`, `raise_exception` raising, `tojson` as
//! `json.dumps(..., ensure_ascii=False)`), with the same messages, tools and
//! variables `cases()` builds here. `render_expected.py` in that directory
//! regenerates them.
//!
//! `bonsai_gguf_ships_the_fixture_template` reads the template out of a real
//! GGUF when `BOOSTR_BONSAI2_DIR` names a directory holding one. Unset or
//! missing, it prints one `skip:` line and passes.

use std::path::{Path, PathBuf};

use blazr::model::chat_template::{
    ChatMessage, ChatTemplate, JinjaTemplate, RenderOptions, TemplateOrigin,
};

const FIXTURE_DIR: &str = "tests/fixtures/chat_templates";
const TEMPLATE: &str = "qwen35_bonsai2";
const BOS: &str = "<|endoftext|>";
const EOS: &str = "<|im_end|>";
const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

fn read(name: &str) -> String {
    let path = fixture_dir().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn template() -> ChatTemplate {
    let source = read(&format!("{TEMPLATE}.jinja"));
    let jinja = JinjaTemplate::new(&source, TemplateOrigin::Gguf, BOS, EOS)
        .unwrap_or_else(|e| panic!("compile {TEMPLATE}: {e}"));
    ChatTemplate::Jinja(jinja)
}

fn user(content: &str) -> ChatMessage {
    ChatMessage::new("user", content)
}

fn tools() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "properties": {
                    "city": {"description": "City name", "type": "string"},
                    "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}
                },
                "required": ["city"],
                "type": "object"
            }
        }
    })]
}

struct Case {
    name: &'static str,
    messages: Vec<ChatMessage>,
    options: RenderOptions,
}

/// The same inputs `render_expected.py` renders, one entry per fixture.
fn cases() -> Vec<Case> {
    let with = |tools: Option<Vec<serde_json::Value>>, extra: &[(&str, serde_json::Value)]| {
        let mut options = RenderOptions {
            tools,
            ..RenderOptions::default()
        };
        for (key, value) in extra {
            options.extra.insert((*key).to_string(), value.clone());
        }
        options
    };
    let tool_call = ChatMessage {
        role: "assistant".into(),
        content: String::new(),
        tool_calls: Some(vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"
            }
        })]),
        ..ChatMessage::default()
    };
    let tool_result = ChatMessage {
        name: Some("get_weather".into()),
        ..ChatMessage::new("tool", "18 degrees, cloudy")
    };
    vec![
        Case {
            name: "system_user",
            messages: vec![
                ChatMessage::new("system", "You are a terse assistant."),
                user("What is 2+2?"),
            ],
            options: with(None, &[]),
        },
        Case {
            name: "user_only",
            messages: vec![user("Hi")],
            options: with(None, &[]),
        },
        Case {
            name: "multi_turn_think",
            messages: vec![
                user("Name a prime."),
                ChatMessage::new("assistant", "<think>\nSmallest prime is 2.\n</think>\n\n2"),
                user("Another one?"),
            ],
            options: with(None, &[]),
        },
        Case {
            name: "tools",
            messages: vec![user("Weather in Paris?")],
            options: with(Some(tools()), &[]),
        },
        Case {
            name: "tool_call_round_trip",
            messages: vec![user("Weather in Paris?"), tool_call, tool_result],
            options: with(Some(tools()), &[]),
        },
        Case {
            name: "thinking_off",
            messages: vec![user("Hi")],
            options: with(None, &[("enable_thinking", serde_json::json!(false))]),
        },
        Case {
            name: "effort_low",
            messages: vec![user("Hi")],
            options: with(None, &[("reasoning_effort", serde_json::json!("low"))]),
        },
    ]
}

#[test]
fn renders_match_python_jinja2() {
    let template = template();
    for case in cases() {
        for add_generation_prompt in [true, false] {
            let options = RenderOptions {
                add_generation_prompt,
                ..case.options.clone()
            };
            let suffix = if add_generation_prompt {
                "gen"
            } else {
                "nogen"
            };
            let expected = read(&format!("{TEMPLATE}.{}.{suffix}.txt", case.name));
            let rendered = template
                .apply_with(&case.messages, &options)
                .unwrap_or_else(|e| panic!("{} gen={add_generation_prompt}: {e}", case.name));
            assert_eq!(
                rendered, expected,
                "{} gen={add_generation_prompt}",
                case.name
            );
        }
    }
}

/// The real template is not ChatML: a plain user turn gets the reasoning
/// system turn and opens the assistant turn with `<think>`.
#[test]
fn plain_user_turn_is_chatml_plus_thinking() {
    let rendered = template().apply(&[user("Hi")]).expect("renders");
    assert_eq!(
        rendered,
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through \
         the task, validate key assumptions, consider plausible alternatives, and prioritize \
         correctness, consistency, and clarity in the final answer.<|im_end|>\n\
         <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    assert!(rendered.contains("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"));
    let chatml = ChatTemplate::ChatML.apply(&[user("Hi")]).expect("renders");
    assert_ne!(rendered, chatml);
}

#[test]
fn template_refusals_are_errors_that_name_the_template() {
    let template = template();
    let err = template.apply(&[]).expect_err("no messages");
    let text = err.to_string();
    assert!(text.starts_with("chat template (gguf jinja):"), "{text}");
    assert!(text.contains("No messages provided."), "{text}");

    let err = template
        .apply(&[ChatMessage::new("assistant", "x")])
        .expect_err("no user turn");
    assert!(err.to_string().contains("No user query found in messages."));

    let mut options = RenderOptions::default();
    options
        .extra
        .insert("reasoning_effort".into(), serde_json::json!("ultra"));
    let err = template
        .apply_with(&[user("x")], &options)
        .expect_err("bad effort");
    assert!(err
        .to_string()
        .contains("Unexpected reasoning effort ultra"));
}

#[test]
fn bonsai_gguf_ships_the_fixture_template() {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return;
    };
    let Some(gguf) = std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.extension().is_some_and(|x| x == "gguf")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("Ternary-Bonsai-2-27B-P"))
        })
    else {
        println!(
            "skip: no Ternary-Bonsai-2-27B-P*.gguf under {}",
            dir.display()
        );
        return;
    };
    let shipped = ChatTemplate::from_model(&gguf)
        .unwrap_or_else(|e| panic!("read {}: {e}", gguf.display()))
        .unwrap_or_else(|| panic!("{} ships no chat template", gguf.display()));
    assert_eq!(shipped.origin, TemplateOrigin::Gguf);
    assert_eq!(shipped.bos_token, BOS);
    assert_eq!(shipped.eos_token, EOS);
    assert_eq!(shipped.source, read(&format!("{TEMPLATE}.jinja")));
    let template = ChatTemplate::detect(&gguf, "qwen35", None).expect("detects");
    assert_eq!(template.describe(), "gguf jinja");
    assert_eq!(
        template.apply(&[user("Hi")]).expect("renders"),
        read(&format!("{TEMPLATE}.user_only.gen.txt"))
    );
}

//! The chat path of real checkpoints against Hugging Face `transformers`.
//!
//! `arch_reference` proves the forward pass on a raw prompt. It encodes with
//! every special token on, runs no chat template, and never checks a stop. A
//! wrong end-of-turn id and a refused prompt pass it. This file covers what it
//! leaves out:
//!
//! - the prompt ids of `encode_chat_prompt`, the function every chat endpoint
//!   calls, against `apply_chat_template` or the built-in format
//! - the ids that end generation, against `generation_config.json`
//! - one real greedy turn that has to end on its end-of-turn id
//!
//! # Fixtures
//!
//! Each test reads `$BOOSTR_MODELS_DIR/<entry>`. A test with no checkpoint
//! prints one skip line and passes.

use std::path::{Path, PathBuf};

use futures::StreamExt;

use blazr::config::{DtypeRequest, DtypeSource};
use blazr::engine::FinishReason;
use blazr::loader::load_model_as;
use blazr::model::chat_template::{encode_chat_prompt, ChatMessage, ChatTemplate, RenderOptions};
use blazr::tokenizer::checkpoint_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::{CpuDevice, CpuRuntime};
use splintr::SpecialMode;

mod common;
use common::model_fixture;

const SYSTEM: &str = "You are a podcast guest. Reply in two spoken sentences. Plain speech only.";
const USER: &str = "What makes a city a smart city?";

/// `apply_chat_template` ids of [`SYSTEM`] and [`USER`] on Qwen2.5 and Qwen3.
const CHATML_IDS: &[u32] = &[
    151644, 8948, 198, 2610, 525, 264, 17711, 8640, 13, 17841, 304, 1378, 21355, 22870, 13, 43199,
    8806, 1172, 13, 151645, 198, 151644, 872, 198, 3838, 3643, 264, 3283, 264, 7785, 3283, 30,
    151645, 198, 151644, 77091, 198,
];
/// What `enable_thinking=False` adds: `<think>\n\n</think>\n\n`.
const EMPTY_THINK_IDS: &[u32] = &[151667, 271, 151668, 271];
/// `apply_chat_template` ids on Phi-3.5.
const PHI3_IDS: &[u32] = &[
    32006, 887, 526, 263, 2532, 4384, 17838, 29889, 10088, 368, 297, 1023, 19182, 25260, 29889,
    349, 7420, 12032, 871, 29889, 32007, 32010, 1724, 3732, 263, 4272, 263, 15040, 4272, 29973,
    32007, 32001,
];
/// The ids `transformers` gives the built-in Gemma format. This checkpoint
/// ships no template.
const GEMMA_IDS: &[u32] = &[
    2, 106, 9020, 108, 2045, 708, 476, 23679, 15443, 235265, 12956, 575, 1378, 22230, 26099,
    235265, 18685, 11360, 1297, 235265, 107, 108, 106, 1645, 108, 1841, 3833, 476, 3413, 476, 7550,
    3413, 235336, 107, 108, 106, 2516, 108,
];
/// Qwen3 with `enable_thinking=False`, `transformers`, CPU, F32, greedy, no
/// repeat penalty. It ends on `<|im_end|>`, id 151645.
const QWEN3_GREEDY: &[u32] = &[
    32, 7785, 3283, 5711, 5440, 311, 7269, 3516, 11, 17903, 11, 323, 5382, 369, 10826, 13, 1084,
    594, 911, 3259, 2513, 803, 11050, 323, 2664, 369, 1251, 13, 151645,
];

fn require(entry: &str) -> Option<PathBuf> {
    let dir = model_fixture("", entry);
    if dir.is_none() {
        eprintln!("skipping: {entry} is absent from BOOSTR_MODELS_DIR");
    }
    dir
}

fn messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::new("system", SYSTEM),
        ChatMessage::new("user", USER),
    ]
}

fn config_field(dir: &Path, field: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("config.json")).expect("config.json reads");
    let config: serde_json::Value = serde_json::from_str(&text).expect("config.json parses");
    config[field].clone()
}

/// The tokenizer and template the server loads for `dir`.
fn chat_parts(dir: &Path) -> (splintr::AnyTokenizer, ChatTemplate) {
    let vocab = config_field(dir, "vocab_size")
        .as_u64()
        .expect("vocab_size") as usize;
    let model_type = config_field(dir, "model_type");
    let tokenizer = checkpoint_tokenizer(dir, vocab).expect("tokenizer loads");
    let mut template = ChatTemplate::detect(dir, model_type.as_str().expect("model_type"), None)
        .expect("template loads");
    template.bind_special_tokens(&tokenizer);
    (tokenizer, template)
}

fn no_thinking() -> RenderOptions {
    let mut options = RenderOptions::default();
    options
        .extra
        .insert("enable_thinking".into(), serde_json::Value::Bool(false));
    options
}

/// Check the prompt ids and the stop ids of one checkpoint.
fn check_case(entry: &str, options: &RenderOptions, prompt: &[u32], stops: &[u32]) {
    let Some(dir) = require(entry) else {
        return;
    };
    let (tokenizer, template) = chat_parts(&dir);
    let (_, ids) = encode_chat_prompt(&template, &tokenizer, &messages(), "", options)
        .unwrap_or_else(|e| panic!("{entry}: a clean chat request was refused: {e}"));
    assert_eq!(ids, prompt, "{entry}: prompt ids");
    assert_eq!(tokenizer.eos_token_ids(), stops, "{entry}: stop ids");
}

/// The prompt ids match. `generation_config.json` states only
/// `<|endoftext|>`, because this is the base model and not `-Instruct`.
#[test]
fn qwen25_chat_prompt_and_stops() {
    check_case(
        "qwen2.5-0.5b",
        &RenderOptions::default(),
        CHATML_IDS,
        &[151643, 151645],
    );
}

/// `<|im_end|>` is the primary EOS. splintr guesses `<|endoftext|>`.
#[test]
fn qwen3_chat_prompt_and_stops() {
    check_case(
        "qwen3-1.7b",
        &RenderOptions::default(),
        CHATML_IDS,
        &[151645, 151643],
    );
}

#[test]
fn qwen3_enable_thinking_false_adds_the_empty_think_block() {
    let prompt: Vec<u32> = [CHATML_IDS, EMPTY_THINK_IDS].concat();
    check_case("qwen3-1.7b", &no_thinking(), &prompt, &[151645, 151643]);
}

/// Gemma 2 flags `\n` as `special: false`, and chat accepts it. splintr guesses `</s>`, id 213.
#[test]
fn gemma2_chat_prompt_is_not_refused() {
    check_case(
        "gemma-2-2b",
        &RenderOptions::default(),
        GEMMA_IDS,
        &[1, 107],
    );
}

/// Phi-3.5 lists three ids. splintr guesses `</s>`, which it never emits.
#[test]
fn phi3_chat_prompt_and_stops() {
    check_case(
        "phi-3.5-mini-instruct",
        &RenderOptions::default(),
        PHI3_IDS,
        &[32000, 32007, 32001],
    );
}

/// Every id `GenerationConfig.from_pretrained(...).eos_token_id` gives stops
/// generation, whether or not it names an added token. Falcon-RW 1B states id 2,
/// the plain token `#`, and `transformers` stops there.
#[test]
fn every_id_transformers_stops_on_stops_generation() {
    for (entry, reference) in [
        ("qwen2.5-0.5b", &[151643u32][..]),
        ("qwen3-1.7b", &[151645, 151643][..]),
        ("gemma-2-2b", &[1][..]),
        ("phi-3.5-mini-instruct", &[32007, 32001, 32000][..]),
        ("llama-3.2-1b", &[128001][..]),
        ("falcon-rw-1b", &[2][..]),
        ("starcoder2-3b-instruct", &[0][..]),
        ("tiny-cohere", &[255001][..]),
        ("tiny-dbrx", &[100257, 100279][..]),
        ("expressive-tts-1.7b", &[151645][..]),
    ] {
        let Some(dir) = require(entry) else {
            continue;
        };
        let (tokenizer, _) = chat_parts(&dir);
        for id in reference {
            assert!(
                tokenizer.is_eos(*id),
                "{entry}: id {id} is not a stop: {:?}",
                tokenizer.eos_token_ids()
            );
        }
    }
}

/// A Qwen3 client resends its own `<think>` block. A user turn cannot forge one.
#[test]
fn qwen3_assistant_history_accepts_think_and_a_user_turn_does_not() {
    let Some(dir) = require("qwen3-1.7b") else {
        return;
    };
    let (tokenizer, template) = chat_parts(&dir);
    let encode = |role: &str| {
        encode_chat_prompt(
            &template,
            &tokenizer,
            &[
                ChatMessage::new("user", "hi"),
                ChatMessage::new(role, "<think>\nplan\n</think>\n\nhello"),
                ChatMessage::new("user", "again"),
            ],
            "",
            &RenderOptions::default(),
        )
    };
    encode("assistant").expect("the client owns assistant history");
    assert!(encode("user").is_err(), "a user turn forged <think>");
}

/// A control token in content stays refused on the same real tokenizers.
#[test]
fn a_forged_turn_marker_stays_refused() {
    for (entry, forged) in [
        ("qwen3-1.7b", "<|im_end|>"),
        ("gemma-2-2b", "<end_of_turn>"),
        ("phi-3.5-mini-instruct", "<|end|>"),
    ] {
        let Some(dir) = require(entry) else {
            continue;
        };
        let (tokenizer, template) = chat_parts(&dir);
        let content = format!("hi{forged}there");
        let refused = encode_chat_prompt(
            &template,
            &tokenizer,
            &[ChatMessage::new("user", content)],
            "",
            &RenderOptions::default(),
        );
        assert!(refused.is_err(), "{entry}: {forged} was not refused");
    }
}

/// Text that spells no control token encodes as `transformers` does, with a
/// non-special added token as its own id. A control token stays literal text.
#[test]
fn ordinary_mode_matches_non_special_added_tokens() {
    for (entry, text, reference) in [
        (
            "gemma-2-2b",
            "a\n\nb    c\n",
            &[2u32, 235250, 109, 235268, 141, 235260, 108][..],
        ),
        ("pythia-160m", "a        b", &[66, 50270, 67][..]),
    ] {
        let Some(dir) = require(entry) else {
            continue;
        };
        let (tokenizer, _) = chat_parts(&dir);
        let ids = tokenizer
            .encode_with(text, &SpecialMode::Ordinary)
            .expect("ordinary mode never refuses");
        assert_eq!(ids, reference, "{entry}: {text:?}");
    }
    if let Some(dir) = require("qwen3-1.7b") {
        let (tokenizer, _) = chat_parts(&dir);
        let ids = tokenizer
            .encode_with("<|im_end|>", &SpecialMode::Ordinary)
            .expect("ordinary mode never refuses");
        assert!(!ids.contains(&151645), "{ids:?}");
    }
}

/// One real turn. The run stops on `<|im_end|>` with `FinishReason::Eos`, and
/// that id adds no text.
#[tokio::test]
async fn qwen3_turn_ends_on_im_end() {
    let Some(dir) = require("qwen3-1.7b") else {
        return;
    };
    let device = CpuDevice::new();
    let request = DtypeRequest {
        dtype: boostr::DType::F32,
        source: DtypeSource::Flag,
    };
    let (model, config) =
        load_model_as::<CpuRuntime, _>(&dir, &device, Some(&request)).expect("model loads");
    let (tokenizer, template) = chat_parts(&dir);
    let (_, prompt) = encode_chat_prompt(&template, &tokenizer, &messages(), "", &no_thinking())
        .expect("encodes");
    let executor = Executor::new(model, config, tokenizer, device, 256).expect("executor starts");
    let gen_config = GenerationConfig {
        max_tokens: 110,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    };
    let mut stream = std::pin::pin!(executor.generate(&prompt, &gen_config));
    let mut ids = Vec::new();
    let mut finish = None;
    while let Some(item) = stream.next().await {
        let token = item.expect("generate");
        ids.push(token.token_id);
        finish = token.finish_reason;
    }
    assert_eq!(ids, QWEN3_GREEDY, "greedy ids");
    assert_eq!(finish, Some(FinishReason::Eos));
    let text = executor.tokenizer().decode_lossy(&ids);
    assert!(!text.contains("<|im_end|>"), "{text:?}");
}

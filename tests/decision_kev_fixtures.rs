//! Kev's prompt and wire answers without weights: each fixture's request,
//! rendered by blazr, gives Kev's token rows, decide positions and
//! option-end positions exactly, and Kev's own served probabilities,
//! through boostr's answer stage and blazr's Kev formatter, give Kev's wire
//! answers exactly.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the Kev checkpoint directory.
//! - `BLAZR_DECISION_FIXTURE_DIR`: its `kev_*.json` fixtures, schema
//!   `boostr.decision.pointer.v1`.
//! - `BLAZR_DECISION_BASE` (optional): the base checkpoint. Its
//!   `tokenizer.json` encodes the prompts, as Kev and the server do; unset,
//!   the copy in the Kev directory does.
//!
//! Unset or missing, each test prints one `skip:` line and passes.

#![cfg(feature = "server")]

mod decision_kev;

use blazr::boostr::inference::decision::{
    answer_from_probs, AnswerOptions, DecisionResult, QuestionKind, RowSlot,
};
use blazr::model::decision_template::kev::packed_tokens;
use blazr::model::decision_template::{DecisionRenderer, Family, JevRequest, KevRenderer};
use blazr::server::decision::answers_json;
use serde_json::Value;

use decision_kev::fixture::{env_path, model_and_fixtures, Fixture, ENV_BASE};

/// The renderer the server builds: over the base's tokenizer when
/// `BLAZR_DECISION_BASE` is set, else the Kev directory's copy.
fn renderer(dir: &std::path::Path) -> KevRenderer {
    let base = env_path(ENV_BASE).filter(|b| b.join("tokenizer.json").is_file());
    let tokenizer_dir = base.as_deref().unwrap_or(dir);
    println!(
        "tokenizer: {}",
        tokenizer_dir.join("tokenizer.json").display()
    );
    KevRenderer::load(tokenizer_dir)
        .unwrap_or_else(|e| panic!("load {}: {e}", tokenizer_dir.display()))
}

fn parse(fx: &Fixture) -> JevRequest {
    JevRequest::parse(&fx.request, Family::Kev)
        .unwrap_or_else(|e| panic!("{}: parse: {e}", fx.label()))
}

#[test]
fn rendering_reproduces_kev_rows_and_positions() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let renderer = renderer(&dir);
    for fx in &fixtures {
        let what = fx.label();
        let d = renderer.delimiters();
        let s = &fx.special_tokens;
        assert_eq!(
            [d.state, d.question, d.option, d.option_end, d.decide],
            [s.state, s.question, s.option, s.option_end, s.decide],
            "{what}: delimiter ids"
        );
        let out = renderer
            .render(&parse(fx))
            .unwrap_or_else(|e| panic!("{what}: render: {e}"));
        assert_eq!(out.family, Family::Kev);
        assert_eq!(out.rows.len(), fx.rows.len(), "{what}: row count");
        for (r, (got, want)) in out.rows.iter().zip(&fx.rows).enumerate() {
            assert_eq!(got, want, "{what}: row {r}");
        }
        assert_eq!(out.state_tokens, fx.state_len, "{what}: state tokens");
        assert_eq!(
            packed_tokens(&out),
            fx.input_tokens(),
            "{what}: input tokens"
        );
        assert!(out.warnings.is_empty(), "{what}: {:?}", out.warnings);
        assert_eq!(out.questions.len(), fx.questions.len(), "{what}: questions");
        for (i, fq) in fx.questions.iter().enumerate() {
            let q = &out.questions[i];
            let sr = &out.scoring_rows[i];
            let at = format!("{what} {}", fq.id);
            assert_eq!(q.id, fq.id, "{at}: order");
            assert_eq!(q.names, fq.keys, "{at}: keys");
            assert_eq!(q.meta.options, fq.options, "{at}: options");
            let kind = match &q.meta.kind {
                QuestionKind::Noul { yes: 1 } => "noul",
                QuestionKind::Choice if q.legend.is_empty() => "choice",
                QuestionKind::Choice | QuestionKind::Score { .. } => "score",
                other => panic!("{at}: kind {other:?}"),
            };
            assert_eq!(kind, fq.kind, "{at}: kind");
            assert_eq!(
                sr.slot,
                RowSlot {
                    row: fq.row,
                    pos: fq.decide
                },
                "{at}: decide"
            );
            assert_eq!(sr.option_ends, fq.option_ends, "{at}: option ends");
            assert!(sr.labels.is_empty(), "{at}: a pointer row has no labels");
            assert_eq!(q.scoring_rows, i..i + 1, "{at}: scoring rows");
        }
        println!("ok: {what} renders Kev's rows");
    }
}

/// The renderer `DecisionRenderer::load` builds from the Kev directory
/// alone is a Kev renderer over the directory's tokenizer copy.
#[test]
fn a_kev_directory_loads_a_kev_renderer() {
    let Some((dir, _)) = model_and_fixtures() else {
        return;
    };
    let loaded =
        DecisionRenderer::load(&dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    assert_eq!(loaded.family(), Family::Kev);
}

/// The fixtures' served probabilities are `softmax(raw / temperature)`:
/// the one fixed temperature the server answers with.
#[test]
fn served_probabilities_are_the_tempered_raw_logits() {
    let Some((_, fixtures)) = model_and_fixtures() else {
        return;
    };
    for fx in &fixtures {
        for fq in &fx.questions {
            let what = format!("{} {}", fx.label(), fq.id);
            let max = fq
                .raw_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = fq
                .raw_logits
                .iter()
                .map(|z| ((z - max) / fx.temperature).exp())
                .collect();
            let total: f64 = e.iter().sum();
            for (j, (x, p)) in e.iter().zip(&fq.probs).enumerate() {
                let want = x / total;
                assert!((want - p).abs() < 1e-5, "{what} p[{j}]: {want} vs {p}");
            }
        }
    }
}

#[test]
fn formatter_reproduces_kev_answers() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let renderer = renderer(&dir);
    for fx in &fixtures {
        let what = fx.label();
        let out = renderer
            .render(&parse(fx))
            .unwrap_or_else(|e| panic!("{what}: render: {e}"));
        let answers = out
            .questions
            .iter()
            .zip(&fx.questions)
            .map(|(q, fq)| {
                answer_from_probs(&q.meta, fq.probs.clone(), &AnswerOptions::default())
                    .unwrap_or_else(|e| panic!("{what} {}: answer: {e}", q.id))
            })
            .collect();
        let result = DecisionResult {
            answers,
            raw_logits: Vec::new(),
            temperatures: Vec::new(),
        };
        let got = answers_json(Family::Kev, &out.questions, &result)
            .unwrap_or_else(|e| panic!("{what}: format: {e}"));
        assert_eq!(
            Value::Object(got),
            Value::Object(fx.reference_answers.clone()),
            "{what}: answers"
        );
        println!("ok: {what} formats Kev's answers");
    }
}

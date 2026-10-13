//! The decision prompt renderers against the reference fixtures.
//!
//! `BLAZR_DECISION_FIXTURE_DIR` names a directory of one family's fixtures
//! (schema `boostr.decision.label_token.v2`, written by the reference
//! harness `ref_label_token.py` from the family's own renderer).
//! `BLAZR_DECISION_MODEL_DIR` names that family's model directory, holding
//! `tokenizer.json`, the chat template and the family's sidecar
//! (`prompt_contract.json` or `decider_config.json`).
//!
//! For every fixture, rendering its `request` must give its `rows` token
//! for token, and every scoring row's slot, label ids and question, and
//! every question's kind, option count and scoring rows. The rows must also
//! plan through boostr's `plan_from_rows` with the fixture's shared prefix.
//!
//! Either variable unset: one `skip:` line, and the test passes.

use std::path::{Path, PathBuf};

use blazr::boostr::inference::decision::{plan_from_rows, QuestionKind, RowSlot};
use blazr::boostr::model::decision::LabelTokenItem;
use blazr::model::decision_template::{DecisionRenderer, JevRequest, RenderedDecision};
use serde_json::Value;

const SCHEMA: &str = "boostr.decision.label_token.v2";

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

#[test]
fn decision_template_fixtures() {
    let (Some(fixtures), Some(model)) = (
        env_dir("BLAZR_DECISION_FIXTURE_DIR"),
        env_dir("BLAZR_DECISION_MODEL_DIR"),
    ) else {
        println!(
            "skip: set BLAZR_DECISION_FIXTURE_DIR and BLAZR_DECISION_MODEL_DIR to check the \
             decision renderers against the reference fixtures"
        );
        return;
    };
    let renderer =
        DecisionRenderer::load(&model).unwrap_or_else(|e| panic!("load {}: {e}", model.display()));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixtures.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no fixtures in {}", fixtures.display());

    let mut failures = Vec::new();
    for path in &files {
        match check(&renderer, path) {
            Ok(()) => println!("ok: {}", path.display()),
            Err(msg) => failures.push(format!("{}: {msg}", path.display())),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} fixtures differ:\n{}",
        failures.len(),
        files.len(),
        failures.join("\n")
    );
}

fn check(renderer: &DecisionRenderer, path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let fx: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    if fx["schema"] != SCHEMA {
        return Err(format!("schema {} is not {SCHEMA}", fx["schema"]));
    }
    let family = renderer.family();
    if fx["family"] != family.name() {
        return Err(format!(
            "fixture family {} but the model directory is {}",
            fx["family"],
            family.name()
        ));
    }
    let request = JevRequest::parse(&fx["request"], family).map_err(|e| format!("parse: {e}"))?;
    let out = renderer
        .render(&request)
        .map_err(|e| format!("render: {e}"))?;
    check_rows(&fx, &out)?;
    check_scoring_rows(&fx, &out)?;
    check_questions(&fx, &out)?;
    check_plan(&fx, &out)
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_default()
}

fn usize_at(value: &Value) -> Option<usize> {
    value.as_u64().map(|v| v as usize)
}

fn check_rows(fx: &Value, out: &RenderedDecision) -> Result<(), String> {
    let want: Vec<Vec<u32>> = fx["rows"]
        .as_array()
        .ok_or("no rows")?
        .iter()
        .map(ids)
        .collect();
    if want.len() != out.rows.len() {
        return Err(format!(
            "{} rows, the reference has {}",
            out.rows.len(),
            want.len()
        ));
    }
    for (r, (got, want)) in out.rows.iter().zip(&want).enumerate() {
        if got != want {
            let at = got
                .iter()
                .zip(want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            let lo = at.saturating_sub(3);
            return Err(format!(
                "row {r}: {} tokens, the reference has {}; first difference at {at}: \
                 ours {:?}, reference {:?}",
                got.len(),
                want.len(),
                &got[lo..(at + 4).min(got.len())],
                &want[lo..(at + 4).min(want.len())]
            ));
        }
    }
    Ok(())
}

fn check_scoring_rows(fx: &Value, out: &RenderedDecision) -> Result<(), String> {
    let want = fx["scoring_rows"].as_array().ok_or("no scoring_rows")?;
    if want.len() != out.scoring_rows.len() {
        return Err(format!(
            "{} scoring rows, the reference has {}",
            out.scoring_rows.len(),
            want.len()
        ));
    }
    for (i, (got, want)) in out.scoring_rows.iter().zip(want).enumerate() {
        let expected = (
            usize_at(&want["row"]),
            usize_at(&want["slot"]),
            ids(&want["labels"]),
            usize_at(&want["question"]),
        );
        let ours = (
            Some(got.slot.row),
            Some(got.slot.pos),
            got.labels.clone(),
            Some(got.question),
        );
        if ours != expected {
            return Err(format!(
                "scoring row {i}: (row, slot, labels, question) {ours:?}, reference {expected:?}"
            ));
        }
    }
    Ok(())
}

fn check_questions(fx: &Value, out: &RenderedDecision) -> Result<(), String> {
    let want = fx["questions"].as_array().ok_or("no questions")?;
    if want.len() != out.questions.len() {
        return Err(format!(
            "{} questions, the reference has {}",
            out.questions.len(),
            want.len()
        ));
    }
    for (i, (got, want)) in out.questions.iter().zip(want).enumerate() {
        let (kind, isolated, yes) = match &got.meta.kind {
            QuestionKind::Choice => ("choice", false, None),
            QuestionKind::Noul { yes } => ("noul", false, Some(*yes)),
            QuestionKind::Score { isolated, .. } => ("score", *isolated, None),
            _ => ("unknown", false, None),
        };
        let ours = (
            got.id.as_str(),
            kind,
            isolated,
            got.meta.options,
            yes,
            got.scoring_rows.start,
            got.scoring_rows.len(),
        );
        let span = want["scoring_rows"].as_array().cloned().unwrap_or_default();
        let expected = (
            want["id"].as_str().unwrap_or(""),
            want["kind"].as_str().unwrap_or(""),
            want["isolated"].as_bool().unwrap_or(false),
            usize_at(&want["options"]).unwrap_or(0),
            usize_at(&want["yes_index"]),
            span.first().and_then(usize_at).unwrap_or(usize::MAX),
            span.get(1).and_then(usize_at).unwrap_or(usize::MAX),
        );
        if ours != expected {
            return Err(format!(
                "question {i}: (id, kind, isolated, options, yes_index, first, count) \
                 {ours:?}, reference {expected:?}"
            ));
        }
        if got.meta.scoring_rows() != got.scoring_rows.len() {
            return Err(format!(
                "question {i}: QuestionMeta expects {} scoring rows, the render gives {}",
                got.meta.scoring_rows(),
                got.scoring_rows.len()
            ));
        }
    }
    Ok(())
}

/// The rows plan through boostr with the fixture's shared prefix, and each
/// scoring row becomes a label-token item over its slot's gather.
fn check_plan(fx: &Value, out: &RenderedDecision) -> Result<(), String> {
    let slots: Vec<RowSlot> = out.scoring_rows.iter().map(|s| s.slot).collect();
    let planned =
        plan_from_rows(out.rows.clone(), &slots, 0).map_err(|e| format!("plan_from_rows: {e}"))?;
    let want = usize_at(&fx["shared_prefix_len"]).ok_or("no shared_prefix_len")?;
    if planned.plan.prefix_len() != want {
        return Err(format!(
            "plan prefix {} tokens, the reference shares {want}",
            planned.plan.prefix_len()
        ));
    }
    let items: Vec<LabelTokenItem> = out
        .scoring_rows
        .iter()
        .zip(&planned.slot_gathers)
        .map(|(s, &g)| LabelTokenItem {
            slot: g,
            labels: s.labels.clone(),
        })
        .collect();
    if items.len() != slots.len() {
        return Err("a scoring row has no gather".to_string());
    }
    Ok(())
}

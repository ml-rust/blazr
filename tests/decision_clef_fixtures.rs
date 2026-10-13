//! Clef's prompt and head without the backbone. Each fixture's request,
//! rendered by blazr, gives the reference's record ids and every span
//! exactly. The release head, run through blazr's own path (prepare, the
//! joint-schema readout, boostr's decision run, the Clef formatter) over a
//! stand-in backbone that serves the fixture's saved hidden rows and
//! `lm_head` rows, gives the reference's answers.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the Clef release directory. Only its
//!   `tokenizer.json`, `joint_head_config.json` and
//!   `joint_head.safetensors` are read; the backbone weights are not.
//! - `BLAZR_DECISION_FIXTURE_DIR`: the `*.json` fixtures (schema
//!   `boostr.decision.joint.v1`) and their `.npy` arrays.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! # Tolerances
//!
//! blazr runs the head in F32 on the fixture's BF16 hidden rows, as the
//! reference's F32 head does: raw logits within 2e-3 and probabilities
//! within 1e-3, the bands of boostr's `decision_joint_schema_real_file`.
//! Against the release's BF16 head, probabilities within 3e-3: the
//! measured gap between the two reference heads is 0.0019 (`multi`).

#![cfg(feature = "server")]

mod decision_clef;

use blazr::boostr::error::{Error, Result};
use blazr::boostr::inference::decision::{
    DecisionBackbone, PrefixState, RunnerConfig, TemperaturePolicy,
};
use blazr::boostr::inference::{LayeredGdnState, LayeredKvCache};
use blazr::boostr::model::decision::{FieldKind, HeadDtype, JointSchemaHead};
use blazr::boostr::model::{GdnConfig, ModelClient};
use blazr::boostr::{CpuDevice, CpuRuntime, DType, Runtime, Tensor};
use blazr::config::DecisionConfig;
use blazr::engine::decision::{prepare, PrefixRule, ReadoutHead};
use blazr::model::decision_template::clef_text::render;
use blazr::model::decision_template::{DecisionRenderer, Family, JevRequest, RenderedDecision};
use blazr::server::decision::clef_answers;

use decision_clef::compare;
use decision_clef::fixture::{model_and_fixtures, Fixture};

const F32_LOGIT: f64 = 2e-3;
const F32_PROB: f64 = 1e-3;
const RELEASE_PROB: f64 = 3e-3;

fn renderer(dir: &std::path::Path) -> DecisionRenderer {
    let r = DecisionRenderer::load(dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    assert_eq!(r.family(), Family::Clef, "{}", dir.display());
    r
}

fn render_fixture(r: &DecisionRenderer, fx: &Fixture) -> RenderedDecision {
    let request = JevRequest::parse(&fx.request, Family::Clef)
        .unwrap_or_else(|e| panic!("{}: parse: {e}", fx.label()));
    r.render(&request)
        .unwrap_or_else(|e| panic!("{}: render: {e}", fx.label()))
}

fn kind_id(kind: FieldKind) -> usize {
    kind.type_id()
}

#[test]
fn rendering_reproduces_clef_records() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let r = renderer(&dir);
    for fx in &fixtures {
        let what = fx.label();
        let joined: Vec<u32> = fx.segments.iter().flat_map(|s| s.ids.clone()).collect();
        assert_eq!(joined, fx.input_ids, "{what}: the fixture's pieces");
        assert_eq!(
            render(&fx.request["state"]),
            fx.segments[1].text,
            "{what}: the state's JSON text"
        );

        let out = render_fixture(&r, fx);
        assert_eq!(out.family, Family::Clef);
        assert_eq!(out.rows.len(), 1, "{what}: one record");
        assert_eq!(out.rows[0], fx.input_ids, "{what}: record ids");
        assert_eq!(out.state_tokens, fx.segments[1].ids.len(), "{what}: state");
        assert!(out.warnings.is_empty(), "{what}: {:?}", out.warnings);
        assert_eq!(
            fx.answers["usage"]["input_tokens"],
            serde_json::json!(out.rows[0].len()),
            "{what}: usage"
        );
        let layout = out.joint.as_ref().expect("a joint layout");
        assert_eq!(layout.fields.len(), fx.questions.len(), "{what}: fields");
        for (i, fq) in fx.questions.iter().enumerate() {
            let at = format!("{what} {}", fq.id);
            let (q, f) = (&out.questions[i], &layout.fields[i]);
            assert_eq!(q.id, fq.id, "{at}: order");
            assert_eq!(kind_id(f.item.kind), fq.type_id, "{at}: type {}", fq.kind);
            assert_eq!(
                [f.item.question.start, f.item.question.end],
                fq.question_span,
                "{at}: question span"
            );
            let spans: Vec<[usize; 2]> = f
                .item
                .options
                .iter()
                .map(|o| [o.span.start, o.span.end])
                .collect();
            assert_eq!(spans, fq.option_spans, "{at}: option spans");
            let ids: Vec<Vec<u32>> = f.item.options.iter().map(|o| o.ids.clone()).collect();
            assert_eq!(ids, fq.option_token_ids, "{at}: option ids");
            assert_eq!(q.names, fq.option_ids, "{at}: option ids in head order");
            assert_eq!(q.meta.options, fq.option_ids.len(), "{at}: options");
            assert_eq!(
                out.scoring_rows[i].slot.pos, fx.global_row,
                "{at}: global row"
            );
        }
        println!("ok: {what} renders the reference's record");
    }
}

/// A backbone that serves one fixture's saved rows: the hidden rows its
/// prefill keeps, and the `lm_head` rows of its option tokens.
struct FixtureBackbone {
    device: CpuDevice,
    hidden: Vec<f32>,
    width: usize,
    positions: usize,
    row_ids: Vec<u32>,
    rows: Vec<f32>,
    gdn: GdnConfig,
}

fn refused(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "backbone",
        reason,
    }
}

impl FixtureBackbone {
    fn new(fx: &Fixture, device: &CpuDevice) -> Self {
        let (hidden, width) = fx.hidden_rows();
        let gdn: GdnConfig = serde_json::from_value(serde_json::json!({"hidden_size": 1,
            "state_size": 1, "key_heads": 1, "value_heads": 1, "inner_size": 1}))
        .expect("a GDN config");
        Self {
            device: device.clone(),
            positions: hidden.len() / width,
            hidden,
            width,
            row_ids: fx.lm_head.row_ids.clone(),
            rows: fx.lm_rows(),
            gdn,
        }
    }
}

impl DecisionBackbone<CpuRuntime> for FixtureBackbone {
    fn max_positions(&self) -> usize {
        262_144
    }

    fn vocab_size(&self) -> usize {
        248_320
    }

    fn prefill_prefix<C: ModelClient<CpuRuntime>>(
        &self,
        _client: &C,
        tokens: &[u32],
        rows: &[usize],
    ) -> Result<PrefixState<CpuRuntime>> {
        if tokens.len() != self.positions {
            return Err(refused(format!(
                "the fixture holds {} rows, the prefix has {}",
                self.positions,
                tokens.len()
            )));
        }
        let w = self.width;
        let mut data = Vec::with_capacity(rows.len() * w);
        for &r in rows {
            let row = self
                .hidden
                .get(r * w..(r + 1) * w)
                .ok_or_else(|| refused(format!("no hidden row {r}")))?;
            data.extend_from_slice(row);
        }
        let hidden = if rows.is_empty() {
            None
        } else {
            Some(Tensor::<CpuRuntime>::from_slice(
                &data,
                &[rows.len(), w],
                &self.device,
            )?)
        };
        Ok(PrefixState {
            kv: LayeredKvCache::new_positional(0, 1, 1, 1, 1, 1, DType::F32, &self.device)?,
            gdn: LayeredGdnState::zeros(0, &self.gdn, 1, DType::F32, &self.device)?,
            len: tokens.len(),
            rope_end: tokens.len(),
            rows: rows.to_vec(),
            hidden,
        })
    }

    fn lane_caches(
        &self,
        _lanes: usize,
        _width: usize,
        _device: &CpuDevice,
    ) -> Result<(LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>)> {
        Err(refused("a record plan runs no lanes".into()))
    }

    fn lane_bytes(&self, _width: usize) -> usize {
        0
    }

    fn forward_rows<C: ModelClient<CpuRuntime>>(
        &self,
        _client: &C,
        _ids: &Tensor<CpuRuntime>,
        _prefix: &PrefixState<CpuRuntime>,
        _gdn: &mut LayeredGdnState<CpuRuntime>,
        _kv: &mut LayeredKvCache<CpuRuntime>,
        _gather: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        Err(refused("a record plan runs no rows".into()))
    }

    fn candidate_logits<C: ModelClient<CpuRuntime>>(
        &self,
        _client: &C,
        _hidden: &Tensor<CpuRuntime>,
        _ids: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        Err(refused(
            "the joint-schema head reads no candidate logits".into(),
        ))
    }

    fn output_rows<C: ModelClient<CpuRuntime>>(
        &self,
        _client: &C,
        ids: &[u32],
    ) -> Result<Tensor<CpuRuntime>> {
        let w = self.width;
        let mut data = Vec::with_capacity(ids.len() * w);
        for id in ids {
            let r =
                self.row_ids.iter().position(|x| x == id).ok_or_else(|| {
                    refused(format!("the fixture has no lm_head row for id {id}"))
                })?;
            data.extend_from_slice(&self.rows[r * w..(r + 1) * w]);
        }
        Ok(Tensor::<CpuRuntime>::from_slice(
            &data,
            &[ids.len(), w],
            &self.device,
        )?)
    }
}

/// The F32 head through blazr's readout: raw logits, probabilities and
/// wire answers against the reference's F32 head, and the wire answers
/// against the release's BF16 head.
#[test]
fn head_only_answers_match_reference() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let head = JointSchemaHead::<CpuRuntime>::from_dir(&client, &dir, HeadDtype::F32)
        .unwrap_or_else(|e| panic!("the head loads from {}: {e}", dir.display()));
    let readout = ReadoutHead::JointSchema(Box::new(head));
    let r = renderer(&dir);
    let defaults = DecisionConfig::default();
    let rule = PrefixRule {
        min_tokens: defaults.shared_min_tokens,
        at_state: defaults.prefix_cache_mb > 0,
    };
    for fx in &fixtures {
        let what = fx.label();
        let rendered = render_fixture(&r, fx);
        let prepared = prepare(&rendered, TemperaturePolicy::Fixed(1.0), rule)
            .unwrap_or_else(|e| panic!("{what}: prepare: {e}"));
        assert!(prepared.has_prefix(), "{what}: a record runs alone");
        let backbone = FixtureBackbone::new(fx, &device);
        let result = readout
            .run(
                &client,
                &backbone,
                &prepared.request,
                None,
                &RunnerConfig::default(),
            )
            .unwrap_or_else(|e| panic!("{what}: run: {e}"));
        assert_eq!(result.raw_logits.len(), fx.questions.len(), "{what}");
        for (fq, got) in fx.questions.iter().zip(&result.raw_logits) {
            assert_eq!(got.len(), fq.logits_f32.len(), "{what} {}", fq.id);
            for (j, (g, w)) in got.iter().zip(&fq.logits_f32).enumerate() {
                let d = (f64::from(*g) - w).abs();
                assert!(
                    d <= F32_LOGIT,
                    "{what} {} option {j}: logit {g} vs {w}",
                    fq.id
                );
            }
        }
        let answers =
            clef_answers(&rendered, &result).unwrap_or_else(|e| panic!("{what}: format: {e}"));
        compare::answers(&answers, &fx.answers_f32, F32_PROB, &what);
        let release = format!("{what} against the BF16 release");
        compare::answers(&answers, &fx.release_answers(), RELEASE_PROB, &release);
        println!("ok: {what} answers as the reference head");
    }
}

//! `blazr decision calibrate`: render each labelled request with the
//! model's renderer, run it raw on the model's decision worker, fit one
//! temperature per answer type with boostr, report the fits, and write
//! `blazr_calibration.json`.
//!
//! A label file of precomputed records only fits without loading the
//! model's weights.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use boostr::inference::decision::TemperatureBounds;

use super::command::{CalibrateArgs, DecisionCommand};
use crate::cli::build_gpu_index;
use crate::config::{resolve, resolve_decision, DecisionConfig, EnvVars, FileConfig};
use crate::engine::decision::calibration::{
    collect, fit_types, outcome_name, parse_label_lines, records_only, Collected, TypeReport,
};
use crate::engine::Scheduler;
use crate::loader::decision::blazr_calibration::{
    vendor_calibration, CALIBRATION_SCHEMA, CALIBRATION_WRITER,
};
use crate::loader::decision::sidecar::VENDOR_SIDECARS;
use crate::loader::decision::{
    detect_decision, read_calibration_file, write_calibration_file, Calibration, CalibrationFile,
    DecisionSource, TypeFit, BLAZR_CALIBRATION,
};
use crate::runtime::{open_device, ServerRuntime};

/// Run a `blazr decision` command.
///
/// # Errors
///
/// The command's error.
pub async fn decision(command: DecisionCommand) -> Result<()> {
    match command {
        DecisionCommand::Calibrate(args) => calibrate(args).await,
    }
}

/// The file a set of fits writes.
fn calibration_file(
    source: &DecisionSource,
    model: &str,
    reports: &[TypeReport],
    fitted_at: String,
) -> CalibrationFile {
    let mut file = CalibrationFile {
        schema: CALIBRATION_SCHEMA.to_string(),
        family: source.family.name().to_string(),
        fitted_at,
        model: model.to_string(),
        temperatures: Default::default(),
        fits: Default::default(),
        source: CALIBRATION_WRITER.to_string(),
    };
    for report in reports {
        let Some(fit) = &report.fit else {
            continue;
        };
        let kind = report.kind.name().to_string();
        file.temperatures.insert(kind.clone(), fit.t);
        file.fits.insert(
            kind,
            TypeFit {
                n: report.n,
                nll_before: fit.nll_before,
                nll_after: fit.nll_after,
                t: fit.t,
                outcome: outcome_name(fit.outcome).to_string(),
            },
        );
    }
    file
}

/// One report line per answer type.
fn report_lines(
    reports: &[TypeReport],
    min_rows: usize,
    vendor: Option<&Calibration>,
) -> Vec<String> {
    let mut lines = vec![format!(
        "{:<7} {:>6} {:>10} {:>10} {:>9}  {:<5}  outcome",
        "type", "n", "nll(T=1)", "nll(T)", "T", "bound"
    )];
    for report in reports {
        let kind = report.kind.name();
        let line = match &report.fit {
            Some(fit) => format!(
                "{kind:<7} {:>6} {:>10.5} {:>10.5} {:>9.4}  {:<5}  {}",
                report.n,
                fit.nll_before,
                fit.nll_after,
                fit.t,
                if fit.at_bound { "yes" } else { "no" },
                outcome_name(fit.outcome)
            ),
            None => {
                let stays = match vendor.and_then(|v| Some((v.temperatures.get(kind)?, v))) {
                    Some((t, v)) => {
                        let file = v.sources.get(kind).unwrap_or(&v.source);
                        format!("{file} temperature {t} stays")
                    }
                    None => "no vendor temperature: it stays 1".to_string(),
                };
                format!(
                    "{kind:<7} {:>6} {:>10} {:>10} {:>9}  {:<5}  not fitted: fewer than \
                     --min-rows {}; {stays}",
                    report.n,
                    "-",
                    "-",
                    "-",
                    "-",
                    min_rows.max(1)
                )
            }
        };
        lines.push(line);
    }
    lines
}

/// The decision model `model` resolves to under the user config.
fn decision_source(
    scheduler: &Scheduler<ServerRuntime>,
    model: &str,
    cfg: &DecisionConfig,
) -> Result<DecisionSource> {
    let path = scheduler.model_path(model).ok_or_else(|| {
        anyhow!(
            "model '{model}' not found: it is neither a path nor an entry of {}",
            scheduler.model_dir().display()
        )
    })?;
    detect_decision(&path, model, cfg)
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| {
            anyhow!(
                "model '{model}' is not a System One decision model: it ships no \
                 decider_config.json, prompt_contract.json or jebadiah.json, its GGUF header \
                 names no decision model, and no --decision-family applies to it"
            )
        })
}

async fn calibrate(args: CalibrateArgs) -> Result<()> {
    let file = FileConfig::load()?;
    let env = EnvVars::from_process()?;
    let labels = &args.labels;
    let text = std::fs::read_to_string(labels)
        .map_err(|e| anyhow!("{} does not read: {e}", labels.display()))?;
    let lines = parse_label_lines(&text).map_err(|e| anyhow!("{}: {e}", labels.display()))?;
    if lines.is_empty() {
        bail!("{} holds no labelled request or record", labels.display());
    }

    let cfg = resolve_decision(args.decision.into_flags(), &file, Some(args.model.clone()))
        .map_err(anyhow::Error::msg)?;
    let device = open_device(build_gpu_index(&file)?)?;
    let model_dir = resolve::resolve_model_dir(None, &env, &file);
    let scheduler = Scheduler::<ServerRuntime>::new(model_dir, device)
        .with_num_ctx(resolve::resolve_num_ctx(None, &file))
        .with_inference_overlay(file.inference.clone())
        .with_dtype(resolve::resolve_dtype(None, &file))
        .with_decision(cfg.clone());
    let source = decision_source(&scheduler, &args.model, &cfg)?;
    let existing = source.sidecar_dir.join(BLAZR_CALIBRATION);
    if let Err(e) = read_calibration_file(&source.sidecar_dir, source.family) {
        bail!(
            "{e}. The model does not load while {} is invalid: move it away, then run \
             calibrate again",
            existing.display()
        );
    }
    let out = args.out.clone().unwrap_or(existing);
    refuse_vendor_file(&out)?;

    let collected: Collected = match records_only(&lines) {
        Some(collected) => collected,
        None => {
            let service = scheduler
                .decision_service(&args.model)
                .await
                .map_err(|e| anyhow!("model '{}': {e}", args.model))?;
            collect(&service, lines)
                .await
                .map_err(|e| anyhow!("{}: {e}", labels.display()))?
        }
    };
    let reports = fit_types(
        &collected.samples,
        args.min_rows,
        TemperatureBounds::default(),
    )
    .map_err(anyhow::Error::msg)?;
    let vendor = vendor_calibration(&source).map_err(anyhow::Error::msg)?;

    println!(
        "model '{}' ({}): {} requests, {} records, {} unlabelled questions skipped",
        args.model,
        source.family.name(),
        collected.requests,
        collected.records,
        collected.unlabelled
    );
    for line in report_lines(&reports, args.min_rows, vendor.as_ref()) {
        println!("{line}");
    }
    if reports.iter().all(|r| r.fit.is_none()) {
        bail!(
            "no answer type has --min-rows {} labelled answers, so nothing is fitted and \
             nothing is written. Label more answers, or lower --min-rows",
            args.min_rows.max(1)
        );
    }

    let fitted_at = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let out_file = calibration_file(&source, &args.model, &reports, fitted_at);
    if args.dry_run {
        let json = serde_json::to_string_pretty(&out_file)?;
        println!("{json}");
        println!("--dry-run: {} not written", out.display());
        return Ok(());
    }
    write_calibration_file(&out, &out_file).map_err(anyhow::Error::msg)?;
    println!("wrote {}", out.display());
    note_where_serving_reads(&out, &source);
    Ok(())
}

/// Refuse to write over a vendor sidecar.
fn refuse_vendor_file(out: &Path) -> Result<()> {
    let name = out.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    if VENDOR_SIDECARS.contains(&name) {
        bail!(
            "--out {} names a vendor sidecar, which calibrate never writes. Write \
             {BLAZR_CALIBRATION} instead",
            out.display()
        );
    }
    Ok(())
}

/// Tell the user when `out` is not the file `blazr serve` reads.
fn note_where_serving_reads(out: &Path, source: &DecisionSource) {
    let served = source.sidecar_dir.join(BLAZR_CALIBRATION);
    if out != served {
        println!(
            "blazr serve reads {}: copy the file there to serve with it",
            served.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::decision::calibration::AnswerType;
    use crate::model::decision_template::Family;
    use boostr::inference::decision::{FitOutcome, TemperatureFit};
    use std::collections::BTreeMap;

    fn source(dir: &Path) -> DecisionSource {
        DecisionSource {
            family: Family::Decider,
            detected_by: "test".into(),
            model_dir: Some(dir.to_path_buf()),
            gguf: None,
            gguf_facts: None,
            sidecar_dir: dir.to_path_buf(),
            config_file: None,
        }
    }

    fn reports() -> Vec<TypeReport> {
        let fit = TemperatureFit {
            t: 1.4,
            nll_before: 0.6,
            nll_after: 0.5,
            iterations: 4,
            at_bound: false,
            outcome: FitOutcome::Converged,
        };
        vec![
            TypeReport {
                kind: AnswerType::Choice,
                n: 70,
                fit: Some(fit),
            },
            TypeReport {
                kind: AnswerType::Noul,
                n: 3,
                fit: None,
            },
            TypeReport {
                kind: AnswerType::Score,
                n: 0,
                fit: None,
            },
        ]
    }

    #[test]
    fn only_fitted_types_are_written_and_the_file_round_trips() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = calibration_file(&source(dir.path()), "d", &reports(), "2026-10-13".into());
        assert_eq!(
            file.temperatures,
            BTreeMap::from([("choice".to_string(), 1.4)])
        );
        assert_eq!(file.fits["choice"].outcome, "converged");
        assert_eq!(file.family, "decider");
        let path = dir.path().join(BLAZR_CALIBRATION);
        write_calibration_file(&path, &file).expect("writes");
        let back = read_calibration_file(dir.path(), Family::Decider)
            .expect("valid")
            .expect("present");
        assert_eq!(back, file);
    }

    #[test]
    fn vendor_sidecars_are_never_an_output() {
        for name in VENDOR_SIDECARS {
            assert!(
                refuse_vendor_file(&Path::new("/m").join(name)).is_err(),
                "{name}"
            );
        }
        assert!(refuse_vendor_file(Path::new("/m/blazr_calibration.json")).is_ok());
    }

    #[test]
    fn the_report_names_the_vendor_temperature_that_stays() {
        let vendor = Calibration::from_file(
            "decider_config.json",
            BTreeMap::from([("noul".to_string(), 1.56)]),
        );
        let lines = report_lines(&reports(), 50, Some(&vendor));
        assert_eq!(lines.len(), 4);
        assert!(
            lines[1].starts_with("choice") && lines[1].contains("1.4000"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("decider_config.json temperature 1.56 stays"),
            "{}",
            lines[2]
        );
        assert!(lines[3].contains("it stays 1"), "{}", lines[3]);
    }
}

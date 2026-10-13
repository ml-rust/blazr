//! Calibration samples per answer type, from precomputed records and from
//! the raw option logits of run requests.

use boostr::inference::decision::{CalibrationSample, IsolatedSample, QuestionKind, Target};

use super::labels::{AnswerType, Record, RecordLogits};
use crate::model::decision_template::DecisionQuestion;

/// The labelled answers of each type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeSamples {
    /// Choice answers.
    pub choice: Vec<CalibrationSample>,
    /// Noul answers.
    pub noul: Vec<CalibrationSample>,
    /// Score answers read over one row of every level.
    pub score: Vec<CalibrationSample>,
    /// Score answers read one yes/no row per level.
    pub score_isolated: Vec<IsolatedSample>,
}

impl TypeSamples {
    /// The labelled answers of `kind`.
    pub fn count(&self, kind: AnswerType) -> usize {
        match kind {
            AnswerType::Choice => self.choice.len(),
            AnswerType::Noul => self.noul.len(),
            AnswerType::Score => self.score.len() + self.score_isolated.len(),
        }
    }

    fn listed_mut(&mut self, kind: AnswerType) -> &mut Vec<CalibrationSample> {
        match kind {
            AnswerType::Choice => &mut self.choice,
            AnswerType::Noul => &mut self.noul,
            AnswerType::Score => &mut self.score,
        }
    }

    /// Add a precomputed record.
    pub fn push_record(&mut self, record: Record) {
        let target = Target::Hard(record.gold);
        match record.logits {
            RecordLogits::Listed(logits) => self
                .listed_mut(record.kind)
                .push(CalibrationSample { logits, target }),
            RecordLogits::Isolated(levels) => {
                self.score_isolated.push(IsolatedSample { levels, target })
            }
        }
    }

    /// Add `question`'s answer with gold option or level `gold`, read from
    /// `raw_logits`, the run's raw logits of every scoring row.
    ///
    /// # Errors
    ///
    /// When the question's scoring rows are missing from `raw_logits`, or
    /// a row does not hold the logits its kind needs.
    pub fn push_answer(
        &mut self,
        question: &DecisionQuestion,
        raw_logits: &[Vec<f32>],
        gold: usize,
    ) -> Result<(), String> {
        let rows = raw_logits
            .get(question.scoring_rows.clone())
            .ok_or_else(|| {
                format!(
                    "question '{}': the run returned {} scoring rows, the question reads rows \
                     {:?}",
                    question.id,
                    raw_logits.len(),
                    question.scoring_rows
                )
            })?;
        let target = Target::Hard(gold);
        let kind = match &question.meta.kind {
            QuestionKind::Score { isolated: true, .. } => {
                let levels = rows
                    .iter()
                    .map(|row| match row[..] {
                        [no, yes] => Ok([f64::from(no), f64::from(yes)]),
                        _ => Err(format!(
                            "question '{}': a level row holds {} logits, not [no, yes]",
                            question.id,
                            row.len()
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.score_isolated.push(IsolatedSample { levels, target });
                return Ok(());
            }
            QuestionKind::Choice => AnswerType::Choice,
            QuestionKind::Noul { .. } => AnswerType::Noul,
            QuestionKind::Score { .. } => AnswerType::Score,
            _ => {
                return Err(format!(
                    "question '{}' has a kind calibration does not read",
                    question.id
                ))
            }
        };
        let [row] = rows else {
            return Err(format!(
                "question '{}' reads {} scoring rows, one expected",
                question.id,
                rows.len()
            ));
        };
        self.listed_mut(kind).push(CalibrationSample {
            logits: row.iter().map(|&v| f64::from(v)).collect(),
            target,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::QuestionMeta;

    fn question(
        kind: QuestionKind,
        options: usize,
        rows: std::ops::Range<usize>,
    ) -> DecisionQuestion {
        DecisionQuestion {
            id: "q".into(),
            meta: QuestionMeta { kind, options },
            names: (0..options).map(|i| i.to_string()).collect(),
            legend: Vec::new(),
            scoring_rows: rows,
        }
    }

    #[test]
    fn answers_land_in_their_type() {
        let raw = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![0.5, -0.5],
            vec![4.0, 1.0],
            vec![1.0, 4.0],
            vec![2.0, 2.5, 3.0, 0.0],
        ];
        let mut s = TypeSamples::default();
        s.push_answer(&question(QuestionKind::Choice, 3, 0..1), &raw, 2)
            .expect("choice");
        s.push_answer(&question(QuestionKind::Noul { yes: 1 }, 2, 1..2), &raw, 1)
            .expect("noul");
        let isolated = QuestionKind::Score {
            values: None,
            isolated: true,
        };
        s.push_answer(&question(isolated, 2, 2..4), &raw, 0)
            .expect("isolated");
        let listed = QuestionKind::Score {
            values: None,
            isolated: false,
        };
        s.push_answer(&question(listed, 4, 4..5), &raw, 3)
            .expect("listed");
        assert_eq!(s.choice[0].logits, vec![1.0, 2.0, 3.0]);
        assert_eq!(s.choice[0].target, Target::Hard(2));
        assert_eq!(s.noul[0].target, Target::Hard(1));
        assert_eq!(s.score_isolated[0].levels, vec![[4.0, 1.0], [1.0, 4.0]]);
        assert_eq!(s.score[0].logits.len(), 4);
        assert_eq!(
            (
                s.count(AnswerType::Choice),
                s.count(AnswerType::Noul),
                s.count(AnswerType::Score)
            ),
            (1, 1, 2)
        );
        assert!(s
            .push_answer(&question(QuestionKind::Choice, 3, 5..6), &raw, 0)
            .is_err());
    }

    #[test]
    fn records_land_in_their_type() {
        let mut s = TypeSamples::default();
        s.push_record(Record {
            line: 1,
            kind: AnswerType::Score,
            logits: RecordLogits::Isolated(vec![[0.0, 1.0], [1.0, 0.0]]),
            gold: 1,
        });
        s.push_record(Record {
            line: 2,
            kind: AnswerType::Score,
            logits: RecordLogits::Listed(vec![0.0, 1.0, 2.0]),
            gold: 2,
        });
        s.push_record(Record {
            line: 3,
            kind: AnswerType::Noul,
            logits: RecordLogits::Listed(vec![0.0, 1.0]),
            gold: 0,
        });
        assert_eq!(
            (s.score.len(), s.score_isolated.len(), s.noul.len()),
            (1, 1, 1)
        );
        assert_eq!(s.count(AnswerType::Score), 2);
    }
}

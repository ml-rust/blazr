//! `blazr decision calibrate` on the engine: the label file, gold answers
//! as option indices, the labelled answers of each type, the raw runs that
//! produce them, and one temperature fit per answer type.

pub mod collect;
pub mod fit;
pub mod gold;
pub mod labels;
pub mod samples;

pub use collect::{collect, records_only, Collected};
pub use fit::{fit_types, outcome_name, TypeReport, DEFAULT_MIN_ROWS};
pub use gold::gold_index;
pub use labels::{parse_label_lines, AnswerType, LabelLine, LabelledRequest, Record, RecordLogits};
pub use samples::TypeSamples;

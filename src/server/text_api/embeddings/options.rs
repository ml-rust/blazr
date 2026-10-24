//! Validation of embeddings request options and vector post-processing.

use crate::engine::pooling::l2_normalize;

/// Output encoding of each vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EncodingFormat {
    Float,
    Base64,
}

impl EncodingFormat {
    /// Parse `encoding_format`. An absent value means `float`.
    pub(super) fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("float") => Ok(Self::Float),
            Some("base64") => Ok(Self::Base64),
            Some(other) => Err(format!(
                "Invalid encoding_format '{other}'. Must be one of: float, base64"
            )),
        }
    }
}

/// How token states combine into one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pooling {
    Mean,
    Cls,
    Last,
    None,
}

impl Pooling {
    pub(super) fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last" => Ok(Self::Last),
            "none" => Ok(Self::None),
            other => Err(format!(
                "Invalid pooling strategy '{other}'. Must be one of: mean, cls, last, none"
            )),
        }
    }
}

const MATRYOSHKA_NOTE: &str =
    "Truncation keeps the first values only. Quality holds only for Matryoshka-trained models.";

/// Check `dimensions` against the pooling mode before any model work.
pub(in crate::server) fn check_dimensions(
    dimensions: Option<usize>,
    pooling: Pooling,
) -> Result<(), String> {
    match dimensions {
        None => Ok(()),
        Some(0) => Err(format!("dimensions must be at least 1. {MATRYOSHKA_NOTE}")),
        Some(_) if pooling == Pooling::None => Err(
            "dimensions cannot combine with pooling 'none', which returns one vector per token"
                .to_string(),
        ),
        Some(_) => Ok(()),
    }
}

/// Check every token id against the vocabulary size.
pub(in crate::server::text_api) fn check_token_ids(
    ids: &[u32],
    vocab_size: usize,
) -> Result<(), String> {
    match ids.iter().find(|&&id| id as usize >= vocab_size) {
        Some(id) => Err(format!(
            "Token id {id} is outside the vocabulary of size {vocab_size}"
        )),
        None => Ok(()),
    }
}

/// Truncate to `dimensions` when set, then L2-normalize when `normalize` is set.
pub(super) fn finish_vector(
    vector: &mut Vec<f32>,
    dimensions: Option<usize>,
    normalize: bool,
) -> Result<(), String> {
    if let Some(dims) = dimensions {
        if dims > vector.len() {
            return Err(format!(
                "dimensions {dims} exceeds the model dimension {}. {MATRYOSHKA_NOTE}",
                vector.len()
            ));
        }
        vector.truncate(dims);
    }
    if normalize {
        l2_normalize(vector);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_format_accepts_only_float_and_base64() {
        assert_eq!(EncodingFormat::parse(None), Ok(EncodingFormat::Float));
        assert_eq!(
            EncodingFormat::parse(Some("float")),
            Ok(EncodingFormat::Float)
        );
        assert_eq!(
            EncodingFormat::parse(Some("base64")),
            Ok(EncodingFormat::Base64)
        );
        let err = EncodingFormat::parse(Some("hex")).unwrap_err();
        assert!(err.contains("'hex'") && err.contains("float, base64"));
        assert!(EncodingFormat::parse(Some("")).is_err());
        assert!(EncodingFormat::parse(Some("Float")).is_err());
    }

    #[test]
    fn pooling_parse() {
        assert_eq!(Pooling::parse("last"), Ok(Pooling::Last));
        assert!(Pooling::parse("max").is_err());
    }

    #[test]
    fn dimensions_zero_and_pooling_none_are_errors() {
        assert!(check_dimensions(None, Pooling::None).is_ok());
        assert!(check_dimensions(Some(4), Pooling::Mean).is_ok());
        assert!(check_dimensions(Some(0), Pooling::Mean)
            .unwrap_err()
            .contains("Matryoshka"));
        assert!(check_dimensions(Some(4), Pooling::None)
            .unwrap_err()
            .contains("pooling 'none'"));
    }

    #[test]
    fn token_ids_must_be_below_vocab() {
        assert!(check_token_ids(&[0, 9], 10).is_ok());
        assert!(check_token_ids(&[0, 10], 10).unwrap_err().contains("10"));
    }

    #[test]
    fn truncate_then_renormalize() {
        let mut v = vec![3.0, 4.0, 100.0];
        finish_vector(&mut v, Some(2), true).unwrap();
        assert_eq!(v.len(), 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn truncate_without_normalize_keeps_values() {
        let mut v = vec![3.0, 4.0, 5.0];
        finish_vector(&mut v, Some(2), false).unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
    }

    #[test]
    fn dimensions_above_model_dimension_is_error() {
        let mut v = vec![1.0, 2.0];
        let err = finish_vector(&mut v, Some(3), true).unwrap_err();
        assert!(err.contains("exceeds the model dimension 2"));
    }

    #[test]
    fn full_dimension_is_allowed() {
        let mut v = vec![3.0, 4.0];
        finish_vector(&mut v, Some(2), true).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6);
    }
}

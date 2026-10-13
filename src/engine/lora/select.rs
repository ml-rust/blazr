//! [`select_model`]: the model a request runs on, from its `model` and
//! `lora_adapter` fields.

use super::error::LoraError;
use super::spec::variant_id;

/// The model name a request for `model` with `requested` as its
/// `lora_adapter` resolves to. `default` is the live
/// `generation.lora_adapter`. `adapter_of` gives the adapter name of a
/// registered variant id.
///
/// - `model` is a registered variant: that variant. A `requested` adapter
///   must be the same adapter. `default` does not apply.
/// - Else `requested`, else `default`, names an adapter: the variant
///   `<model>:<adapter>`, which must be registered.
/// - Else `model` unchanged.
///
/// An empty adapter name counts as none.
///
/// # Errors
///
/// [`LoraError::Invalid`] when `model` and `requested` name different
/// adapters. [`LoraError::AdapterNotFound`] when the adapter is not
/// registered for `model`. A request never falls back to the base model.
pub fn select_model(
    model: &str,
    requested: Option<&str>,
    default: Option<&str>,
    adapter_of: impl Fn(&str) -> Option<String>,
) -> Result<String, LoraError> {
    let requested = requested.filter(|a| !a.is_empty());
    if let Some(named) = adapter_of(model) {
        return match requested {
            Some(adapter) if adapter != named => Err(LoraError::Invalid(format!(
                "model {model:?} is the variant of LoRA adapter {named:?}, but lora_adapter is \
                 {adapter:?}. Name the adapter once, or name the same adapter in both"
            ))),
            _ => Ok(model.to_string()),
        };
    }
    let Some(adapter) = requested.or(default.filter(|a| !a.is_empty())) else {
        return Ok(model.to_string());
    };
    let id = variant_id(model, adapter);
    if adapter_of(&id).is_some() {
        Ok(id)
    } else {
        Err(not_registered(model, adapter))
    }
}

/// The error for adapter `adapter`, which is not registered for `model`.
pub fn not_registered(model: &str, adapter: &str) -> LoraError {
    LoraError::AdapterNotFound(format!(
        "LoRA adapter {adapter:?} is not registered for model {model:?}. Register it with \
         --lora or POST /v1/lora; GET /v1/lora lists the registered adapters"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m:kev` and `m:jev` are registered.
    fn registered(id: &str) -> Option<String> {
        match id {
            "m:kev" => Some("kev".into()),
            "m:jev" => Some("jev".into()),
            _ => None,
        }
    }

    fn select(
        model: &str,
        requested: Option<&str>,
        default: Option<&str>,
    ) -> Result<String, LoraError> {
        select_model(model, requested, default, registered)
    }

    #[test]
    fn model_or_lora_adapter_picks_the_variant() {
        assert_eq!(select("m", None, None).unwrap(), "m");
        assert_eq!(select("m:kev", None, None).unwrap(), "m:kev");
        assert_eq!(select("m", Some("kev"), None).unwrap(), "m:kev");
        assert_eq!(select("m:kev", Some("kev"), None).unwrap(), "m:kev");
        assert_eq!(select("m", Some(""), None).unwrap(), "m");
    }

    #[test]
    fn model_and_lora_adapter_must_agree() {
        let err = select("m:kev", Some("jev"), None).unwrap_err();
        assert!(matches!(err, LoraError::Invalid(_)), "{err:?}");
        assert!(err.message().contains("\"kev\""), "{err}");
        assert!(err.message().contains("\"jev\""), "{err}");
    }

    #[test]
    fn an_unknown_adapter_never_falls_back_to_the_base() {
        let err = select("m", Some("nope"), None).unwrap_err();
        assert!(matches!(err, LoraError::AdapterNotFound(_)), "{err:?}");
        let err = select("other", Some("kev"), None).unwrap_err();
        assert!(matches!(err, LoraError::AdapterNotFound(_)), "{err:?}");
    }

    #[test]
    fn the_default_applies_only_when_the_request_names_no_adapter() {
        assert_eq!(select("m", None, Some("jev")).unwrap(), "m:jev");
        assert_eq!(select("m", Some("kev"), Some("jev")).unwrap(), "m:kev");
        assert_eq!(select("m:kev", None, Some("jev")).unwrap(), "m:kev");
        assert_eq!(select("m", None, Some("")).unwrap(), "m");
        let err = select("other", None, Some("kev")).unwrap_err();
        assert!(matches!(err, LoraError::AdapterNotFound(_)), "{err:?}");
    }
}

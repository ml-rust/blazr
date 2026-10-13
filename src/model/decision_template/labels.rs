//! Label token tables: the option labels each family shows, and the one
//! token each label reads as at the answer slot, found by asking the
//! tokenizer.

use super::encode::DecisionEncoder;
use super::family::DecisionError;

/// The widest option set AINode labels (`A` .. `IU`).
pub const AINODE_MAX_OPTIONS: usize = super::ainode::MAX_OPTIONS;

/// AINode's `option_label`: zero-based index to `A`..`Z`, `AA`, `AB`, ...,
/// bijective base 26 like spreadsheet columns.
pub fn option_label(index: usize) -> String {
    let mut n = index + 1;
    let mut out = Vec::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        n = (n - 1) / 26;
        out.push(b'A' + rem as u8);
    }
    out.reverse();
    String::from_utf8_lossy(&out).into_owned()
}

/// A label and its token id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelToken {
    /// The label as the prompt shows it.
    pub text: String,
    /// The one token it reads as at the answer slot.
    pub id: u32,
}

/// Jebadiah's two label alphabets, probed at the answer boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AinodeLabels {
    /// AINode's labels in order, up to the first that is not one ordinary
    /// token right after the generation prompt.
    pub ainode: Vec<LabelToken>,
    /// `A`..`Z` (those that are single tokens), then every two-letter
    /// upper-case string that is one ordinary token there, in alphabetical
    /// order: the alphabet for a question wider than [`Self::ainode`].
    pub extended: Vec<LabelToken>,
}

impl AinodeLabels {
    /// Probe the labels. `base` is the ids of a rendered prompt;
    /// `with_label(label)` gives the ids of the same prompt with `label`
    /// appended.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when fewer than 26 AINode labels are single
    /// tokens, or the probe does not encode.
    pub fn probe(
        encoder: &DecisionEncoder,
        base: &[u32],
        with_label: impl Fn(&str) -> Result<Vec<u32>, DecisionError>,
    ) -> Result<Self, DecisionError> {
        let single = |label: &str| -> Result<Option<u32>, DecisionError> {
            let ids = with_label(label)?;
            Ok(match ids.strip_prefix(base) {
                Some([id]) if !encoder.is_added_token(*id) => Some(*id),
                _ => None,
            })
        };
        let mut ainode = Vec::new();
        for i in 0..AINODE_MAX_OPTIONS {
            let text = option_label(i);
            match single(&text)? {
                Some(id) => ainode.push(LabelToken { text, id }),
                None => break,
            }
        }
        if ainode.len() < 26 {
            return Err(DecisionError::Model(format!(
                "only {} labels are single tokens in this tokenizer",
                ainode.len()
            )));
        }
        let mut extended: Vec<LabelToken> = ainode
            .iter()
            .filter(|l| l.text.len() == 1)
            .cloned()
            .collect();
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                let text = String::from_utf8_lossy(&[a, b]).into_owned();
                if let Some(id) = single(&text)? {
                    extended.push(LabelToken { text, id });
                }
            }
        }
        Ok(Self { ainode, extended })
    }
}

/// The decider's widest option set.
pub const DECIDER_MAX_OPTIONS: usize = super::systemone::MAX_CHOICE;

/// The decider's `label_table`: `A`..`Z` then the two-letter upper-case
/// strings, each kept when it encodes alone to one token, the first 255.
/// The first ten are `A`..`J`, the narrow labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeciderLabels {
    /// The 255 labels.
    pub labels: Vec<LabelToken>,
    /// The ids of `"\n("`, the opening of a wide option line.
    pub open_ids: Vec<u32>,
}

impl DeciderLabels {
    /// Build the table.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when fewer than 255 labels are single
    /// tokens, two share a token, or `A`..`J` are not the first ten.
    pub fn probe(encoder: &DecisionEncoder) -> Result<Self, DecisionError> {
        let names = (b'A'..=b'Z')
            .map(|a| vec![a])
            .chain((b'A'..=b'Z').flat_map(|a| (b'A'..=b'Z').map(move |b| vec![a, b])));
        let mut labels = Vec::with_capacity(DECIDER_MAX_OPTIONS);
        for name in names {
            let text = String::from_utf8_lossy(&name).into_owned();
            if let [id] = encoder.template(&text)?.as_slice() {
                labels.push(LabelToken { text, id: *id });
            }
            if labels.len() == DECIDER_MAX_OPTIONS {
                break;
            }
        }
        let mut ids: Vec<u32> = labels.iter().map(|l| l.id).collect();
        ids.sort_unstable();
        ids.dedup();
        if labels.len() != DECIDER_MAX_OPTIONS || ids.len() != DECIDER_MAX_OPTIONS {
            return Err(DecisionError::Model(format!(
                "the decider needs {DECIDER_MAX_OPTIONS} distinct single-token labels; this \
                 tokenizer has {} ({} distinct)",
                labels.len(),
                ids.len()
            )));
        }
        let narrow = labels.iter().take(10).map(|l| l.text.as_str());
        if !narrow.eq(super::decider::LETTERS) {
            return Err(DecisionError::Model(
                "the letters A..J are not all single tokens in this tokenizer".to_string(),
            ));
        }
        let open_ids = encoder.template("\n(")?;
        Ok(Self { labels, open_ids })
    }

    /// The label ids of the first `n` options.
    pub fn ids(&self, n: usize) -> Vec<u32> {
        self.labels.iter().take(n).map(|l| l.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AINode's own examples: `A`, `Z`, `AA`, `AZ`, `BA`, `IU` (index 254).
    #[test]
    fn option_labels_are_bijective_base_26() {
        assert_eq!(option_label(0), "A");
        assert_eq!(option_label(25), "Z");
        assert_eq!(option_label(26), "AA");
        assert_eq!(option_label(51), "AZ");
        assert_eq!(option_label(52), "BA");
        assert_eq!(option_label(67), "BP");
        assert_eq!(option_label(254), "IU");
        assert_eq!(option_label(701), "ZZ");
        assert_eq!(option_label(702), "AAA");
    }
}

//! Which loaded models an unload, a full scheduler or an expired
//! keep-alive drops.
//!
//! A variant shares its base's weight storage, so it never outlives its
//! base: dropping a base drops each of its variants first. A variant drops
//! alone. Only base models count against the loaded-model limit, since a
//! variant adds an executor and its adapter factors, not a second copy of
//! the weights.

use std::time::{Duration, Instant};

/// One loaded model, as eviction sees it.
#[derive(Debug, Clone)]
pub struct Resident<'a> {
    /// The loaded name: a model name or a variant id.
    pub name: &'a str,
    /// The base model of a variant. `None` for a base model.
    pub variant_of: Option<&'a str>,
    /// The last request, or a request for one of its variants.
    pub last_accessed: Instant,
    /// Idle time before an automatic unload. `None` keeps it loaded.
    pub keep_alive: Option<Duration>,
}

/// The names to drop to unload `name`, in order: each loaded variant of
/// `name` first, then `name`. Empty when `name` is not loaded.
pub fn unload_order(residents: &[Resident<'_>], name: &str) -> Vec<String> {
    let Some(target) = residents.iter().find(|r| r.name == name) else {
        return Vec::new();
    };
    let mut order: Vec<String> = Vec::new();
    if target.variant_of.is_none() {
        order.extend(
            residents
                .iter()
                .filter(|r| r.variant_of == Some(name))
                .map(|r| r.name.to_string()),
        );
    }
    order.push(name.to_string());
    order
}

/// The number of loaded base models.
pub fn base_count(residents: &[Resident<'_>]) -> usize {
    residents.iter().filter(|r| r.variant_of.is_none()).count()
}

/// The least recently used base model: the one to evict for room.
pub fn lru_base(residents: &[Resident<'_>]) -> Option<String> {
    residents
        .iter()
        .filter(|r| r.variant_of.is_none())
        .min_by_key(|r| r.last_accessed)
        .map(|r| r.name.to_string())
}

/// The names whose keep-alive ran out at `now`, with the variants of each
/// expired base. Every variant comes before its base.
pub fn expired(residents: &[Resident<'_>], now: Instant) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let is_expired = |r: &Resident<'_>| {
        r.keep_alive
            .is_some_and(|ttl| now.saturating_duration_since(r.last_accessed) > ttl)
    };
    for resident in residents.iter().filter(|r| is_expired(r)) {
        for name in unload_order(residents, resident.name) {
            if !order.contains(&name) {
                order.push(name);
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident<'a>(
        name: &'a str,
        variant_of: Option<&'a str>,
        age_s: u64,
        keep_alive_s: Option<u64>,
        now: Instant,
    ) -> Resident<'a> {
        Resident {
            name,
            variant_of,
            last_accessed: now - Duration::from_secs(age_s),
            keep_alive: keep_alive_s.map(Duration::from_secs),
        }
    }

    #[test]
    fn a_base_unloads_after_its_variants_and_a_variant_alone() {
        let now = Instant::now();
        let residents = [
            resident("m", None, 0, None, now),
            resident("m:kev", Some("m"), 0, None, now),
            resident("n", None, 0, None, now),
            resident("m:jev", Some("m"), 0, None, now),
        ];
        assert_eq!(unload_order(&residents, "m"), ["m:kev", "m:jev", "m"]);
        assert_eq!(unload_order(&residents, "m:kev"), ["m:kev"]);
        assert_eq!(unload_order(&residents, "n"), ["n"]);
        assert!(unload_order(&residents, "absent").is_empty());
    }

    #[test]
    fn only_bases_count_and_the_lru_base_goes() {
        let now = Instant::now();
        let residents = [
            resident("m", None, 5, None, now),
            resident("m:kev", Some("m"), 50, None, now),
            resident("n", None, 10, None, now),
        ];
        assert_eq!(base_count(&residents), 2);
        assert_eq!(lru_base(&residents).as_deref(), Some("n"));
        assert_eq!(unload_order(&residents, "n"), ["n"]);
        assert!(lru_base(&[]).is_none());
    }

    #[test]
    fn an_expired_base_takes_its_variants_first() {
        let now = Instant::now();
        let residents = [
            resident("m:kev", Some("m"), 1, None, now),
            resident("m", None, 100, Some(10), now),
            resident("n:jev", Some("n"), 100, Some(10), now),
            resident("n", None, 1, Some(10), now),
        ];
        assert_eq!(expired(&residents, now), ["m:kev", "m", "n:jev"]);
    }

    #[test]
    fn an_expired_variant_listed_first_stays_before_its_base() {
        let now = Instant::now();
        let residents = [
            resident("m:kev", Some("m"), 100, Some(10), now),
            resident("m", None, 100, Some(10), now),
        ];
        assert_eq!(expired(&residents, now), ["m:kev", "m"]);
    }
}

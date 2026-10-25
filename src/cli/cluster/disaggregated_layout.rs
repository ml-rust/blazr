//! The rank layout of a disaggregated cluster, from the worker counts and `--index`.
//!
//! Every node derives the same layout from `--prefill-count` and
//! `--decode-count`: the router on rank 0, prefill index `i` on rank `1 + i`,
//! decode index `i` on rank `1 + prefill_count + i`. Each worker requests its
//! rank, so the ranks never depend on start order.

use anyhow::{anyhow, bail, Result};
use nexar::Rank;

use crate::cli::DisaggRoleArg;
use crate::distributed::disaggregated::{DisaggConfig, DisaggRole};

/// The largest serialized KV cache one request transfers.
pub(super) const MAX_KV_TRANSFER_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// One node's place in the cluster.
#[derive(Debug, Clone)]
pub(super) struct DisaggLayout {
    /// The ranks of every role.
    pub config: DisaggConfig,
    /// This node's role.
    pub role: DisaggRole,
    /// This node's rank.
    pub rank: Rank,
}

impl DisaggLayout {
    /// The layout of a `role` node with `--index index` in a cluster of
    /// `prefill` prefill and `decode` decode workers.
    ///
    /// # Errors
    ///
    /// A count of 0, an `--index` on the router, or an index at or past its
    /// role's count.
    pub(super) fn resolve(
        role: DisaggRoleArg,
        index: Option<u32>,
        prefill: u32,
        decode: u32,
    ) -> Result<Self> {
        let config = DisaggConfig::standard(prefill, decode, MAX_KV_TRANSFER_BYTES)
            .map_err(|e| anyhow!("{e:#}. Pass --prefill-count and --decode-count of 1 or more"))?;
        let (role, name, count) = match role {
            DisaggRoleArg::Router => (DisaggRole::Router, "router", 1),
            DisaggRoleArg::Prefill => (DisaggRole::Prefill, "prefill", prefill),
            DisaggRoleArg::Decode => (DisaggRole::Decode, "decode", decode),
        };
        if role == DisaggRole::Router && index.is_some() {
            bail!("the router always holds rank 0 and takes no --index. Remove --index");
        }
        let index = index.unwrap_or(0);
        if index >= count {
            bail!(
                "--index {index} is out of range: --{name}-count is {count}, so a {name} worker \
                 takes an index from 0 to {}. Pass a lower --index, or raise --{name}-count to \
                 the same value on every node",
                count - 1
            );
        }
        let rank = config.rank_of(role, index)?;
        Ok(Self { config, role, rank })
    }

    /// Ranks in the cluster.
    pub(super) fn world_size(&self) -> u32 {
        self.config.world_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank(role: DisaggRoleArg, index: Option<u32>) -> Result<Rank> {
        DisaggLayout::resolve(role, index, 2, 3).map(|l| l.rank)
    }

    #[test]
    fn each_role_and_index_maps_to_its_rank() -> Result<()> {
        assert_eq!(rank(DisaggRoleArg::Router, None)?, 0);
        assert_eq!(rank(DisaggRoleArg::Prefill, None)?, 1);
        assert_eq!(rank(DisaggRoleArg::Prefill, Some(1))?, 2);
        assert_eq!(rank(DisaggRoleArg::Decode, Some(0))?, 3);
        assert_eq!(rank(DisaggRoleArg::Decode, Some(2))?, 5);
        let layout = DisaggLayout::resolve(DisaggRoleArg::Decode, Some(1), 2, 3)?;
        assert_eq!(layout.world_size(), 6);
        assert_eq!(layout.config.prefill_workers, vec![1, 2]);
        assert_eq!(layout.config.decode_workers, vec![3, 4, 5]);
        Ok(())
    }

    #[test]
    fn an_index_past_its_count_names_the_flags() {
        let err = rank(DisaggRoleArg::Prefill, Some(2))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--index 2"), "{err}");
        assert!(err.contains("--prefill-count is 2"), "{err}");
        assert!(err.contains("from 0 to 1"), "{err}");
        let err = rank(DisaggRoleArg::Decode, Some(3))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--decode-count is 3"), "{err}");
    }

    #[test]
    fn the_router_takes_no_index_and_a_zero_count_is_refused() {
        let err = rank(DisaggRoleArg::Router, Some(0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no --index"), "{err}");
        let err = DisaggLayout::resolve(DisaggRoleArg::Router, None, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--prefill-count"), "{err}");
    }
}

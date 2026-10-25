//! nexar cluster formation for `swarm` and `disagg`: one node's
//! `ClusterConfig`, formation errors with their fix, and the end of a run.

use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use nexar::{ClusterConfig, NexarError, Rank};

use crate::cli::MeshArgs;

/// How long a node waits for its other client handles to drop before its
/// nexar shutdown.
pub(super) const RELEASE_WITHIN: Duration = Duration::from_secs(10);

/// One node's mesh settings, parsed from its flags.
#[derive(Debug, Clone)]
pub(super) struct MeshSettings {
    /// The mesh listener: QUIC on UDP and the bulk sidecar on TCP.
    pub listen: SocketAddr,
    /// The address peers dial, when it differs from `listen`.
    pub advertise: Option<SocketAddr>,
    /// The cluster token. `None`: nexar reads `NEXAR_CLUSTER_TOKEN`.
    pub token: Option<String>,
    /// Limit for the whole formation.
    pub join_timeout: Duration,
}

/// `text` as a socket address, or an error that names `flag`.
pub(super) fn parse_addr(flag: &str, text: &str) -> Result<SocketAddr> {
    text.parse().map_err(|e| {
        anyhow!("{flag} '{text}' is not an IP:port address ({e}). Pass one such as 10.0.0.1:9191")
    })
}

impl MeshSettings {
    /// The settings of `--listen-addr listen`, the shared mesh flags and `token`.
    ///
    /// # Errors
    ///
    /// An address that does not parse, or a join timeout of 0.
    pub(super) fn parse(listen: &str, mesh: &MeshArgs, token: Option<String>) -> Result<Self> {
        if mesh.join_timeout == 0 {
            bail!("--join-timeout is 0: pass the seconds to wait for the cluster, such as 600");
        }
        Ok(Self {
            listen: parse_addr("--listen-addr", listen)?,
            advertise: mesh
                .advertise_addr
                .as_deref()
                .map(|a| parse_addr("--advertise-addr", a))
                .transpose()?,
            token: token.filter(|t| !t.is_empty()),
            join_timeout: Duration::from_secs(mesh.join_timeout),
        })
    }

    /// The nexar config of a node in a cluster of `world_size` ranks that asks for `rank`.
    pub(super) fn cluster_config(&self, world_size: u32, rank: Option<Rank>) -> ClusterConfig {
        let mut cfg = ClusterConfig::new(world_size)
            .with_bind_addr(self.listen)
            .with_join_timeout(self.join_timeout);
        if let Some(addr) = self.advertise {
            cfg = cfg.with_advertise_addr(addr);
        }
        if let Some(token) = &self.token {
            cfg = cfg.with_cluster_token(token.clone());
        }
        if let Some(rank) = rank {
            cfg = cfg.with_requested_rank(rank);
        }
        cfg
    }

    /// Checks that the peers of the seed process can dial it.
    ///
    /// The seed process joins its own seed over loopback, so the seed sees no
    /// address for it. With an unspecified seed and mesh IP, it needs one.
    ///
    /// # Errors
    ///
    /// A cluster of more than one rank, an unspecified IP in both `seed` and
    /// `--listen-addr`, and no `--advertise-addr`. The message names `seed`
    /// as `seed_flag`.
    pub(super) fn check_seed_reachable(
        &self,
        seed: SocketAddr,
        seed_flag: &str,
        world_size: u32,
    ) -> Result<()> {
        let advertised = self.advertise.is_some_and(|a| !a.ip().is_unspecified());
        if world_size > 1
            && !advertised
            && seed.ip().is_unspecified()
            && self.listen.ip().is_unspecified()
        {
            bail!(
                "peers cannot dial this node: {seed_flag} {seed} and --listen-addr {} name no \
                 host IP. Pass --advertise-addr <this-host-ip>:{}, or a --listen-addr with \
                 this host's IP",
                self.listen,
                self.listen.port()
            );
        }
        Ok(())
    }
}

/// `e`, the error of `what`, as a CLI error with the fix.
///
/// `token_flag` names how this command sets the cluster token.
pub(super) fn formation_error(what: &str, e: NexarError, token_flag: &str) -> anyhow::Error {
    let fix = match &e {
        NexarError::TokenRequired { addr } => format!(
            "{addr} is not a loopback address, and no cluster token is set. Pass {token_flag} \
             with the same secret on every node"
        ),
        NexarError::ClusterTokenMismatch => format!(
            "this node's cluster token differs from the seed's. Pass the same {token_flag} on \
             every node"
        ),
        NexarError::RankConflict { rank, detail } => format!(
            "rank {rank} is not free ({detail}). Give each worker of a role its own --index, \
             and pass the same counts on every node"
        ),
        NexarError::ClusterFormationTimeout { joined, expected } => format!(
            "{joined} of {expected} nodes joined within --join-timeout. Start the missing \
             nodes, check that UDP reaches the seed port, or raise --join-timeout"
        ),
        NexarError::JoinTimeout { .. } => format!(
            "{e}. Check the seed address and that UDP reaches its port, or raise --join-timeout"
        ),
        _ => e.to_string(),
    };
    anyhow!("{what} failed: {fix}")
}

/// Runs `work` until it ends, or until SIGINT or SIGTERM.
///
/// A signal ends the run with `Ok`, so the caller still shuts down its
/// nexar client before the process exits.
pub(super) async fn until_signal(role: &str, work: impl Future<Output = Result<()>>) -> Result<()> {
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| anyhow!("installing the SIGTERM handler failed: {e}"))?;
    #[cfg(unix)]
    let terminate = sigterm.recv();
    #[cfg(not(unix))]
    let terminate = std::future::pending::<Option<()>>();
    tokio::select! {
        result = work => result,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("{role} received SIGINT, shutting down");
            Ok(())
        }
        _ = terminate => {
            tracing::info!("{role} received SIGTERM, shutting down");
            Ok(())
        }
    }
}

/// The run's result after the nexar shutdown of `role`.
///
/// A failed shutdown is a warning: the run's own result wins.
pub(super) fn after_shutdown(role: &str, run: Result<()>, shutdown: Result<()>) -> Result<()> {
    if let Err(e) = shutdown {
        tracing::warn!("{role} nexar shutdown: {e:#}");
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(advertise: Option<&str>, join_timeout: u64) -> MeshArgs {
        MeshArgs {
            advertise_addr: advertise.map(str::to_string),
            join_timeout,
        }
    }

    #[test]
    fn parse_names_the_flag_of_a_bad_address() {
        let err = MeshSettings::parse("nope", &mesh(None, 60), None).unwrap_err();
        assert!(err.to_string().contains("--listen-addr 'nope'"), "{err}");
        let err = MeshSettings::parse("0.0.0.0:1", &mesh(Some("x"), 60), None).unwrap_err();
        assert!(err.to_string().contains("--advertise-addr 'x'"), "{err}");
        let err = MeshSettings::parse("0.0.0.0:1", &mesh(None, 0), None).unwrap_err();
        assert!(err.to_string().contains("--join-timeout"), "{err}");
    }

    #[test]
    fn the_config_carries_every_setting() {
        let s = MeshSettings::parse(
            "0.0.0.0:9191",
            &mesh(Some("192.0.2.7:9191"), 30),
            Some("s3cret".into()),
        )
        .unwrap();
        let cfg = s.cluster_config(4, Some(2));
        assert_eq!(cfg.world_size, 4);
        assert_eq!(cfg.bind_addr, parse_addr("x", "0.0.0.0:9191").unwrap());
        assert_eq!(cfg.advertise_addr, parse_addr("x", "192.0.2.7:9191").ok());
        assert_eq!(cfg.requested_rank, Some(2));
        assert_eq!(cfg.cluster_token.as_deref(), Some("s3cret"));
        assert_eq!(cfg.join_timeout, Duration::from_secs(30));
        let empty = MeshSettings::parse("0.0.0.0:1", &mesh(None, 1), Some(String::new())).unwrap();
        assert_eq!(empty.token, None);
    }

    #[test]
    fn a_seed_without_a_dialable_address_is_refused() {
        let wild = MeshSettings::parse("0.0.0.0:9191", &mesh(None, 60), None).unwrap();
        let any_seed = parse_addr("x", "0.0.0.0:9190").unwrap();
        let err = wild
            .check_seed_reachable(any_seed, "--router-addr", 3)
            .unwrap_err();
        assert!(err.to_string().contains("--advertise-addr"), "{err}");
        wild.check_seed_reachable(any_seed, "--router-addr", 1)
            .unwrap();
        let host_seed = parse_addr("x", "10.0.0.1:9190").unwrap();
        wild.check_seed_reachable(host_seed, "--router-addr", 3)
            .unwrap();
        let advertised =
            MeshSettings::parse("0.0.0.0:9191", &mesh(Some("10.0.0.1:9191"), 60), None).unwrap();
        advertised
            .check_seed_reachable(any_seed, "--router-addr", 3)
            .unwrap();
    }

    #[test]
    fn a_missing_token_error_names_the_flag() {
        let addr = parse_addr("x", "10.0.0.1:9190").unwrap();
        let err = formation_error(
            "router bootstrap",
            NexarError::TokenRequired { addr },
            "--cluster-token <secret> (or NEXAR_CLUSTER_TOKEN)",
        );
        let text = err.to_string();
        assert!(text.starts_with("router bootstrap failed"), "{text}");
        assert!(
            text.contains("10.0.0.1:9190") && text.contains("--cluster-token"),
            "{text}"
        );
    }
}

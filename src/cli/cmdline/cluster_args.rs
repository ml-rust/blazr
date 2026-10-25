//! Arguments of the multi-node commands `swarm` and `disagg`.

use clap::{Args, ValueEnum};

/// nexar mesh flags that `swarm` and `disagg` share.
#[derive(Debug, Clone, Args)]
pub struct MeshArgs {
    /// Address peers dial to reach this node's mesh listener, for NAT and
    /// containers. Its port carries UDP and TCP [default: --listen-addr, with
    /// the IP the seed sees when that IP is 0.0.0.0]
    #[arg(long, value_name = "ADDR")]
    pub advertise_addr: Option<String>,

    /// Seconds every node waits for the whole cluster to form. Workers load
    /// their model before they join, so cover the slowest model load
    #[arg(long, value_name = "SECS", default_value_t = 600)]
    pub join_timeout: u64,
}

/// A `swarm` role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SwarmRoleArg {
    Leader,
    Worker,
}

/// `blazr swarm`: a pipeline leader and its workers, one process each.
#[derive(Debug, Clone, Args)]
pub struct SwarmArgs {
    /// Role of this node
    #[arg(long, value_enum, default_value = "worker")]
    pub role: SwarmRoleArg,

    /// Shared secret of the swarm: the nexar cluster token. Use the same
    /// value on every node
    #[arg(long, env = "BLAZR_SWARM_TOKEN", hide_env_values = true)]
    pub token: String,

    /// Number of worker nodes. Every node passes the same value
    #[arg(long, default_value_t = 1)]
    pub workers: u32,

    /// Model to serve (leader only) [default: default_model]
    #[arg(long, short)]
    pub model: Option<String>,

    /// Leader's seed address to join (worker only), for example "192.168.1.10:9090"
    #[arg(long, required_if_eq("role", "worker"))]
    pub leader: Option<String>,

    /// UDP port of the leader's seed, on the --listen-addr IP (leader only)
    #[arg(long, default_value_t = 9090)]
    pub swarm_port: u16,

    /// This node's mesh listener: QUIC on UDP and the bulk sidecar on TCP, one port
    #[arg(long, value_name = "ADDR", default_value = "0.0.0.0:9091")]
    pub listen_addr: String,

    #[command(flatten)]
    pub mesh: MeshArgs,

    /// Port for HTTP API (leader only) [default: server.port, else 8080]
    #[arg(long)]
    pub port: Option<u16>,

    /// Host for HTTP API (leader only) [default: server.host, else 0.0.0.0]
    #[arg(long)]
    pub host: Option<String>,

    /// API key for the leader's HTTP API [default: $BLAZR_API_KEY]
    #[arg(long)]
    pub api_key: Option<String>,
}

/// A `disagg` role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DisaggRoleArg {
    Router,
    Prefill,
    Decode,
}

/// `blazr disagg`: a router, prefill workers and decode workers, one process each.
#[derive(Debug, Clone, Args)]
pub struct DisaggArgs {
    /// Role of this node
    #[arg(long, value_enum, default_value = "router")]
    pub role: DisaggRoleArg,

    /// Model. Router: required, the tokenizer and chat template source
    /// (no weights load). Prefill and decode: the model to load
    /// [default: default_model]
    #[arg(long, short, required_if_eq("role", "router"))]
    pub model: Option<String>,

    /// The router's seed address, on every node. The router binds it, and
    /// each worker dials it. Its port carries UDP
    #[arg(long, value_name = "ADDR")]
    pub router_addr: String,

    /// This node's mesh listener: QUIC on UDP and the bulk sidecar on TCP, one port
    #[arg(long, value_name = "ADDR", default_value = "0.0.0.0:9191")]
    pub listen_addr: String,

    /// Number of prefill workers. Every node passes the same value
    #[arg(long, default_value_t = 1)]
    pub prefill_count: u32,

    /// Number of decode workers. Every node passes the same value
    #[arg(long, default_value_t = 1)]
    pub decode_count: u32,

    /// This worker's index within its role, from 0. Prefill index i is rank
    /// 1 + i. Decode index i is rank 1 + prefill-count + i. The router takes
    /// no index [default: 0]
    #[arg(long)]
    pub index: Option<u32>,

    /// nexar cluster token, the same on every node. Required when an address
    /// is not loopback
    #[arg(long, env = "NEXAR_CLUSTER_TOKEN", hide_env_values = true)]
    pub cluster_token: Option<String>,

    #[command(flatten)]
    pub mesh: MeshArgs,

    /// HTTP API port (router role only) [default: server.port, else 8080]
    #[arg(long)]
    pub port: Option<u16>,

    /// HTTP API host (router role only) [default: server.host, else 0.0.0.0]
    #[arg(long)]
    pub host: Option<String>,
}

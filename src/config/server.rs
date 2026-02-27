//! Server configuration settings

use serde::{Deserialize, Serialize};

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Maximum concurrent requests
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,

    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,

    /// Allowed CORS origins (empty = all)
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Enable request logging
    #[serde(default = "default_true")]
    pub request_logging: bool,

    /// Maximum request body size in bytes
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
}

fn default_port() -> u16 {
    8080
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_max_concurrent() -> usize {
    16
}

fn default_timeout() -> u64 {
    300 // 5 minutes
}

fn default_true() -> bool {
    true
}

fn default_max_body_size() -> usize {
    10 * 1024 * 1024 // 10 MB
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
            max_concurrent_requests: default_max_concurrent(),
            request_timeout_secs: default_timeout(),
            cors_enabled: true,
            cors_origins: Vec::new(),
            request_logging: true,
            max_body_size: default_max_body_size(),
        }
    }
}

impl ServerConfig {
    /// Get the socket address string
    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

//! The process's rustls crypto provider.
//!
//! ring is the process-wide provider. hf-hub's reqwest and Xet client also
//! link aws-lc-rs, but read the installed default first. The HTTPS listener
//! names ring itself.

use std::sync::Arc;

use rustls::crypto::CryptoProvider;

/// Installs ring as the process-wide rustls provider.
///
/// reqwest and hf-hub read this default before any provider they link.
/// A provider the embedding program installed first stays in place.
pub fn install_crypto_provider() {
    // `Err` means a provider is already installed, which is the goal.
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// The installed provider, or ring when none is installed.
pub fn crypto_provider() -> Arc<CryptoProvider> {
    CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::ring::default_provider()))
}

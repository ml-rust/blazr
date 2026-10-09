//! The process's rustls crypto provider.
//!
//! blazr links one provider, ring: hf-hub's ureq client hard-wires it, and
//! reqwest and the HTTPS listener use the same one.

use std::sync::Arc;

use rustls::crypto::CryptoProvider;

/// Installs ring as the process-wide rustls provider.
///
/// reqwest is built without a bundled provider and reads this default.
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

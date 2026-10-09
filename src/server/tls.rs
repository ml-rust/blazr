//! HTTPS listener config from PEM files.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use rustls::pki_types::pem::{self, PemObject};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// Builds the rustls server config from a PEM certificate chain and key.
///
/// The config names its crypto provider, so it builds even when more
/// than one provider is linked into the process.
pub fn build_tls_config(cert_path: &Path, key_path: &Path) -> Result<rustls::ServerConfig> {
    let cert_pem = std::fs::read(cert_path)
        .with_context(|| format!("Failed to read TLS cert {}", cert_path.display()))?;
    let key_pem = std::fs::read(key_path)
        .with_context(|| format!("Failed to read TLS key {}", key_path.display()))?;

    let certs = CertificateDer::pem_slice_iter(&cert_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            anyhow!(
                "Failed to parse TLS certificates in {}: {e}",
                cert_path.display()
            )
        })?;
    if certs.is_empty() {
        return Err(anyhow!("No certificate found in {}", cert_path.display()));
    }
    let key = match PrivateKeyDer::from_pem_slice(&key_pem) {
        Ok(key) => key,
        Err(pem::Error::NoItemsFound) => {
            return Err(anyhow!("No private key found in {}", key_path.display()))
        }
        Err(e) => {
            return Err(anyhow!(
                "Failed to parse TLS private key in {}: {e}",
                key_path.display()
            ))
        }
    };

    rustls::ServerConfig::builder_with_provider(crate::tls::crypto_provider())
        .with_safe_default_protocol_versions()
        .map_err(|e| anyhow!("Failed to select TLS protocol versions: {e}"))?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| anyhow!("Failed to build TLS config: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tls")
            .join(name)
    }

    #[test]
    fn builds_server_config_from_pem_files() {
        build_tls_config(&fixture("cert.pem"), &fixture("key.pem"))
            .expect("the fixture cert and key build a server config");
    }

    #[test]
    fn rejects_a_file_without_a_key() {
        let err = build_tls_config(&fixture("cert.pem"), &fixture("cert.pem"))
            .expect_err("a certificate is not a private key");
        assert!(err.to_string().contains("No private key found"), "{err}");
    }

    #[test]
    fn rejects_a_file_without_a_certificate() {
        let err = build_tls_config(&fixture("key.pem"), &fixture("key.pem"))
            .expect_err("a private key is not a certificate");
        assert!(err.to_string().contains("No certificate found"), "{err}");
    }
}

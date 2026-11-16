//! `server.cors_origins` entries: each one an origin a browser sends.

/// An error naming the first entry of `origins` that is no origin.
///
/// A browser sends an origin as `scheme://host[:port]`, so an entry with a
/// path, such as a trailing `/`, would match no request.
pub fn check_cors_origins(origins: &[String]) -> Result<(), String> {
    for origin in origins {
        if let Err(why) = check_origin(origin) {
            return Err(format!(
                "server.cors_origins entry {origin:?} is not an origin: {why}. Write it as \
                 scheme://host[:port], such as https://app.example.com"
            ));
        }
    }
    Ok(())
}

/// Why `origin` is no origin.
fn check_origin(origin: &str) -> Result<(), &'static str> {
    let Some((scheme, host)) = origin.split_once("://") else {
        return Err("it has no scheme");
    };
    if scheme.is_empty() || host.is_empty() {
        return Err("it has no scheme or no host");
    }
    if host.contains('/') {
        return Err("it has a path");
    }
    if !origin.bytes().all(|b| b.is_ascii_graphic()) {
        return Err("it holds a space, a control character or a non-ASCII character");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn origins_pass() {
        let ok = list(&["https://app.example.com", "http://10.0.0.5:3000"]);
        assert_eq!(check_cors_origins(&ok), Ok(()));
        assert_eq!(check_cors_origins(&[]), Ok(()));
    }

    /// A bad entry used to be dropped without a log line.
    #[test]
    fn a_bad_entry_is_an_error_that_names_it() {
        for bad in [
            "app.example.com",
            "https://app.example.com/",
            "https://",
            "https://app example.com",
            "https://app.example.com\n",
        ] {
            let err = check_cors_origins(&list(&["https://ok.example.com", bad])).expect_err(bad);
            assert!(
                err.starts_with(&format!("server.cors_origins entry {bad:?}")),
                "{err}"
            );
        }
    }
}

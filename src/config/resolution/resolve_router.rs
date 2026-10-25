//! The disaggregated router's model: `--model`, never `default_model`.

use super::resolve_cluster::ClusterSettings;

impl ClusterSettings {
    /// The disaggregated router's model, from `--model` only: it must name
    /// the workers' model, never a local `default_model`.
    ///
    /// # Errors
    ///
    /// An error when `--model` is not given.
    pub fn router_model(&self) -> Result<String, String> {
        match (&self.model, self.model_flag) {
            (Some(model), true) => Ok(model.clone()),
            _ => Err(
                "the disagg router role needs --model: the workers' model, whose \
                 tokenizer and chat template the router loads"
                    .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::config::resolution::{resolve_cluster, ClusterFlags, EnvVars};
    use crate::config::FileConfig;

    #[test]
    fn the_router_takes_its_model_from_the_flag_only() {
        let env = EnvVars::default();
        let file = FileConfig::parse("default_model: f\n", Path::new("config.yaml"))
            .unwrap()
            .config;
        let s = resolve_cluster(ClusterFlags::default(), &env, &file).unwrap();
        let err = s.router_model().unwrap_err();
        assert!(err.contains("needs --model"), "{err}");
        let flags = ClusterFlags {
            model: Some("llama".into()),
            ..Default::default()
        };
        let s = resolve_cluster(flags, &env, &file).unwrap();
        assert_eq!(s.router_model().unwrap(), "llama");
    }
}

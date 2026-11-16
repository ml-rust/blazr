//! Hub clients, model repo handles, and cache-first file fetches.

use std::path::{Path, PathBuf};

use hf_hub::repository::RepoTypeModel;
use hf_hub::{split_id, HFClient, HFClientSync, HFError, HFRepository, HFRepositorySync, HFResult};

use super::progress::BarProgress;
use crate::offline::{guard_when, is_offline, Need};

/// `Err` when offline mode blocks a hub request for the file `name` of
/// `owner/repo`, or for its file list when `name` is `None`.
fn hub_guard(owner: &str, repo: &str, name: Option<&str>) -> HFResult<()> {
    hub_guard_when(is_offline(), owner, repo, name)
}

/// [`hub_guard`] with offline mode given as `offline`.
fn hub_guard_when(offline: bool, owner: &str, repo: &str, name: Option<&str>) -> HFResult<()> {
    let target = match name {
        Some(name) => format!("{owner}/{repo}: {name}"),
        None => format!("the file list of {owner}/{repo}"),
    };
    guard_when(offline, &target, Need::HubModel).map_err(|e| HFError::Other(e.to_string()))
}

/// A blocking hub client. `cache` overrides the hub cache the environment
/// names.
pub fn sync_client(cache: Option<&Path>) -> HFResult<HFClientSync> {
    HFClientSync::from_inner(async_client(cache)?)
}

/// An async hub client. `cache` overrides the hub cache the environment
/// names.
pub fn async_client(cache: Option<&Path>) -> HFResult<HFClient> {
    let mut builder = HFClient::builder();
    if let Some(cache) = cache {
        builder = builder.cache_dir(cache);
    }
    builder.build()
}

/// The model repo `id`, as `owner/name`, on a blocking client.
pub fn sync_model(client: &HFClientSync, id: &str) -> HFRepositorySync<RepoTypeModel> {
    let (owner, name) = split_id(id);
    client.model(owner, name)
}

/// The model repo `id`, as `owner/name`, on an async client.
pub fn async_model(client: &HFClient, id: &str) -> HFRepository<RepoTypeModel> {
    let (owner, name) = split_id(id);
    client.model(owner, name)
}

/// Every file of `repo`, as paths in the repo. Offline mode blocks it.
pub fn sync_files(repo: &HFRepositorySync<RepoTypeModel>) -> HFResult<Vec<String>> {
    hub_guard(repo.owner(), repo.name(), None)?;
    let info = repo.info().send()?;
    Ok(file_names(info.siblings))
}

/// Every file of `repo`, as paths in the repo. Offline mode blocks it.
pub async fn async_files(repo: &HFRepository<RepoTypeModel>) -> HFResult<Vec<String>> {
    hub_guard(repo.owner(), repo.name(), None)?;
    let info = repo.info().send().await?;
    Ok(file_names(info.siblings))
}

/// The repo paths of `siblings`.
fn file_names(siblings: Option<Vec<hf_hub::repository::RepoSibling>>) -> Vec<String> {
    siblings
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.rfilename)
        .collect()
}

/// The repo file `name`: the cached copy when the cache holds one, else
/// downloaded into the cache. `progress` draws a bar on standard error.
/// Offline mode blocks the download, never the cached copy.
pub fn sync_get(
    repo: &HFRepositorySync<RepoTypeModel>,
    name: &str,
    progress: bool,
) -> HFResult<PathBuf> {
    if let Ok(path) = repo
        .download_file()
        .filename(name)
        .local_files_only(true)
        .send()
    {
        return Ok(path);
    }
    hub_guard(repo.owner(), repo.name(), Some(name))?;
    repo.download_file()
        .filename(name)
        .maybe_progress(progress.then(|| BarProgress::progress(name)))
        .send()
}

/// The repo file `name`: the cached copy when the cache holds one, else
/// downloaded into the cache. `progress` draws a bar on standard error.
/// Offline mode blocks the download, never the cached copy.
pub async fn async_get(
    repo: &HFRepository<RepoTypeModel>,
    name: &str,
    progress: bool,
) -> HFResult<PathBuf> {
    if let Ok(path) = repo
        .download_file()
        .filename(name)
        .local_files_only(true)
        .send()
        .await
    {
        return Ok(path);
    }
    hub_guard(repo.owner(), repo.name(), Some(name))?;
    repo.download_file()
        .filename(name)
        .maybe_progress(progress.then(|| BarProgress::progress(name)))
        .send()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_blocks_a_hub_fetch_and_names_the_file() {
        assert!(hub_guard_when(false, "org", "m", Some("config.json")).is_ok());
        let err = hub_guard_when(true, "org", "m", Some("config.json"))
            .expect_err("offline blocks")
            .to_string();
        assert!(
            err.starts_with("org/m: config.json needs the network"),
            "{err}"
        );
        let err = hub_guard_when(true, "org", "m", None)
            .expect_err("offline blocks")
            .to_string();
        assert!(
            err.starts_with("the file list of org/m needs the network"),
            "{err}"
        );
    }
}

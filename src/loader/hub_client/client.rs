//! Hub clients, model repo handles, and cache-first file fetches.

use std::path::{Path, PathBuf};

use hf_hub::repository::RepoTypeModel;
use hf_hub::{split_id, HFClient, HFClientSync, HFRepository, HFRepositorySync, HFResult};

use super::progress::BarProgress;

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

/// The repo file `name`: the cached copy when the cache holds one, else
/// downloaded into the cache. `progress` draws a bar on standard error.
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
    repo.download_file()
        .filename(name)
        .maybe_progress(progress.then(|| BarProgress::progress(name)))
        .send()
}

/// The repo file `name`: the cached copy when the cache holds one, else
/// downloaded into the cache. `progress` draws a bar on standard error.
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
    repo.download_file()
        .filename(name)
        .maybe_progress(progress.then(|| BarProgress::progress(name)))
        .send()
        .await
}

//! The Hugging Face hub client: model repos by id, and files fetched
//! cache-first into a hub cache in hub layout.

mod client;
mod progress;

pub use client::{async_client, async_get, async_model, sync_client, sync_get, sync_model};

//! Multi-replica and multi-GPU serving: data and tensor parallelism.

pub mod cache_router;
pub mod data_parallel;
pub mod tensor_parallel;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub mod tp_group;

pub use cache_router::{CacheAwareRouter, RouteReason};
pub use data_parallel::{route_through, DataParallelGroup, ReplicaLease};

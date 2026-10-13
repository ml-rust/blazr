//! Inference slot management
//!
//! Slots represent named inference sessions that can be allocated, queried,
//! and freed. Each slot tracks its model, creation time, and last access.
//!
//! Future: slots will hold saved KV cache state for multi-turn conversations,
//! avoiding re-prefill on each turn.

use std::collections::HashMap;
use std::time::Instant;

use tokio::sync::RwLock;

/// Information about an active inference slot
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// Slot identifier
    pub id: String,
    /// Model name this slot is bound to
    pub model: String,
    /// When the slot was created
    pub created: Instant,
    /// When the slot was last used
    pub last_accessed: Instant,
    /// Number of tokens processed in this slot's lifetime
    pub total_tokens: usize,
}

/// Manages inference slot allocation and lifecycle
pub struct SlotManager {
    /// Active slots by ID
    slots: RwLock<HashMap<String, SlotInfo>>,
    /// Maximum number of concurrent slots (0 = unlimited)
    max_slots: usize,
}

impl SlotManager {
    /// Create a new slot manager
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: RwLock::new(HashMap::new()),
            max_slots,
        }
    }

    /// Allocate a new slot. Returns the slot ID, or an error if at capacity.
    pub async fn allocate(&self, model: &str) -> Result<String, String> {
        let mut slots = self.slots.write().await;
        if self.max_slots > 0 && slots.len() >= self.max_slots {
            return Err(format!("Maximum slot count ({}) reached", self.max_slots));
        }
        let id = uuid::Uuid::new_v4().to_string();
        let now = Instant::now();
        slots.insert(
            id.clone(),
            SlotInfo {
                id: id.clone(),
                model: model.to_string(),
                created: now,
                last_accessed: now,
                total_tokens: 0,
            },
        );
        Ok(id)
    }

    /// Touch a slot (update last_accessed and add tokens)
    pub async fn touch(&self, slot_id: &str, tokens: usize) -> bool {
        let mut slots = self.slots.write().await;
        if let Some(slot) = slots.get_mut(slot_id) {
            slot.last_accessed = Instant::now();
            slot.total_tokens += tokens;
            true
        } else {
            false
        }
    }

    /// Free a slot by ID. Returns true if the slot existed.
    pub async fn free(&self, slot_id: &str) -> bool {
        let mut slots = self.slots.write().await;
        slots.remove(slot_id).is_some()
    }

    /// List all active slots
    pub async fn list(&self) -> Vec<SlotInfo> {
        let slots = self.slots.read().await;
        slots.values().cloned().collect()
    }

    /// Get a specific slot's info
    pub async fn get(&self, slot_id: &str) -> Option<SlotInfo> {
        let slots = self.slots.read().await;
        slots.get(slot_id).cloned()
    }

    /// Get the number of active slots
    pub async fn count(&self) -> usize {
        let slots = self.slots.read().await;
        slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_slot_allocate_and_free() {
        let mgr = SlotManager::new(10);
        let id = mgr.allocate("test-model").await.unwrap();
        assert!(!id.is_empty());
        assert_eq!(mgr.count().await, 1);

        let info = mgr.get(&id).await.unwrap();
        assert_eq!(info.model, "test-model");
        assert_eq!(info.total_tokens, 0);

        assert!(mgr.free(&id).await);
        assert_eq!(mgr.count().await, 0);
        assert!(!mgr.free(&id).await); // already freed
    }

    #[tokio::test]
    async fn test_slot_max_capacity() {
        let mgr = SlotManager::new(2);
        mgr.allocate("model-a").await.unwrap();
        mgr.allocate("model-b").await.unwrap();
        assert!(mgr.allocate("model-c").await.is_err());
    }

    #[tokio::test]
    async fn test_slot_touch() {
        let mgr = SlotManager::new(0);
        let id = mgr.allocate("model").await.unwrap();
        assert!(mgr.touch(&id, 100).await);
        let info = mgr.get(&id).await.unwrap();
        assert_eq!(info.total_tokens, 100);

        assert!(mgr.touch(&id, 50).await);
        let info = mgr.get(&id).await.unwrap();
        assert_eq!(info.total_tokens, 150);
    }

    #[tokio::test]
    async fn test_slot_list() {
        let mgr = SlotManager::new(0);
        mgr.allocate("model-a").await.unwrap();
        mgr.allocate("model-b").await.unwrap();
        let list = mgr.list().await;
        assert_eq!(list.len(), 2);
    }
}

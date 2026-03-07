//! Reranking endpoint (`POST /rerank`, `POST /v1/rerank`)
//!
//! Scores query-document pairs using the model's embedding similarity.
//! Returns documents ranked by relevance score (cosine similarity).

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::error_response;
use super::handlers::AppState;
use super::pooling::pool_mean;

/// Rerank request (Cohere/Jina compatible)
#[derive(Deserialize)]
pub struct RerankRequest {
    /// Model name
    pub model: String,
    /// The search query
    pub query: String,
    /// Documents to rerank
    pub documents: Vec<RerankDocument>,
    /// Number of top results to return (default: all)
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Whether to return the document text in results (default: true)
    #[serde(default = "default_true")]
    pub return_documents: bool,
}

fn default_true() -> bool {
    true
}

/// Document input — string or object with text field
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object { text: String },
}

impl RerankDocument {
    fn text(&self) -> &str {
        match self {
            RerankDocument::Text(s) => s,
            RerankDocument::Object { text } => text,
        }
    }
}

/// Rerank response
#[derive(Serialize)]
pub struct RerankResponse {
    pub object: &'static str,
    pub results: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankResultDocument>,
}

#[derive(Serialize)]
pub struct RerankResultDocument {
    pub text: String,
}

#[derive(Serialize)]
pub struct RerankUsage {
    pub total_tokens: usize,
}

/// Reranking endpoint
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RerankRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    if request.documents.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "documents array must not be empty",
            "invalid_request_error",
        );
    }

    // Get query embedding
    let query_tokens = match executor.tokenizer().encode(&request.query) {
        Ok(ids) => ids,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Tokenization failed: {}", e),
                "server_error",
            );
        }
    };

    let query_embedding = match executor.get_embeddings(&query_tokens).await {
        Ok(emb) => emb,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Embedding generation failed: {}", e),
                "server_error",
            );
        }
    };

    let query_len = query_tokens.len();
    let hidden_size = if query_len > 0 {
        query_embedding.len() / query_len
    } else {
        0
    };

    // Mean-pool query embedding
    let query_pooled = pool_mean(&query_embedding, query_len, hidden_size);

    let mut total_tokens = query_len;
    let mut scored: Vec<(usize, f32, &str)> = Vec::with_capacity(request.documents.len());

    for (i, doc) in request.documents.iter().enumerate() {
        let doc_text = doc.text();
        let doc_tokens = match executor.tokenizer().encode(doc_text) {
            Ok(ids) => ids,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Tokenization failed for document {}: {}", i, e),
                    "server_error",
                );
            }
        };
        let doc_len = doc_tokens.len();
        total_tokens += doc_len;

        let doc_embedding = match executor.get_embeddings(&doc_tokens).await {
            Ok(emb) => emb,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Embedding failed for document {}: {}", i, e),
                    "server_error",
                );
            }
        };

        let doc_pooled = pool_mean(&doc_embedding, doc_len, hidden_size);
        let score = cosine_similarity(&query_pooled, &doc_pooled);
        scored.push((i, score, doc_text));
    }

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top_n
    let top_n = request.top_n.unwrap_or(scored.len()).min(scored.len());
    let results: Vec<RerankResult> = scored[..top_n]
        .iter()
        .map(|(idx, score, text)| RerankResult {
            index: *idx,
            relevance_score: *score,
            document: if request.return_documents {
                Some(RerankResultDocument {
                    text: text.to_string(),
                })
            } else {
                None
            },
        })
        .collect();

    let response = RerankResponse {
        object: "list",
        results,
        model: request.model,
        usage: RerankUsage { total_tokens },
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-12 {
        dot / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_document_text() {
        let json = r#""hello world""#;
        let doc: RerankDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.text(), "hello world");
    }

    #[test]
    fn test_rerank_document_object() {
        let json = r#"{"text": "hello world"}"#;
        let doc: RerankDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.text(), "hello world");
    }

    #[test]
    fn test_rerank_request_deserialization() {
        let json = r#"{
            "model": "test",
            "query": "What is AI?",
            "documents": ["AI is cool", {"text": "Machine learning"}],
            "top_n": 1
        }"#;
        let req: RerankRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "What is AI?");
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.top_n, Some(1));
        assert!(req.return_documents);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rerank_response_serialization() {
        let resp = RerankResponse {
            object: "list",
            results: vec![RerankResult {
                index: 0,
                relevance_score: 0.95,
                document: Some(RerankResultDocument {
                    text: "hello".to_string(),
                }),
            }],
            model: "test".to_string(),
            usage: RerankUsage { total_tokens: 10 },
        };
        let json = serde_json::to_value(&resp).unwrap();
        let score = json["results"][0]["relevance_score"].as_f64().unwrap();
        assert!((score - 0.95).abs() < 1e-5);
        assert_eq!(json["results"][0]["document"]["text"], "hello");
    }

    #[test]
    fn test_return_documents_false() {
        let json = r#"{
            "model": "test",
            "query": "q",
            "documents": ["a"],
            "return_documents": false
        }"#;
        let req: RerankRequest = serde_json::from_str(json).unwrap();
        assert!(!req.return_documents);
    }
}

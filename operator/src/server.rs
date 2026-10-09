//! HTTP server — REST API for vector storage and similarity search.
//!
//! POST   /v1/collections                — create collection
//! GET    /v1/collections                — list collections
//! DELETE /v1/collections/:name          — delete collection
//! POST   /v1/collections/:name/upsert  — upsert vectors
//! POST   /v1/collections/:name/query   — similarity search
//! DELETE /v1/collections/:name/vectors  — delete vectors by ID
//! GET    /v1/collections/:name/stats    — collection stats
//! GET    /health                        — operator health
//! GET    /metrics                       — prometheus

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tokio::sync::watch;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, metrics_handler,
};
use tangle_inference_core::{AppState, FlatRequestCostModel, RequestGuard};

use crate::config::OperatorConfig;
use crate::store::{
    CreateCollectionRequest, DeleteVectorsRequest, QueryRequest, UnsupportedQueryFilter,
    UpsertRequest, VectorStoreBackend,
};

/// Minimum billing cost for admin operations (create/delete/list/stats).
/// Essentially free but requires valid SpendAuth.
const MIN_ADMIN_COST: u64 = 1000;

/// Backend state attached to AppState.
pub struct VectorStoreAppBackend {
    pub store: Arc<dyn VectorStoreBackend>,
    pub config: Arc<OperatorConfig>,
    pub upsert_cost_model: Arc<FlatRequestCostModel>,
    pub query_cost_model: Arc<FlatRequestCostModel>,
}

impl VectorStoreAppBackend {
    pub fn new(config: Arc<OperatorConfig>, store: Arc<dyn VectorStoreBackend>) -> Self {
        // Per-request cost: price_per_k / 1000 (single request cost).
        // For billing_gate, we pass the batch cost inline.
        Self {
            upsert_cost_model: Arc::new(FlatRequestCostModel {
                price_per_request: config.vector_store.price_per_k_upserts,
            }),
            query_cost_model: Arc::new(FlatRequestCostModel {
                price_per_request: config.vector_store.price_per_k_queries,
            }),
            store,
            config,
        }
    }
}

pub fn build_router(state: AppState) -> Router {
    let max_body = state.server_config.max_request_body_bytes;
    let timeout = state.server_config.stream_timeout_secs;
    Router::new()
        .route("/v1/collections", post(create_collection))
        .route("/v1/collections", get(list_collections))
        .route("/v1/collections/:name", delete(delete_collection))
        .route("/v1/collections/:name/upsert", post(upsert_vectors))
        .route("/v1/collections/:name/query", post(query_vectors))
        .route("/v1/collections/:name/vectors", delete(delete_vectors))
        .route("/v1/collections/:name/stats", get(collection_stats))
        .route("/v1/tiers", get(list_tiers))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
        .layer(RequestBodyLimitLayer::new(max_body))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(timeout),
        ))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

pub async fn start(
    state: AppState,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "Vector Store HTTP server listening");

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.wait_for(|&v| v).await;
            })
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// ── Validation ────────────────────────────────────────────────────────

#[expect(
    clippy::result_large_err,
    reason = "HTTP handlers return this validation response directly without an extra allocation."
)]
fn validate_collection_name(name: &str) -> Result<(), Response> {
    if name.is_empty()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
        || name.trim().is_empty()
        || name.len() > 256
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid collection name: {name:?}"),
            "validation_error",
            "invalid_collection_name",
        ));
    }
    Ok(())
}

// ── Handlers ───────────────────────────────────────────────────────────

async fn create_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateCollectionRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(resp) = validate_collection_name(&req.name) {
        return resp;
    }

    if req.dimensions == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "dimensions must be > 0".into(),
            "validation_error",
            "invalid_dimensions",
        );
    }

    // Enforce max_collections capacity limit
    let max_cols = backend.config.vector_store.max_collections;
    if max_cols > 0 {
        if let Ok(existing) = backend.store.list_collections().await {
            if existing.len() >= max_cols as usize {
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("collection limit reached ({max_cols})"),
                    "capacity_error",
                    "max_collections_exceeded",
                );
            }
        }
    }

    match backend
        .store
        .create_collection(&req.name, req.dimensions, req.distance_metric)
        .await
    {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "name": req.name,
                "dimensions": req.dimensions,
                "distance_metric": req.distance_metric,
            })),
        )
            .into_response(),
        Err(e) => error_response(
            StatusCode::CONFLICT,
            format!("{e}"),
            "collection_error",
            "create_failed",
        ),
    }
}

async fn list_collections(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match backend.store.list_collections().await {
        Ok(collections) => Json(serde_json::json!({ "collections": collections })).into_response(),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e}"),
            "store_error",
            "list_failed",
        ),
    }
}

async fn delete_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(resp) = validate_collection_name(&name) {
        return resp;
    }

    match backend.store.delete_collection(&name).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(
            StatusCode::NOT_FOUND,
            format!("{e}"),
            "collection_error",
            "delete_failed",
        ),
    }
}

async fn upsert_vectors(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    if req.vectors.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "vectors array must not be empty".into(),
            "validation_error",
            "empty_vectors",
        );
    }

    // Reject 0-dimension vectors and mismatched dimensions within batch
    let expected_dim = req.vectors[0].vector.len();
    if expected_dim == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "vector dimensions must be > 0".into(),
            "validation_error",
            "zero_dimensions",
        );
    }
    for (i, p) in req.vectors.iter().enumerate() {
        if p.vector.len() != expected_dim {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "dimension mismatch in batch: vector[0] has {} dims, vector[{i}] has {}",
                    expected_dim,
                    p.vector.len()
                ),
                "validation_error",
                "dimension_mismatch",
            );
        }
    }

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Bill proportional to batch size: (count / 1000) * price_per_k_upserts
    let batch_count = req.vectors.len() as u64;
    let estimated_cost =
        batch_count.saturating_mul(backend.config.vector_store.price_per_k_upserts) / 1000;
    let estimated_cost = estimated_cost.max(1); // minimum 1 unit

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, estimated_cost).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut guard = RequestGuard::new("upsert");

    match backend.store.upsert(&name, req.vectors).await {
        Ok(result) => {
            guard.set_success();
            (StatusCode::OK, Json(result)).into_response()
        }
        Err(e) => error_response(
            StatusCode::BAD_REQUEST,
            format!("{e}"),
            "upsert_error",
            "upsert_failed",
        ),
    }
}

async fn query_vectors(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(req): Json<QueryRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    if req.vector.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "query vector must not be empty".into(),
            "validation_error",
            "empty_vector",
        );
    }

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Bill per query: price_per_k_queries / 1000 per single query
    let estimated_cost = backend.config.vector_store.price_per_k_queries / 1000;
    let estimated_cost = estimated_cost.max(1);

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, estimated_cost).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let mut guard = RequestGuard::new("query");

    match backend
        .store
        .query(&name, req.vector, req.top_k, req.filter)
        .await
    {
        Ok(results) => {
            guard.set_success();
            Json(serde_json::json!({ "results": results })).into_response()
        }
        Err(e) if e.is::<UnsupportedQueryFilter>() => error_response(
            StatusCode::BAD_REQUEST,
            format!("{e}"),
            "invalid_request_error",
            "unsupported_filter",
        ),
        Err(e) => error_response(
            StatusCode::BAD_REQUEST,
            format!("{e}"),
            "query_error",
            "query_failed",
        ),
    }
}

async fn delete_vectors(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
    Json(req): Json<DeleteVectorsRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match backend.store.delete_vectors(&name, req.ids).await {
        Ok(result) => Json(result).into_response(),
        Err(e) => error_response(
            StatusCode::NOT_FOUND,
            format!("{e}"),
            "delete_error",
            "delete_failed",
        ),
    }
}

async fn collection_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    let (_spend_auth, _preauth) = match billing_gate(&state, &headers, None, MIN_ADMIN_COST).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match backend.store.collection_stats(&name).await {
        Ok(stats) => Json(stats).into_response(),
        Err(e) => error_response(
            StatusCode::NOT_FOUND,
            format!("{e}"),
            "stats_error",
            "stats_failed",
        ),
    }
}

async fn list_tiers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");
    Json(serde_json::json!({
        "tiers": backend.config.vector_store.tiers,
        "overage_pricing": {
            "price_per_k_upserts": backend.config.vector_store.price_per_k_upserts,
            "price_per_k_queries": backend.config.vector_store.price_per_k_queries,
        },
    }))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "vector-store",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{HttpProxyBackend, ProxyProvider};

    async fn test_server(billing_required: bool) -> (String, tokio::task::JoinHandle<()>) {
        // Local-only synthetic identity. No chain or upstream request is needed by these tests.
        let config: OperatorConfig = serde_json::from_value(serde_json::json!({
            "tangle": {
                "rpc_url": "http://127.0.0.1:1", "chain_id": 31337,
                "operator_key": "11".repeat(32),
                "shielded_credits": "0x0000000000000000000000000000000000000001",
                "blueprint_id": 1, "service_id": 1
            },
            "server": {},
            "billing": {
                "payment_rails": {}, "billing_required": billing_required,
                "max_spend_per_request": 1000000, "min_credit_balance": 0,
                "nonce_store_path": null, "direct_replay_store_path": null
            },
            "vector_store": {}
        }))
        .unwrap();
        let config = Arc::new(config);
        let store = Arc::new(HttpProxyBackend::new(
            "http://127.0.0.1:1".into(),
            ProxyProvider::ChromaDB,
            None,
        ));
        let backend = VectorStoreAppBackend::new(config.clone(), store);
        let state =
            AppState::from_config(&config.tangle, &config.server, &config.billing, 4, backend)
                .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, build_router(state)).await.unwrap();
        });
        (url, task)
    }

    #[tokio::test]
    async fn named_collection_routes_reach_the_existing_authorization_gate() {
        let (url, task) = test_server(true).await;
        let client = reqwest::Client::new();
        let cases = [
            (
                reqwest::Method::DELETE,
                "/v1/collections/docs",
                serde_json::json!({}),
            ),
            (
                reqwest::Method::POST,
                "/v1/collections/docs/upsert",
                serde_json::json!({
                    "vectors": [{"id": "one", "vector": [1.0]}]
                }),
            ),
            (
                reqwest::Method::POST,
                "/v1/collections/docs/query",
                serde_json::json!({
                    "vector": [1.0], "filter": {"must": [{"key": "workspace", "value": "private"}]}
                }),
            ),
            (
                reqwest::Method::DELETE,
                "/v1/collections/docs/vectors",
                serde_json::json!({"ids": ["one"]}),
            ),
            (
                reqwest::Method::GET,
                "/v1/collections/docs/stats",
                serde_json::json!({}),
            ),
        ];
        for (method, path, body) in cases {
            let response = client
                .request(method, format!("{url}{path}"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::PAYMENT_REQUIRED,
                "{path}"
            );
        }
        task.abort();
    }

    #[tokio::test]
    async fn unsupported_filter_fields_are_rejected_at_http_intake() {
        let (url, task) = test_server(false).await;
        for filter in [
            serde_json::json!({"should": [{"key": "workspace", "value": "private"}]}),
            serde_json::json!({"must": [{"key": "workspace", "value": "private", "operator": "not_equal"}]}),
            serde_json::json!({"must": [{"key": "workspace", "value": "private"}], "should": []}),
        ] {
            let response = reqwest::Client::new()
                .post(format!("{url}/v1/collections/docs/query"))
                .json(&serde_json::json!({"vector": [1.0], "filter": filter}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
        }
        task.abort();
    }

    #[tokio::test]
    async fn unsupported_proxy_filter_has_an_actionable_http_error() {
        let (url, task) = test_server(false).await;
        let response = reqwest::Client::new()
            .post(format!("{url}/v1/collections/docs/query"))
            .json(&serde_json::json!({
                "vector": [1.0], "filter": {"must": [{"key": "workspace", "value": "private"}]}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "unsupported_filter");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        task.abort();
    }
}

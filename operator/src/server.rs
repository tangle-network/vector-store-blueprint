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
    CreateCollectionRequest, DeleteVectorsRequest, QueryRequest, UpsertRequest,
    VectorStoreBackend,
};

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
        .route("/v1/collections/{name}", delete(delete_collection))
        .route("/v1/collections/{name}/upsert", post(upsert_vectors))
        .route("/v1/collections/{name}/query", post(query_vectors))
        .route("/v1/collections/{name}/vectors", delete(delete_vectors))
        .route("/v1/collections/{name}/stats", get(collection_stats))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
        .layer(RequestBodyLimitLayer::new(max_body))
        .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, std::time::Duration::from_secs(timeout)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

pub async fn start(
    state: AppState,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let bind = format!(
        "{}:{}",
        state.server_config.host, state.server_config.port
    );
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

// ── Handlers ───────────────────────────────────────────────────────────

async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

    if req.name.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "collection name must not be empty".into(),
            "validation_error",
            "empty_name",
        );
    }

    if req.dimensions == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "dimensions must be > 0".into(),
            "validation_error",
            "invalid_dimensions",
        );
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

async fn list_collections(State(state): State<AppState>) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

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
    Path(name): Path<String>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

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

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Bill proportional to batch size: (count / 1000) * price_per_k_upserts
    let batch_count = req.vectors.len() as u64;
    let estimated_cost = batch_count
        .saturating_mul(backend.config.vector_store.price_per_k_upserts)
        / 1000;
    let estimated_cost = estimated_cost.max(1); // minimum 1 unit

    let (_spend_auth, _preauth) =
        match billing_gate(&state, &headers, None, estimated_cost).await {
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

    let (_spend_auth, _preauth) =
        match billing_gate(&state, &headers, None, estimated_cost).await {
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
    Path(name): Path<String>,
    Json(req): Json<DeleteVectorsRequest>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

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
    Path(name): Path<String>,
) -> Response {
    let backend = state
        .backend::<VectorStoreAppBackend>()
        .expect("VectorStoreAppBackend");

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

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "vector-store",
    }))
}

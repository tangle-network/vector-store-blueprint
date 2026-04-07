pub mod config;
pub mod server;
pub mod store;

pub use tangle_inference_core::{
    AppState, BillingClient, CostModel, CostParams, FlatRequestCostModel, RequestGuard,
    SpendAuthPayload,
};

use crate::config::OperatorConfig;
use crate::server::VectorStoreAppBackend;
use crate::store::build_backend;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::{oneshot, watch};

sol! {
    #[allow(missing_docs)]
    struct VectorStoreJobRequest {
        string collection;
        string operation; // "upsert" | "query" | "delete"
        bytes payload;    // JSON-encoded operation payload
    }

    #[allow(missing_docs)]
    struct VectorStoreJobResult {
        bool success;
        bytes response;   // JSON-encoded response
    }
}

pub const VECTOR_STORE_JOB: u8 = 0;

pub fn router() -> Router {
    Router::new().route(VECTOR_STORE_JOB, run_vector_store_job.layer(TangleLayer))
}

#[debug_job]
pub async fn run_vector_store_job(
    TangleArg(request): TangleArg<VectorStoreJobRequest>,
) -> Result<TangleResult<VectorStoreJobResult>, RunnerError> {
    tracing::info!(
        collection = %request.collection,
        operation = %request.operation,
        "Received on-chain vector store request"
    );

    // On-chain jobs redirect to HTTP API — primary access path is REST + x402
    Ok(TangleResult(VectorStoreJobResult {
        success: true,
        response: b"use /v1/collections HTTP endpoint".to_vec().into(),
    }))
}

/// Background service that starts the vector store HTTP server.
pub struct VectorStoreServer {
    config: OperatorConfig,
}

impl VectorStoreServer {
    pub fn new(config: OperatorConfig) -> Self {
        Self { config }
    }
}

impl BackgroundService for VectorStoreServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = Arc::new(self.config.clone());

        tokio::spawn(async move {
            let store = Arc::from(build_backend(&config.vector_store.backend));

            let backend = VectorStoreAppBackend::new(config.clone(), store);

            let state = match AppState::from_config(
                &config.tangle,
                &config.server,
                &config.billing,
                config.server.max_concurrent_requests,
                backend,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            let (_shutdown_tx, shutdown_rx) = watch::channel(false);

            match server::start(state, shutdown_rx).await {
                Ok(handle) => {
                    tracing::info!("Vector Store HTTP server started");
                    let _ = handle.await;
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                }
            }
        });

        Ok(rx)
    }
}

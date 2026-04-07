use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use vector_store::config::OperatorConfig;
use vector_store::VectorStoreServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();
    dotenvy::dotenv().ok();

    tracing::info!("Vector Store Blueprint starting...");

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config: {e}")))?;

    tracing::info!(
        backend = ?config.vector_store.backend,
        price_per_k_upserts = config.vector_store.price_per_k_upserts,
        price_per_k_queries = config.vector_store.price_per_k_queries,
        max_collections = config.vector_store.max_collections,
        "Config loaded"
    );

    let env = BlueprintEnvironment::load()?;
    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service_id".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client);

    let server = VectorStoreServer::new(config);

    tracing::info!("Starting BlueprintRunner...");

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(vector_store::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(server)
        .run()
        .await?;

    Ok(())
}

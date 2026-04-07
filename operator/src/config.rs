use serde::{Deserialize, Serialize};
pub use tangle_inference_core::{BillingConfig, ServerConfig, TangleConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    pub tangle: TangleConfig,
    pub server: ServerConfig,
    pub billing: BillingConfig,
    pub vector_store: VectorStoreConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend")]
pub enum VectorBackend {
    #[serde(rename = "qdrant")]
    Qdrant { url: String },
    #[serde(rename = "chromadb")]
    ChromaDB { url: String },
    #[serde(rename = "inmemory")]
    InMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    #[serde(default = "default_backend")]
    pub backend: VectorBackend,

    /// Price per 1K vector upserts (payment token base units, e.g. 10000 = 0.01 USDC with 6 decimals).
    #[serde(default = "default_price_per_k_upserts")]
    pub price_per_k_upserts: u64,

    /// Price per 1K similarity queries (payment token base units, e.g. 5000 = 0.005 USDC with 6 decimals).
    #[serde(default = "default_price_per_k_queries")]
    pub price_per_k_queries: u64,

    /// Maximum collections per operator instance.
    #[serde(default = "default_max_collections")]
    pub max_collections: u32,

    /// Maximum vectors per collection.
    #[serde(default = "default_max_vectors_per_collection")]
    pub max_vectors_per_collection: u64,
}

fn default_backend() -> VectorBackend {
    VectorBackend::InMemory
}

fn default_price_per_k_upserts() -> u64 {
    10_000 // 0.01 payment token per 1K vectors
}

fn default_price_per_k_queries() -> u64 {
    5_000 // 0.005 payment token per 1K queries
}

fn default_max_collections() -> u32 {
    100
}

fn default_max_vectors_per_collection() -> u64 {
    10_000_000
}

impl OperatorConfig {
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let builder = config::Config::builder();
        let builder = if let Some(p) = path {
            builder.add_source(config::File::with_name(p))
        } else if let Ok(p) = std::env::var("CONFIG_PATH") {
            builder.add_source(config::File::with_name(&p))
        } else {
            builder.add_source(config::File::with_name("config").required(false))
        };
        let config = builder
            .add_source(config::Environment::with_prefix("VSTORE_OP").separator("__"))
            .build()?;
        Ok(config.try_deserialize()?)
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            price_per_k_upserts: default_price_per_k_upserts(),
            price_per_k_queries: default_price_per_k_queries(),
            max_collections: default_max_collections(),
            max_vectors_per_collection: default_max_vectors_per_collection(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vector_store_config() {
        let cfg = VectorStoreConfig::default();
        assert_eq!(cfg.price_per_k_upserts, 10_000);
        assert_eq!(cfg.price_per_k_queries, 5_000);
        assert_eq!(cfg.max_collections, 100);
        assert!(matches!(cfg.backend, VectorBackend::InMemory));
    }
}

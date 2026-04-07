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
    #[serde(rename = "pgvector")]
    PgVector { connection_string: String },
    #[serde(rename = "milvus")]
    Milvus { url: String },
    #[serde(rename = "weaviate")]
    Weaviate { url: String },
    #[serde(rename = "pinecone")]
    Pinecone {
        api_key: String,
        environment: String,
    },
    #[serde(rename = "inmemory")]
    InMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTier {
    pub name: String,
    pub storage_gb: u64,
    pub included_queries_per_month: u64,
    pub included_upserts_per_month: u64,
    /// Subscription rate in payment token base units per month.
    pub subscription_rate: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    #[serde(default = "default_backend")]
    pub backend: VectorBackend,

    /// Available storage tiers (customers choose at service creation).
    #[serde(default = "default_tiers")]
    pub tiers: Vec<StorageTier>,

    /// Overage price per 1K upserts beyond tier quota (payment token base units).
    #[serde(default = "default_price_per_k_upserts")]
    pub price_per_k_upserts: u64,

    /// Overage price per 1K queries beyond tier quota (payment token base units).
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

fn default_tiers() -> Vec<StorageTier> {
    vec![
        StorageTier {
            name: "starter".into(),
            storage_gb: 1,
            included_queries_per_month: 100_000,
            included_upserts_per_month: 50_000,
            subscription_rate: 100_000,
        },
        StorageTier {
            name: "growth".into(),
            storage_gb: 10,
            included_queries_per_month: 1_000_000,
            included_upserts_per_month: 500_000,
            subscription_rate: 750_000,
        },
        StorageTier {
            name: "scale".into(),
            storage_gb: 100,
            included_queries_per_month: 10_000_000,
            included_upserts_per_month: 5_000_000,
            subscription_rate: 5_000_000,
        },
    ]
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
            tiers: default_tiers(),
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

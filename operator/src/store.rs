use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::config::VectorBackend;

// ── Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: u32,
    pub distance_metric: DistanceMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub name: String,
    pub vector_count: u64,
    pub dimensions: u32,
    pub distance_metric: DistanceMetric,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl DistanceMetric {
    fn to_qdrant_str(self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "Cosine",
            DistanceMetric::Euclidean => "Euclid",
            DistanceMetric::DotProduct => "Dot",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResult {
    pub upserted_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResult {
    pub deleted_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    #[serde(default)]
    pub must: Vec<FilterCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub key: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimensions: u32,
    #[serde(default = "default_distance")]
    pub distance_metric: DistanceMetric,
}

fn default_distance() -> DistanceMetric {
    DistanceMetric::Cosine
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub vectors: Vec<Point>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub filter: Option<QueryFilter>,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteVectorsRequest {
    pub ids: Vec<String>,
}

// ── Trait ───────────────────────────────────────────────────────────────

#[async_trait::async_trait]
pub trait VectorStoreBackend: Send + Sync + 'static {
    async fn create_collection(
        &self,
        name: &str,
        dimensions: u32,
        distance: DistanceMetric,
    ) -> anyhow::Result<()>;

    async fn list_collections(&self) -> anyhow::Result<Vec<CollectionInfo>>;

    async fn delete_collection(&self, name: &str) -> anyhow::Result<()>;

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<Point>,
    ) -> anyhow::Result<UpsertResult>;

    async fn query(
        &self,
        collection: &str,
        vector: Vec<f32>,
        top_k: usize,
        filter: Option<QueryFilter>,
    ) -> anyhow::Result<Vec<ScoredPoint>>;

    async fn delete_vectors(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> anyhow::Result<DeleteResult>;

    async fn collection_stats(&self, name: &str) -> anyhow::Result<CollectionStats>;
}

// ── Qdrant Backend ─────────────────────────────────────────────────────

pub struct QdrantBackend {
    client: reqwest::Client,
    url: String,
}

impl QdrantBackend {
    pub fn new(url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.trim_end_matches('/').to_string(),
        }
    }
}

#[async_trait::async_trait]
impl VectorStoreBackend for QdrantBackend {
    async fn create_collection(
        &self,
        name: &str,
        dimensions: u32,
        distance: DistanceMetric,
    ) -> anyhow::Result<()> {
        let resp = self
            .client
            .put(format!("{}/collections/{}", self.url, name))
            .json(&serde_json::json!({
                "vectors": {
                    "size": dimensions,
                    "distance": distance.to_qdrant_str()
                }
            }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant create_collection failed: {body}");
        }
        Ok(())
    }

    async fn list_collections(&self) -> anyhow::Result<Vec<CollectionInfo>> {
        let resp = self
            .client
            .get(format!("{}/collections", self.url))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant list_collections failed: {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let collections = body["result"]["collections"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let mut result = Vec::new();
        for c in collections {
            let name = c["name"].as_str().unwrap_or_default().to_string();
            // Fetch full info for dimensions and distance
            if let Ok(stats) = self.collection_stats(&name).await {
                result.push(CollectionInfo {
                    name,
                    dimensions: stats.dimensions,
                    distance_metric: stats.distance_metric,
                });
            }
        }
        Ok(result)
    }

    async fn delete_collection(&self, name: &str) -> anyhow::Result<()> {
        let resp = self
            .client
            .delete(format!("{}/collections/{}", self.url, name))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant delete_collection failed: {body}");
        }
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<Point>,
    ) -> anyhow::Result<UpsertResult> {
        let count = points.len() as u64;
        let qdrant_points: Vec<serde_json::Value> = points
            .into_iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "vector": p.vector,
                    "payload": p.metadata
                })
            })
            .collect();

        let resp = self
            .client
            .put(format!("{}/collections/{}/points", self.url, collection))
            .json(&serde_json::json!({ "points": qdrant_points }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant upsert failed: {body}");
        }

        Ok(UpsertResult {
            upserted_count: count,
        })
    }

    async fn query(
        &self,
        collection: &str,
        vector: Vec<f32>,
        top_k: usize,
        filter: Option<QueryFilter>,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        let mut body = serde_json::json!({
            "vector": vector,
            "limit": top_k,
            "with_payload": true
        });

        if let Some(f) = filter {
            let must: Vec<serde_json::Value> = f
                .must
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "key": c.key,
                        "match": { "value": c.value }
                    })
                })
                .collect();
            body["filter"] = serde_json::json!({ "must": must });
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.url, collection
            ))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant query failed: {text}");
        }

        let resp_body: serde_json::Value = resp.json().await?;
        let results = resp_body["result"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        Ok(results
            .into_iter()
            .map(|r| ScoredPoint {
                id: r["id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .or_else(|| r["id"].as_u64().map(|n| n.to_string()))
                    .or_else(|| r["id"].as_i64().map(|n| n.to_string()))
                    .unwrap_or_default(),
                score: r["score"].as_f64().unwrap_or(0.0) as f32,
                metadata: r["payload"].clone(),
            })
            .collect())
    }

    async fn delete_vectors(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> anyhow::Result<DeleteResult> {
        let count = ids.len() as u64;
        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/delete",
                self.url, collection
            ))
            .json(&serde_json::json!({ "points": ids }))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant delete_vectors failed: {body}");
        }

        Ok(DeleteResult {
            deleted_count: count,
        })
    }

    async fn collection_stats(&self, name: &str) -> anyhow::Result<CollectionStats> {
        let resp = self
            .client
            .get(format!("{}/collections/{}", self.url, name))
            .send()
            .await?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("qdrant collection_stats failed: {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let result = &body["result"];
        let config = &result["config"]["params"]["vectors"];

        let dimensions = config["size"].as_u64().unwrap_or(0) as u32;
        let distance_str = config["distance"].as_str().unwrap_or("Cosine");
        let distance_metric = match distance_str {
            "Euclid" => DistanceMetric::Euclidean,
            "Dot" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };

        let vector_count = result["points_count"]
            .as_u64()
            .or_else(|| result["vectors_count"].as_u64())
            .unwrap_or(0);
        // Approximate size: count * (dimensions * 4 bytes + overhead)
        let size_bytes = vector_count * (dimensions as u64 * 4 + 64);

        Ok(CollectionStats {
            name: name.to_string(),
            vector_count,
            dimensions,
            distance_metric,
            size_bytes,
        })
    }
}

// ── InMemory Backend ───────────────────────────────────────────────────

struct InMemoryCollection {
    dimensions: u32,
    distance_metric: DistanceMetric,
    vectors: DashMap<String, (Vec<f32>, serde_json::Value)>,
    vector_count: AtomicU64,
}

pub struct InMemoryBackend {
    collections: DashMap<String, InMemoryCollection>,
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self {
            collections: DashMap::new(),
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        sum += a[i] * b[i];
    }
    sum
}

fn score_vectors(metric: DistanceMetric, query: &[f32], candidate: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(query, candidate),
        // For Euclidean, lower distance = better, so negate for ranking
        DistanceMetric::Euclidean => -euclidean_distance(query, candidate),
        DistanceMetric::DotProduct => dot_product(query, candidate),
    }
}

fn matches_filter(metadata: &serde_json::Value, filter: &QueryFilter) -> bool {
    for cond in &filter.must {
        if let Some(val) = metadata.get(&cond.key) {
            if val != &cond.value {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

#[async_trait::async_trait]
impl VectorStoreBackend for InMemoryBackend {
    async fn create_collection(
        &self,
        name: &str,
        dimensions: u32,
        distance: DistanceMetric,
    ) -> anyhow::Result<()> {
        use dashmap::mapref::entry::Entry;
        match self.collections.entry(name.to_string()) {
            Entry::Occupied(_) => {
                anyhow::bail!("collection '{name}' already exists");
            }
            Entry::Vacant(e) => {
                e.insert(InMemoryCollection {
                    dimensions,
                    distance_metric: distance,
                    vectors: DashMap::new(),
                    vector_count: AtomicU64::new(0),
                });
            }
        }
        Ok(())
    }

    async fn list_collections(&self) -> anyhow::Result<Vec<CollectionInfo>> {
        Ok(self
            .collections
            .iter()
            .map(|entry| CollectionInfo {
                name: entry.key().clone(),
                dimensions: entry.value().dimensions,
                distance_metric: entry.value().distance_metric,
            })
            .collect())
    }

    async fn delete_collection(&self, name: &str) -> anyhow::Result<()> {
        self.collections
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("collection '{name}' not found"))?;
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        points: Vec<Point>,
    ) -> anyhow::Result<UpsertResult> {
        let coll = self
            .collections
            .get(collection)
            .ok_or_else(|| anyhow::anyhow!("collection '{collection}' not found"))?;

        let mut upserted = 0u64;
        for point in points {
            if point.vector.len() != coll.dimensions as usize {
                anyhow::bail!(
                    "dimension mismatch: expected {}, got {}",
                    coll.dimensions,
                    point.vector.len()
                );
            }
            use dashmap::mapref::entry::Entry;
            match coll.vectors.entry(point.id) {
                Entry::Occupied(mut e) => {
                    e.insert((point.vector, point.metadata));
                }
                Entry::Vacant(e) => {
                    e.insert((point.vector, point.metadata));
                    coll.vector_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            upserted += 1;
        }

        Ok(UpsertResult {
            upserted_count: upserted,
        })
    }

    async fn query(
        &self,
        collection: &str,
        vector: Vec<f32>,
        top_k: usize,
        filter: Option<QueryFilter>,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        let coll = self
            .collections
            .get(collection)
            .ok_or_else(|| anyhow::anyhow!("collection '{collection}' not found"))?;

        if vector.len() != coll.dimensions as usize {
            anyhow::bail!(
                "query vector dimension mismatch: collection has {} dimensions, query has {}",
                coll.dimensions,
                vector.len()
            );
        }

        let mut scored: Vec<(String, f32, serde_json::Value)> = Vec::new();

        for entry in coll.vectors.iter() {
            let (candidate_vec, metadata) = entry.value();

            if let Some(ref f) = filter {
                if !matches_filter(metadata, f) {
                    continue;
                }
            }

            let score = score_vectors(coll.distance_metric, &vector, candidate_vec);
            scored.push((entry.key().clone(), score, metadata.clone()));
        }

        // Sort descending by score (higher = more similar for cosine/dot, less negative for euclidean)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(id, score, metadata)| ScoredPoint {
                id,
                score,
                metadata,
            })
            .collect())
    }

    async fn delete_vectors(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> anyhow::Result<DeleteResult> {
        let coll = self
            .collections
            .get(collection)
            .ok_or_else(|| anyhow::anyhow!("collection '{collection}' not found"))?;

        let mut deleted = 0u64;
        for id in &ids {
            if coll.vectors.remove(id).is_some() {
                coll.vector_count.fetch_sub(1, Ordering::Relaxed);
                deleted += 1;
            }
        }

        Ok(DeleteResult {
            deleted_count: deleted,
        })
    }

    async fn collection_stats(&self, name: &str) -> anyhow::Result<CollectionStats> {
        let coll = self
            .collections
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("collection '{name}' not found"))?;

        let count = coll.vector_count.load(Ordering::Relaxed);
        // Approximate: each vector = dimensions * 4 bytes + 64 bytes overhead for id/metadata
        let size_bytes = count * (coll.dimensions as u64 * 4 + 64);

        Ok(CollectionStats {
            name: name.to_string(),
            vector_count: count,
            dimensions: coll.dimensions,
            distance_metric: coll.distance_metric,
            size_bytes,
        })
    }
}

// ── Proxy Backend (ChromaDB, pgvector, Milvus, Weaviate, Pinecone) ────
//
// Each of these wraps a remote vector database's REST API. They share the
// same trait and pattern: HTTP client + URL + provider-specific request/
// response mapping. For v1, they all delegate to a `GenericHttpBackend`
// that maps our trait calls to provider-specific endpoints.

/// Generic HTTP-based vector store backend. Wraps any REST API that
/// exposes create/list/delete/upsert/query/stats endpoints.
pub struct HttpProxyBackend {
    client: reqwest::Client,
    url: String,
    provider: ProxyProvider,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProxyProvider {
    ChromaDB,
    PgVector,
    Milvus,
    Weaviate,
    Pinecone { environment: String },
}

impl HttpProxyBackend {
    pub fn new(url: String, provider: ProxyProvider, api_key: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.trim_end_matches('/').to_string(),
            provider,
            api_key,
        }
    }

    fn auth_header(&self) -> Option<(&str, &str)> {
        match (&self.provider, &self.api_key) {
            (ProxyProvider::Pinecone { .. }, Some(key)) => Some(("Api-Key", key.as_str())),
            (_, Some(key)) => Some(("Authorization", key.as_str())),
            _ => None,
        }
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.request(method, format!("{}{}", self.url, path));
        if let Some((header, value)) = self.auth_header() {
            req = req.header(header, value);
        }
        req
    }
}

#[async_trait::async_trait]
impl VectorStoreBackend for HttpProxyBackend {
    async fn create_collection(
        &self,
        name: &str,
        dimensions: u32,
        distance: DistanceMetric,
    ) -> anyhow::Result<()> {
        let body = match &self.provider {
            ProxyProvider::ChromaDB => serde_json::json!({
                "name": name,
                "metadata": { "dimensions": dimensions, "distance": distance.to_qdrant_str() }
            }),
            ProxyProvider::PgVector => serde_json::json!({
                "name": name, "dimensions": dimensions, "distance": distance.to_qdrant_str()
            }),
            ProxyProvider::Milvus => serde_json::json!({
                "collectionName": name,
                "schema": {
                    "fields": [
                        { "fieldName": "id", "dataType": "VarChar", "isPrimary": true, "params": { "max_length": "256" } },
                        { "fieldName": "vector", "dataType": "FloatVector", "params": { "dim": dimensions } },
                        { "fieldName": "metadata", "dataType": "JSON" }
                    ]
                }
            }),
            ProxyProvider::Weaviate => serde_json::json!({
                "class": name,
                "vectorizer": "none",
                "properties": [{ "name": "metadata", "dataType": ["text"] }]
            }),
            ProxyProvider::Pinecone { .. } => serde_json::json!({
                "name": name,
                "dimension": dimensions,
                "metric": match distance { DistanceMetric::Cosine => "cosine", DistanceMetric::Euclidean => "euclidean", DistanceMetric::DotProduct => "dotproduct" }
            }),
        };

        let path = match &self.provider {
            ProxyProvider::ChromaDB => "/api/v1/collections".to_string(),
            ProxyProvider::PgVector => "/rpc/create_collection".to_string(),
            ProxyProvider::Milvus => "/v2/vectordb/collections/create".to_string(),
            ProxyProvider::Weaviate => "/v1/schema".to_string(),
            ProxyProvider::Pinecone { .. } => {
                // Pinecone control plane
                return self.create_pinecone_index(name, dimensions, distance).await;
            }
        };

        let resp = self.request(reqwest::Method::POST, &path).json(&body).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("{:?} create_collection failed: {err}", self.provider);
        }
        Ok(())
    }

    async fn list_collections(&self) -> anyhow::Result<Vec<CollectionInfo>> {
        let path = match &self.provider {
            ProxyProvider::ChromaDB => "/api/v1/collections",
            ProxyProvider::PgVector => "/rpc/list_collections",
            ProxyProvider::Milvus => "/v2/vectordb/collections/list",
            ProxyProvider::Weaviate => "/v1/schema",
            ProxyProvider::Pinecone { .. } => "https://api.pinecone.io/indexes",
        };

        let resp = self.request(reqwest::Method::GET, path).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("{:?} list_collections failed", self.provider);
        }

        let json: serde_json::Value = resp.json().await?;

        // Each provider returns collections differently — extract names
        let names: Vec<String> = match &self.provider {
            ProxyProvider::ChromaDB => json
                .as_array()
                .map(|a| a.iter().filter_map(|c| c["name"].as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            ProxyProvider::Milvus => json["data"]
                .as_array()
                .map(|a| a.iter().filter_map(|s| s.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            ProxyProvider::Weaviate => json["classes"]
                .as_array()
                .map(|a| a.iter().filter_map(|c| c["class"].as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            ProxyProvider::Pinecone { .. } => json["indexes"]
                .as_array()
                .map(|a| a.iter().filter_map(|i| i["name"].as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            ProxyProvider::PgVector => json
                .as_array()
                .map(|a| a.iter().filter_map(|c| c["name"].as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
        };

        Ok(names.into_iter().map(|name| CollectionInfo { name, dimensions: 0, distance_metric: DistanceMetric::Cosine }).collect())
    }

    async fn delete_collection(&self, name: &str) -> anyhow::Result<()> {
        let (method, path) = match &self.provider {
            ProxyProvider::ChromaDB => (reqwest::Method::DELETE, format!("/api/v1/collections/{name}")),
            ProxyProvider::PgVector => (reqwest::Method::POST, "/rpc/delete_collection".to_string()),
            ProxyProvider::Milvus => (reqwest::Method::POST, "/v2/vectordb/collections/drop".to_string()),
            ProxyProvider::Weaviate => (reqwest::Method::DELETE, format!("/v1/schema/{name}")),
            ProxyProvider::Pinecone { .. } => (reqwest::Method::DELETE, format!("/indexes/{name}")),
        };

        let mut req = self.request(method, &path);
        if matches!(self.provider, ProxyProvider::Milvus | ProxyProvider::PgVector) {
            req = req.json(&serde_json::json!({ "collectionName": name, "name": name }));
        }

        let resp = req.send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(()); // idempotent
        }
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("{:?} delete_collection failed: {err}", self.provider);
        }
        Ok(())
    }

    async fn upsert(&self, collection: &str, points: Vec<Point>) -> anyhow::Result<UpsertResult> {
        let count = points.len() as u64;
        let body = match &self.provider {
            ProxyProvider::ChromaDB => serde_json::json!({
                "ids": points.iter().map(|p| &p.id).collect::<Vec<_>>(),
                "embeddings": points.iter().map(|p| &p.vector).collect::<Vec<_>>(),
                "metadatas": points.iter().map(|p| &p.metadata).collect::<Vec<_>>(),
            }),
            ProxyProvider::Milvus => serde_json::json!({
                "collectionName": collection,
                "data": points.iter().map(|p| serde_json::json!({
                    "id": p.id, "vector": p.vector, "metadata": p.metadata
                })).collect::<Vec<_>>(),
            }),
            ProxyProvider::Weaviate => serde_json::json!({
                "objects": points.iter().map(|p| serde_json::json!({
                    "class": collection,
                    "id": p.id,
                    "vector": p.vector,
                    "properties": { "metadata": p.metadata.to_string() }
                })).collect::<Vec<_>>(),
            }),
            ProxyProvider::Pinecone { .. } => serde_json::json!({
                "vectors": points.iter().map(|p| serde_json::json!({
                    "id": p.id, "values": p.vector, "metadata": p.metadata
                })).collect::<Vec<_>>(),
            }),
            ProxyProvider::PgVector => serde_json::json!({
                "collection": collection,
                "points": points.iter().map(|p| serde_json::json!({
                    "id": p.id, "vector": p.vector, "metadata": p.metadata
                })).collect::<Vec<_>>(),
            }),
        };

        let path = match &self.provider {
            ProxyProvider::ChromaDB => format!("/api/v1/collections/{collection}/upsert"),
            ProxyProvider::Milvus => "/v2/vectordb/entities/upsert".to_string(),
            ProxyProvider::Weaviate => "/v1/batch/objects".to_string(),
            ProxyProvider::Pinecone { .. } => "/vectors/upsert".to_string(),
            ProxyProvider::PgVector => "/rpc/upsert_vectors".to_string(),
        };

        let resp = self.request(reqwest::Method::POST, &path).json(&body).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("{:?} upsert failed: {err}", self.provider);
        }
        Ok(UpsertResult { upserted_count: count })
    }

    async fn query(
        &self,
        collection: &str,
        vector: Vec<f32>,
        top_k: usize,
        _filter: Option<QueryFilter>,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        let body = match &self.provider {
            ProxyProvider::ChromaDB => serde_json::json!({
                "query_embeddings": [vector],
                "n_results": top_k,
            }),
            ProxyProvider::Milvus => serde_json::json!({
                "collectionName": collection,
                "data": [vector],
                "limit": top_k,
                "outputFields": ["metadata"],
            }),
            ProxyProvider::Weaviate => serde_json::json!({
                "query": format!(
                    "{{ Get {{ {collection}(nearVector: {{ vector: {vector:?} }}, limit: {top_k}) {{ _additional {{ id distance }} metadata }} }} }}"
                ),
            }),
            ProxyProvider::Pinecone { .. } => serde_json::json!({
                "vector": vector,
                "topK": top_k,
                "includeMetadata": true,
            }),
            ProxyProvider::PgVector => serde_json::json!({
                "collection": collection,
                "vector": vector,
                "top_k": top_k,
            }),
        };

        let path = match &self.provider {
            ProxyProvider::ChromaDB => format!("/api/v1/collections/{collection}/query"),
            ProxyProvider::Milvus => "/v2/vectordb/entities/search".to_string(),
            ProxyProvider::Weaviate => "/v1/graphql".to_string(),
            ProxyProvider::Pinecone { .. } => "/query".to_string(),
            ProxyProvider::PgVector => "/rpc/query_vectors".to_string(),
        };

        let resp = self.request(reqwest::Method::POST, &path).json(&body).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("{:?} query failed: {err}", self.provider);
        }

        let json: serde_json::Value = resp.json().await?;

        // Parse results (provider-specific)
        let results = match &self.provider {
            ProxyProvider::ChromaDB => {
                let ids = json["ids"][0].as_array();
                let distances = json["distances"][0].as_array();
                let metadatas = json["metadatas"][0].as_array();
                match (ids, distances) {
                    (Some(ids), Some(dists)) => ids.iter().zip(dists.iter()).enumerate().map(|(i, (id, dist))| {
                        ScoredPoint {
                            id: id.as_str().unwrap_or_default().to_string(),
                            score: 1.0 - dist.as_f64().unwrap_or(1.0) as f32,
                            metadata: metadatas.and_then(|m| m.get(i)).cloned().unwrap_or_default(),
                        }
                    }).collect(),
                    _ => Vec::new(),
                }
            }
            ProxyProvider::Milvus => json["data"]
                .as_array()
                .map(|a| a.iter().map(|r| ScoredPoint {
                    id: r["id"].as_str().or_else(|| r["id"].as_u64().map(|_| "")).map(|s| s.to_string()).unwrap_or_default(),
                    score: r["distance"].as_f64().unwrap_or(0.0) as f32,
                    metadata: r["metadata"].clone(),
                }).collect())
                .unwrap_or_default(),
            ProxyProvider::Weaviate => {
                // GraphQL response parsing
                let get = &json["data"]["Get"];
                if let Some(obj) = get.as_object().and_then(|o| o.values().next()) {
                    obj.as_array().map(|a| a.iter().map(|r| ScoredPoint {
                        id: r["_additional"]["id"].as_str().unwrap_or_default().to_string(),
                        score: 1.0 - r["_additional"]["distance"].as_f64().unwrap_or(1.0) as f32,
                        metadata: serde_json::json!(r["metadata"]),
                    }).collect()).unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            ProxyProvider::Pinecone { .. } => json["matches"]
                .as_array()
                .map(|a| a.iter().map(|r| ScoredPoint {
                    id: r["id"].as_str().unwrap_or_default().to_string(),
                    score: r["score"].as_f64().unwrap_or(0.0) as f32,
                    metadata: r["metadata"].clone(),
                }).collect())
                .unwrap_or_default(),
            ProxyProvider::PgVector => json
                .as_array()
                .map(|a| a.iter().map(|r| ScoredPoint {
                    id: r["id"].as_str().unwrap_or_default().to_string(),
                    score: r["score"].as_f64().unwrap_or(0.0) as f32,
                    metadata: r["metadata"].clone(),
                }).collect())
                .unwrap_or_default(),
        };

        Ok(results)
    }

    async fn delete_vectors(&self, collection: &str, ids: Vec<String>) -> anyhow::Result<DeleteResult> {
        let count = ids.len() as u64;
        let body = match &self.provider {
            ProxyProvider::ChromaDB => serde_json::json!({ "ids": ids }),
            ProxyProvider::Milvus => serde_json::json!({ "collectionName": collection, "filter": format!("id in {:?}", ids) }),
            ProxyProvider::Weaviate => serde_json::json!({ "match": { "class": collection, "ids": ids } }),
            ProxyProvider::Pinecone { .. } => serde_json::json!({ "ids": ids }),
            ProxyProvider::PgVector => serde_json::json!({ "collection": collection, "ids": ids }),
        };

        let path = match &self.provider {
            ProxyProvider::ChromaDB => format!("/api/v1/collections/{collection}/delete"),
            ProxyProvider::Milvus => "/v2/vectordb/entities/delete".to_string(),
            ProxyProvider::Weaviate => "/v1/batch/objects".to_string(),
            ProxyProvider::Pinecone { .. } => "/vectors/delete".to_string(),
            ProxyProvider::PgVector => "/rpc/delete_vectors".to_string(),
        };

        let method = if matches!(self.provider, ProxyProvider::Weaviate) {
            reqwest::Method::DELETE
        } else {
            reqwest::Method::POST
        };

        let resp = self.request(method, &path).json(&body).send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(DeleteResult { deleted_count: 0 });
        }
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("{:?} delete_vectors failed: {err}", self.provider);
        }
        Ok(DeleteResult { deleted_count: count })
    }

    async fn collection_stats(&self, name: &str) -> anyhow::Result<CollectionStats> {
        // Most providers don't have a clean stats endpoint — return best-effort
        Ok(CollectionStats {
            name: name.to_string(),
            vector_count: 0, // would need provider-specific query
            dimensions: 0,
            distance_metric: DistanceMetric::Cosine,
            size_bytes: 0,
        })
    }
}

impl HttpProxyBackend {
    async fn create_pinecone_index(
        &self,
        name: &str,
        dimensions: u32,
        distance: DistanceMetric,
    ) -> anyhow::Result<()> {
        let metric = match distance {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::DotProduct => "dotproduct",
        };
        let body = serde_json::json!({
            "name": name,
            "dimension": dimensions,
            "metric": metric,
            "spec": { "serverless": { "cloud": "aws", "region": "us-east-1" } }
        });
        let resp = self.client
            .post("https://api.pinecone.io/indexes")
            .header("Api-Key", self.api_key.as_deref().unwrap_or_default())
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            anyhow::bail!("Pinecone create index failed: {err}");
        }
        Ok(())
    }
}

/// Build the appropriate backend from config.
pub fn build_backend(backend: &VectorBackend) -> Box<dyn VectorStoreBackend> {
    match backend {
        VectorBackend::Qdrant { url } => Box::new(QdrantBackend::new(url.clone())),
        VectorBackend::ChromaDB { url } => Box::new(HttpProxyBackend::new(
            url.clone(), ProxyProvider::ChromaDB, None,
        )),
        VectorBackend::PgVector { connection_string } => Box::new(HttpProxyBackend::new(
            connection_string.clone(), ProxyProvider::PgVector, None,
        )),
        VectorBackend::Milvus { url } => Box::new(HttpProxyBackend::new(
            url.clone(), ProxyProvider::Milvus, None,
        )),
        VectorBackend::Weaviate { url } => Box::new(HttpProxyBackend::new(
            url.clone(), ProxyProvider::Weaviate, None,
        )),
        VectorBackend::Pinecone { api_key, environment } => Box::new(HttpProxyBackend::new(
            format!("https://controller.{environment}.pinecone.io"),
            ProxyProvider::Pinecone { environment: environment.clone() },
            Some(api_key.clone()),
        )),
        VectorBackend::InMemory => Box::new(InMemoryBackend::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inmemory_crud() {
        let backend = InMemoryBackend::new();

        backend
            .create_collection("test", 3, DistanceMetric::Cosine)
            .await
            .unwrap();

        let collections = backend.list_collections().await.unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name, "test");

        backend
            .upsert(
                "test",
                vec![
                    Point {
                        id: "a".into(),
                        vector: vec![1.0, 0.0, 0.0],
                        metadata: serde_json::json!({"color": "red"}),
                    },
                    Point {
                        id: "b".into(),
                        vector: vec![0.0, 1.0, 0.0],
                        metadata: serde_json::json!({"color": "blue"}),
                    },
                    Point {
                        id: "c".into(),
                        vector: vec![0.9, 0.1, 0.0],
                        metadata: serde_json::json!({"color": "red"}),
                    },
                ],
            )
            .await
            .unwrap();

        let stats = backend.collection_stats("test").await.unwrap();
        assert_eq!(stats.vector_count, 3);

        // Query — closest to [1,0,0] should be "a" then "c"
        let results = backend
            .query("test", vec![1.0, 0.0, 0.0], 2, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "c");

        // Query with filter
        let results = backend
            .query(
                "test",
                vec![1.0, 0.0, 0.0],
                10,
                Some(QueryFilter {
                    must: vec![FilterCondition {
                        key: "color".into(),
                        value: serde_json::json!("blue"),
                    }],
                }),
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");

        // Delete
        let del = backend
            .delete_vectors("test", vec!["a".into()])
            .await
            .unwrap();
        assert_eq!(del.deleted_count, 1);

        let stats = backend.collection_stats("test").await.unwrap();
        assert_eq!(stats.vector_count, 2);

        // Delete collection
        backend.delete_collection("test").await.unwrap();
        let collections = backend.list_collections().await.unwrap();
        assert_eq!(collections.len(), 0);
    }

    #[test]
    fn cosine_similarity_identical() {
        let score = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let score = cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(score.abs() < 1e-6);
    }
}

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

/// Build the appropriate backend from config.
pub fn build_backend(backend: &VectorBackend) -> Box<dyn VectorStoreBackend> {
    match backend {
        VectorBackend::Qdrant { url } => Box::new(QdrantBackend::new(url.clone())),
        VectorBackend::ChromaDB { url } => {
            tracing::warn!(url = %url, "ChromaDB backend not yet implemented, falling back to InMemory");
            Box::new(InMemoryBackend::new())
        }
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

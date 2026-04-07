# Vector Store Blueprint

Tangle Blueprint for hosted vector storage and similarity search — the missing piece for RAG.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/collections` | Create a collection |
| GET | `/v1/collections` | List collections |
| DELETE | `/v1/collections/:name` | Delete a collection |
| POST | `/v1/collections/:name/upsert` | Upsert vectors |
| POST | `/v1/collections/:name/query` | Similarity search |
| DELETE | `/v1/collections/:name/vectors` | Delete vectors by ID |
| GET | `/v1/collections/:name/stats` | Collection stats |

## Backends

| Backend | Use Case | Config |
|---------|----------|--------|
| Qdrant | Production RAG | `backend = "qdrant"`, `url = "http://localhost:6333"` |
| ChromaDB | Planned | `backend = "chromadb"` |
| InMemory | Dev/testing | `backend = "inmemory"` (default) |

## Pricing

| Dimension | Default | Unit |
|-----------|---------|------|
| Storage | 0.10 tsUSD | per GB/month |
| Writes | 0.01 tsUSD | per 1K vectors upserted |
| Queries | 0.005 tsUSD | per 1K similarity searches |

## Build

```bash
# Contracts
cd contracts && forge install && forge build

# Operator
cargo build -p vector-store
```

## Run

```bash
# Dev mode (in-memory backend)
VSTORE_OP__VECTOR_STORE__BACKEND=inmemory cargo run -p vector-store

# Production (Qdrant)
VSTORE_OP__VECTOR_STORE__BACKEND=qdrant \
VSTORE_OP__VECTOR_STORE__URL=http://localhost:6333 \
cargo run -p vector-store
```

## Quick Start

```bash
# Create a collection
curl -X POST http://localhost:8080/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{"name": "docs", "dimensions": 1536, "distance_metric": "cosine"}'

# Upsert vectors
curl -X POST http://localhost:8080/v1/collections/docs/upsert \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [{"id": "1", "vector": [0.1, 0.2, ...], "metadata": {"source": "readme"}}]}'

# Query
curl -X POST http://localhost:8080/v1/collections/docs/query \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, ...], "top_k": 5}'
```

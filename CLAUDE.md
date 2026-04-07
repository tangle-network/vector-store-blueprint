# CLAUDE.md

## Project Overview

Vector Store Blueprint for Tangle Network. Operators serve hosted vector storage and similarity search for RAG pipelines. Supports multiple backends: Qdrant (production), ChromaDB (planned), InMemory (dev/testing).

## Architecture

Depends on [`tangle-inference-core`](../tangle-inference-core/) for shared billing, metrics, health, nonce store, x402 payment.

- **contracts/src/VectorStoreBSM.sol** — operator registration, three-dimensional pricing (storage, writes, queries), capacity limits
- **operator/src/store.rs** — `VectorStoreBackend` trait + `QdrantBackend` + `InMemoryBackend` implementations
- **operator/src/server.rs** — REST API (7 endpoints), billing via `billing_gate` for upserts and queries
- **operator/src/config.rs** — imports shared config from core, adds `VectorStoreConfig` (backend, pricing, limits)
- **operator/src/lib.rs** — `VectorStoreServer` BackgroundService, on-chain job handler

## Build

```bash
cd contracts && forge build
cargo build -p vector-store
```

## Billing

Per-request pricing via `FlatRequestCostModel`. Upserts billed proportional to batch size (count/1000 * price_per_k_upserts). Queries billed per request (price_per_k_queries / 1000). Storage billing (per-GB-month) deferred to subscription model.

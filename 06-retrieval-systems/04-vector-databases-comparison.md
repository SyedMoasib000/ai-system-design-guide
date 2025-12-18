# Vector Databases Comparison (Dec 2025)

Vector databases are the storage engines for high-dimensional embeddings. In late 2025, the market has split into **Managed Serverless** and **Specialized High-Performance** engines.

## Table of Contents

- [The 2025 Competitive Landscape](#landscape)
- [Indexing Algorithms: HNSW vs. DiskANN](#indexing)
- [Managed vs. Self-Hosted (TCO Analysis)](#tco)
- [Advanced Features (Tenancy, ACLs, Serverless)](#features)
- [Detailed Database Matrix](#matrix)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The 2025 Competitive Landscape

We no longer ask "Does it support vector search?" (Postgres, Redis, and Mongo all do). We ask **"Does it scale to 100M+ vectors with sub-100ms P99 and full metadata filtering?"**

### 1. Vector-Native (Dedicated)
- **Pinecone**: The serverless standard. 
- **Qdrant**: High-performance Rust, excellent filtering.
- **Weaviate**: Best for graph-like relationships and multi-modal.
- **Milvus**: The enterprise Kubernetes heavy-lifter.

### 2. General-Purpose (Plugin/Extension)
- **pgvector (v0.8+)**: Now supports HNSW and IVFFlat with massive performance improvements.
- **Elasticsearch (v9.0)**: Best for Hybrid Search with cross-entropy fusion.

---

## Indexing Algorithms: HNSW vs. DiskANN

### HNSW (Hierarchical Navigable Small World)
- **Status**: The industry standard for **In-Memory** search.
- **Nuance**: Memory usage is high (O(n * d)). 10M vectors at 1536 dims require ~80GB of RAM.
- **Win**: Fastest query latency (low ms).

### DiskANN (SSD-based)
- **Status**: The industry standard for **Petabyte-Scale** search.
- **Nuance**: Keeps the graph on SSD (NVMe) and only a tiny index in RAM.
- **Win**: 10x cheaper than HNSW for billion-scale datasets with <5ms latency penalty.

---

## Managed vs. Self-Hosted (TCO Analysis)

| Aspect | Pinecone (Serverless) | Self-Hosted (Qdrant/Milvus) |
|--------|-----------------------|-----------------------------|
| **Ops Overhead** | Zero | High (Requires K8s + SRE) |
| **Scaling** | Instant (Scale to zero) | Manual (Node provisioning) |
| **Cost (Small)** | $0 - $100/mo | $50/mo (Minimum instance) |
| **Cost (Scale)** | High per token/vector | Low unit cost |

**2025 Verdict**: Start with Serverless. Only self-host if you have >500M vectors or strict **On-Prem/GPU-Local** requirements.

---

## Advanced Features

- **Multi-Tenancy**: How does the DB handle 1,000 customers?
  - *Hard Isolation*: One index per customer (Expensive).
  - *Soft Isolation*: Metadata filtering (e.g., `WHERE tenant_id = 'A'`).
- **Disk-Native Metadata**: In 2025, modern DBs like **Qdrant** offload metadata to disk-mapped segments, allowing for complex filters (e.g., full-text + geo + vector) without saturating RAM.

---

## Detailed Database Matrix

| Feature | Pinecone | Qdrant | Milvus | pgvector |
|---------|----------|--------|--------|----------|
| **Language** | Proprietary | Rust | Go/C++ | C |
| **Serverless**| Yes (Best) | Yes | Yes (Zilliz)| No (Cloud PG) |
| **Hybrid** | Native | Native | Native | Multi-stage |
| **Cloud-Native**| Any | Any | K8s Only | Any |

---

## Interview Questions

### Q: Why is metadata filtering often the bottleneck in vector databases?

**Strong answer:**
In naive vector search, we find the "Top K" nearest neighbors and THEN filter them by metadata (e.g., "only documents from 2024"). If the filter is very restrictive, we might find 0 results after filtering. In 2025, specialized databases use **Pre-Filtering with HNSW**. They traverse the graph but only consider nodes that satisfy the boolean metadata constraint. This is computationally expensive because it breaks the "short-circuit" logic of HNSW, requiring specialized bitmasks or hardware acceleration (SIMD) to keep latencies low.

### Q: When would you use a Disk-based index (like DiskANN) over a RAM-based index (HNSW)?

**Strong answer:**
I would use a Disk-based index when the memory cost of the index exceeds the budget or the capacity of a single high-memory node. For example, a 100-million-vector index with 1536 dimensions would require nearly 1TB of RAM for HNSW. Using DiskANN, I can store the majority of that 1TB on NVMe SSDs, reducing the RAM requirement by 90-95% while maintaining sub-10ms query times. This represents a massive TCO (Total Cost of Ownership) reduction for non-real-time search applications.

---

## References
- Malkov et al. "HNSW: Efficient and Robust ANN Search" (2018)
- Microsoft Research. "Vamana/DiskANN: A Disk-based Index for ANN Search" (2019/2023)
- Pinecone. "The Managed Architecture of Serverless Vector DBs" (2024)

---

*Next: [Hybrid Search](05-hybrid-search.md)*

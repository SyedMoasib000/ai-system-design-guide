# Vector Databases

Vector databases are purpose-built systems for storing, indexing, and searching high-dimensional embeddings. This chapter covers the options, tradeoffs, and production considerations for choosing and operating vector databases.

## Table of Contents

- [What Is a Vector Database](#what-is-a-vector-database)
- [Vector Search Fundamentals](#vector-search-fundamentals)
- [Indexing Algorithms](#indexing-algorithms)
- [Vector Database Comparison](#vector-database-comparison)
- [Query Patterns](#query-patterns)
- [Production Operations](#production-operations)
- [Cost Analysis](#cost-analysis)
- [Selection Framework](#selection-framework)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## What Is a Vector Database

A vector database stores embeddings (dense vectors) and enables fast similarity search over them.

```
Traditional DB:      SELECT * FROM docs WHERE category = 'tech'
Vector DB:           SELECT * FROM docs ORDER BY similarity(embedding, query_embedding) LIMIT 10
```

### Core Capabilities

| Capability | Purpose |
|------------|---------|
| Vector storage | Persist high-dimensional embeddings |
| Similarity search | Find nearest neighbors quickly |
| Metadata filtering | Combine vector search with attribute filters |
| CRUD operations | Update embeddings as data changes |
| Scaling | Handle millions to billions of vectors |

### Why Not General Databases?

Traditional databases can store vectors but lack optimized search:

| Approach | Search Complexity | Practical at Scale |
|----------|-------------------|-------------------|
| Brute force (PostgreSQL pgvector) | O(n * d) | OK to ~1M vectors |
| ANN index (dedicated vector DB) | O(log n) or O(1) | Yes, billions |

---

## Vector Search Fundamentals

### Exact vs Approximate Search

**Exact (brute force):**
- Compare query to every stored vector
- O(n * d) per query
- Perfect accuracy

**Approximate Nearest Neighbor (ANN):**
- Use index structure to prune search space
- Sub-linear complexity
- Slightly lower recall (typically 95-99%)

### Distance Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| Cosine | 1 - (a . b) / (norm(a) * norm(b)) | [0, 2] | Text embeddings |
| Euclidean (L2) | sqrt(sum((a - b)^2)) | [0, inf) | Image embeddings |
| Dot product | a . b | (-inf, inf) | Already normalized |

**For text embeddings:** Use cosine similarity (or dot product if pre-normalized).

### Recall vs Latency Tradeoff

```
                    ▲ Recall
                    │
               100% ┤ ────────────────── Brute force
                    │         ●          Well-tuned ANN
                    │      ●
                    │   ●
                95% ┤●                   Fast ANN
                    │
                    └─────┬───────┬──────► Latency
                       1ms      10ms
```

ANN indices trade some accuracy for speed. Tune for your requirements.

---

## Indexing Algorithms

### HNSW (Hierarchical Navigable Small World)

The most popular algorithm for production vector search.

**How it works:**
1. Build a graph where nodes are vectors
2. Connect to nearby neighbors
3. Multiple layers of abstraction (hierarchical)
4. Search: navigate from top layer down, greedy nearest neighbor

```
Layer 2:   ●────────●────────●
           │        │        │
Layer 1:   ●──●──●──●──●──●──●
           │  │  │  │  │  │  │
Layer 0:   ●●●●●●●●●●●●●●●●●●●●  (all vectors)
```

**Pros:**
- Excellent recall/latency tradeoff
- No training required
- Supports updates natively

**Cons:**
- Memory-intensive (graph structure)
- Index size: ~1.5-2x vector data

**Key parameters:**
- `M`: Max connections per node (16-64)
- `ef_construction`: Build-time exploration (100-500)
- `ef_search`: Query-time exploration (50-200)

### IVF (Inverted File Index)

Partition vectors into clusters, search only relevant clusters.

**How it works:**
1. Use k-means to create centroids
2. Assign each vector to nearest centroid
3. At query time: find nearest centroids, search those clusters

**Pros:**
- Lower memory than HNSW
- Can use quantization (IVF-PQ)

**Cons:**
- Requires training
- Updates need re-clustering or hybrid approach

**Key parameters:**
- `nlist`: Number of clusters (sqrt(n) rule of thumb)
- `nprobe`: Clusters to search at query time

### Product Quantization (PQ)

Compress vectors to reduce memory and speed up comparison.

**How it works:**
1. Split vector into subvectors
2. Quantize each subvector to a codebook
3. Store codes instead of full vectors

**Memory reduction:** 4-32x typical

**Tradeoff:** Lower accuracy due to quantization loss

### Flat Index (Brute Force)

No approximation, exact search.

**Use when:**
- Less than 100K vectors
- Accuracy is critical
- Latency budget is generous

### Algorithm Comparison

| Algorithm | Memory | Build Time | Query Speed | Recall | Updates |
|-----------|--------|------------|-------------|--------|---------|
| HNSW | High | Medium | Very fast | 95-99% | Good |
| IVF | Medium | Fast | Fast | 90-98% | Fair |
| IVF-PQ | Low | Fast | Fast | 85-95% | Fair |
| Flat | Low | None | Slow | 100% | Instant |

---

## Vector Database Comparison

### Major Options (December 2025)

| Database | Type | Best For | Pricing Model |
|----------|------|----------|---------------|
| **Pinecone** | Managed cloud | Easy start, scale | Per vector-hour |
| **Qdrant** | Open source / Cloud | Self-hosted control | Per GB (cloud) or free |
| **Weaviate** | Open source / Cloud | Multimodal, ML integration | Per dimension-hour |
| **Chroma** | Open source | Prototyping, local | Free |
| **Milvus** | Open source / Cloud | On-prem enterprise | Free (self-host) |
| **pgvector** | PostgreSQL extension | Small scale, existing PG | Compute only |

### Feature Comparison

| Feature | Pinecone | Qdrant | Weaviate | Milvus | pgvector |
|---------|----------|--------|----------|--------|----------|
| Hosted option | Yes | Yes | Yes | Yes (Zilliz) | Via cloud PG |
| Self-hosted | No | Yes | Yes | Yes | Yes |
| Metadata filtering | Good | Excellent | Good | Good | Via SQL |
| Hybrid search | Yes | Yes | Yes | Yes | Limited |
| Max vectors | Billions | Billions | Billions | Billions | ~10M |
| HNSW index | Yes | Yes | Yes | Yes | Yes |

### Metadata Filtering

Critical for multi-tenant and filtering use cases:

```python
# Pinecone
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={"tenant_id": "123", "category": {"$in": ["tech", "science"]}}
)

# Qdrant
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    query_filter=Filter(
        must=[
            FieldCondition(key="tenant_id", match=MatchValue(value="123")),
            FieldCondition(key="category", match=MatchAny(any=["tech", "science"]))
        ]
    )
)
```

**Performance impact:** Filtering happens during search, not after. Pre-filtered indices are faster but less flexible.

---

## Query Patterns

### Pattern 1: Simple Semantic Search

```python
def semantic_search(query: str, top_k: int = 5) -> list[Document]:
    query_embedding = embed(query)
    results = vector_db.search(query_embedding, top_k=top_k)
    return [Document(id=r.id, text=r.payload["text"], score=r.score) for r in results]
```

### Pattern 2: Filtered Search

```python
def filtered_search(query: str, filters: dict, top_k: int = 5) -> list[Document]:
    query_embedding = embed(query)
    results = vector_db.search(
        query_embedding,
        top_k=top_k,
        filter=filters  # {"tenant_id": "abc", "created_after": "2025-01-01"}
    )
    return results
```

### Pattern 3: Hybrid Search (Dense + Sparse)

```python
def hybrid_search(query: str, alpha: float = 0.5, top_k: int = 5) -> list[Document]:
    # Dense (semantic)
    dense_embedding = embed(query)
    dense_results = vector_db.search(dense_embedding, top_k=top_k * 2)
    
    # Sparse (keyword)
    sparse_results = bm25_search(query, top_k=top_k * 2)
    
    # Combine with reciprocal rank fusion
    combined = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        weights=[alpha, 1 - alpha]
    )
    
    return combined[:top_k]
```

Some databases (Weaviate, Qdrant, Pinecone) support hybrid search natively:

```python
# Weaviate native hybrid
results = client.query.get("Document", ["text"]).with_hybrid(
    query=query,
    alpha=0.5  # 0 = BM25 only, 1 = vector only
).with_limit(5).do()
```

### Pattern 4: Multi-Vector Query

For parent-child or multi-aspect retrieval:

```python
def multi_vector_search(queries: list[str], top_k: int = 5) -> list[Document]:
    all_results = []
    
    for query in queries:
        embedding = embed(query)
        results = vector_db.search(embedding, top_k=top_k)
        all_results.extend(results)
    
    # Dedupe and rerank
    unique = dedupe_by_id(all_results)
    reranked = rerank(queries[0], unique)  # Use primary query for reranking
    
    return reranked[:top_k]
```

---

## Production Operations

### Capacity Planning

```python
def estimate_resources(
    num_vectors: int,
    dimensions: int,
    metadata_size_bytes: int = 500
) -> dict:
    # Vector storage
    vector_size = dimensions * 4  # float32
    total_vector_storage = num_vectors * vector_size
    
    # Index overhead (HNSW ~1.5x)
    index_overhead = total_vector_storage * 1.5
    
    # Metadata
    metadata_storage = num_vectors * metadata_size_bytes
    
    # Total
    total_gb = (total_vector_storage + index_overhead + metadata_storage) / 1e9
    
    # QPS estimate (rough)
    qps_per_gb = 50  # depends heavily on config
    estimated_qps = total_gb * qps_per_gb
    
    return {
        "storage_gb": total_gb,
        "estimated_qps": estimated_qps,
        "recommended_replicas": max(1, int(total_gb / 50))  # ~50GB per replica
    }
```

### Index Maintenance

```python
class VectorDBMaintenance:
    def __init__(self, client):
        self.client = client
    
    def add_documents(self, documents: list[Document]):
        """Upsert documents with batching."""
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = embed_batch([d.text for d in batch])
            
            self.client.upsert([
                {
                    "id": doc.id,
                    "vector": embedding,
                    "payload": doc.metadata
                }
                for doc, embedding in zip(batch, embeddings)
            ])
    
    def delete_documents(self, doc_ids: list[str]):
        """Delete by document ID."""
        self.client.delete(ids=doc_ids)
    
    def update_metadata(self, doc_id: str, metadata: dict):
        """Update metadata without re-embedding."""
        self.client.set_payload(
            collection_name="documents",
            payload=metadata,
            points=[doc_id]
        )
```

### High Availability

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Replica 1   │ │  Replica 2   │ │  Replica 3   │
    │   (Read)     │ │   (Read)     │ │   (Primary)  │
    └──────────────┘ └──────────────┘ └──────────────┘
                                            │
                                      (Replication)
                                            │
                                      ┌─────▼─────┐
                                      │  Storage  │
                                      └───────────┘
```

**Key patterns:**
- Leader-follower for writes
- Read replicas for query scaling
- Async replication for HA

### Monitoring

```python
VECTOR_DB_METRICS = [
    "query_latency_p50",
    "query_latency_p99",
    "queries_per_second",
    "index_size_gb",
    "vector_count",
    "filter_latency",
    "upsert_latency",
    "cache_hit_rate"
]

def alert_rules():
    return {
        "query_latency_p99_high": {
            "condition": "query_latency_p99 > 500ms",
            "severity": "warning"
        },
        "query_latency_p99_critical": {
            "condition": "query_latency_p99 > 2000ms",
            "severity": "critical"
        },
        "low_recall": {
            "condition": "bench_recall < 0.90",
            "severity": "warning"
        }
    }
```

---

## Cost Analysis

### Managed Service Pricing (December 2025, verify current)

| Provider | Model | Example: 10M vectors, 1536 dims |
|----------|-------|--------------------------------|
| Pinecone | Pod-based or Serverless | ~$70-150/month serverless |
| Qdrant Cloud | Per GB | ~$50/month (20GB) |
| Weaviate Cloud | Per dimensions | ~$100/month |
| Zilliz (Milvus) | Per CU | ~$75/month |

### Self-Hosted Costs

```python
def estimate_self_hosted_cost(
    vectors: int,
    dimensions: int,
    cloud: str = "aws"
) -> dict:
    storage_gb = (vectors * dimensions * 4 * 2.5) / 1e9  # 2.5x for index
    
    # Instance sizing
    if storage_gb < 50:
        instance = "r6g.large"  # 16 GB RAM, ~$60/month
    elif storage_gb < 200:
        instance = "r6g.xlarge"  # 32 GB RAM, ~$120/month
    else:
        instance = "r6g.2xlarge"  # 64 GB RAM, ~$240/month
    
    return {
        "storage_gb": storage_gb,
        "instance": instance,
        "monthly_compute": instance_pricing[instance],
        "monthly_storage": storage_gb * 0.10,  # EBS
        "total_monthly": instance_pricing[instance] + storage_gb * 0.10
    }
```

### Decision: Managed vs Self-Hosted

| Factor | Managed | Self-Hosted |
|--------|---------|-------------|
| Ops overhead | Low | High |
| Cost at small scale | Higher | Lower |
| Cost at large scale | Variable | Often lower |
| Control | Less | Full |
| Compliance | Depends | Full control |
| Vendor lock-in | Yes | No (if open source) |

---

## Selection Framework

### Decision Tree

```
Need < 100K vectors?
├── Yes → pgvector (if already using PostgreSQL)
│         └── Chroma (for prototyping)
│
└── No → Need managed service?
         ├── Yes → Cloud-first?
         │         ├── Yes → Pinecone (easiest)
         │         └── No → Qdrant Cloud or Zilliz
         │
         └── No → Need enterprise features?
                  ├── Yes → Milvus on Kubernetes
                  └── No → Qdrant or Weaviate self-hosted
```

### Evaluation Criteria

| Criterion | Weight | Questions to Ask |
|-----------|--------|------------------|
| Scale | High | How many vectors now? In 1 year? |
| Latency | High | What are p99 requirements? |
| Ops capacity | High | Can we operate this? |
| Cost | Medium | Budget constraints? |
| Features | Medium | Hybrid search? Multimodal? |
| Lock-in risk | Low-Medium | Open source preferred? |

### Proof of Concept Checklist

Before committing to a vector database:

- [ ] Load representative data volume
- [ ] Benchmark query latency at target QPS
- [ ] Test metadata filtering performance
- [ ] Verify update/delete performance
- [ ] Test failure recovery
- [ ] Evaluate monitoring and observability
- [ ] Calculate total cost of ownership

---

## Interview Questions

### Q: How would you choose between Pinecone and a self-hosted solution?

**Strong answer:**
Decision depends on several factors:

**Choose Pinecone when:**
- Team lacks ops capacity for stateful infrastructure
- Need to move quickly (days not weeks)
- Scale is moderate (under 100M vectors)
- Budget allows managed service premium
- Compliance allows cloud-vendor dependency

**Choose self-hosted (Qdrant, Milvus) when:**
- Have Kubernetes and ops expertise
- Cost sensitivity at scale
- Need full control over data
- Specific compliance requirements
- Want to avoid vendor lock-in

For most startups, I would start with Pinecone or Qdrant Cloud for velocity, then evaluate migration if costs become prohibitive at scale. The switching cost is moderate since vector DBs have similar APIs.

### Q: Explain how HNSW works and when you would not use it.

**Strong answer:**
HNSW builds a hierarchical graph of vectors:

**How it works:**
1. Insert vectors as nodes in a multi-layer graph
2. Higher layers have fewer nodes, larger jumps
3. Search: start at top layer, greedily navigate to nearest neighbor
4. Descend layers until bottom (all vectors)

**Why it is good:**
- O(log n) query complexity
- No training required
- Supports real-time updates
- Excellent recall/latency tradeoff

**When not to use:**
- Very small datasets (<10K): brute force is fine
- Extremely memory constrained: HNSW uses 1.5-2x vector size for graph
- Need exact search: HNSW is approximate
- Heavy update workload with tight latency: updates can cause temporary degradation

Alternatives:
- IVF-PQ for memory constraints
- Flat index for exact search
- LSH for very high-dimensional sparse vectors

### Q: How do you handle multi-tenancy in a vector database?

**Strong answer:**
Three main approaches:

**1. Metadata filtering (most common):**
```python
results = db.search(
    vector=query,
    filter={"tenant_id": current_tenant}
)
```
- Pros: Simple, single index
- Cons: All tenants share resources, potential for bugs exposing data

**2. Collection per tenant:**
```python
results = db.collection(f"tenant_{tenant_id}").search(vector=query)
```
- Pros: Strong isolation, per-tenant scaling
- Cons: Many collections, operational overhead

**3. Namespace per tenant (Pinecone):**
```python
results = index.query(vector=query, namespace=tenant_id)
```
- Pros: Isolation within single index
- Cons: Vendor-specific

**I would choose:**
- Metadata filtering for most cases (simple, cost-effective)
- Separate collections for high-security requirements
- Never post-filter (retrieve all, filter after) due to leakage risk

---

## References

- Malkov and Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (HNSW, 2018)
- Pinecone Documentation: https://docs.pinecone.io/
- Qdrant Documentation: https://qdrant.tech/documentation/
- Weaviate Documentation: https://weaviate.io/developers/weaviate
- Milvus Documentation: https://milvus.io/docs
- pgvector: https://github.com/pgvector/pgvector

---

*Previous: [Chunking Strategies](02-chunking-strategies.md) | Next: [Hybrid Search](04-hybrid-search.md)*

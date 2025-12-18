# Hybrid Search (Dec 2025)

Hybrid search is the combination of **Dense (Semantic)** and **Sparse (Keyword)** retrieval to overcome the weaknesses of each. In late 2025, hybrid search is the baseline for all production RAG systems.

## Table of Contents

- [The Retrieval Gap (Semantic vs. Lexical)](#gap)
- [Reciprocal Rank Fusion (RRF)](#rrf)
- [Weighted Score Fusion](#weighted)
- [Learned Sparse Embeddings (SPLADE)](#splade)
- [Implementation Architectures](#architectures)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Retrieval Gap

| Aspect | Dense (Vector) | Sparse (Keyword/BM25) |
|--------|----------------|-----------------------|
| **Best for** | Concepts, Synonyms, Paraphrases | Product IDs, Acronyms, Rare Terms |
| **Weakness** | Misses exact name/ID matches | Misses conceptual meaning |
| **Logic** | "Show me luxury cars" | "Find part #XJ-102" |

**2025 Nuance**: "Dense-only" retrieval fails on technical documentation where specific version numbers and function names carry 90% of the information value.

---

## Reciprocal Rank Fusion (RRF)

RRF is the gold standard for combining results from two different search engines (e.g., Pinecone + Elasticsearch).

- **How it works**: It doesn't look at the *score* (which is incomparable across engines). It looks at the **rank**.
- **Formula**: `Score = sum(1 / (k + rank))`, where `k` is a smoothing constant (usually 60).
- **Benefit**: It prevents a single engine from "dominating" the results just because it has a high numerical score.

---

## Weighted Score Fusion

For engines that support native hybrid (like Qdrant or Weaviate), you use a weighted sum of normalized scores.

`Final_Score = (alpha * dense_score) + ((1 - alpha) * sparse_score)`

**2025 Best Practice**: 
- Use **alpha = 0.5** for general search.
- Use **alpha = 0.3** (keyword heavy) for technical docs and code.
- Use **alpha = 0.7** (semantic heavy) for chat and creative exploration.

---

## Learned Sparse Embeddings (SPLADE)

In late 2025, we have moved beyond BM25 (simple word frequency) to **Learned Sparse Embeddings**.

- **Technique**: Models like **SPLADE v3** predict "importance weights" for every word in the dictionary.
- **Why?**: It can "expand" queries. If you search for "CPU," it might automatically add a small weight to the term "processor," even if "processor" isn't in your query.
- **Benefit**: It combines the exact-match power of sparse search with the conceptual power of dense search in a single storage format.

---

## Implementation Architectures

### 1. Parallel (Two-Engine)
- **Flow**: Query -> [Vector DB] AND [Elasticsearch] -> RRF -> Combined Top-K.
- **Pros**: Can use the best-in-class for each (e.g., Pinecone + Algolia).
- **Cons**: High latency (must wait for the slower engine).

### 2. Native Multi-Vector (Single DB)
- **Flow**: Qdrant/Milvus stores both dense and sparse vectors in the same record.
- **Pros**: Lower latency, single database to manage.
- **Cons**: Less flexibility in scaling keyword vs. vector infra.

---

## Interview Questions

### Q: Why is Reciprocal Rank Fusion (RRF) safer than "Simple Score Addition"?

**Strong answer:**
Simple score addition is dangerous because vector scores (e.g., Cosine Similarity: 0.0 to 1.0) and keyword scores (e.g., BM25: 0 to infinity) use completely different scales. An extremely high BM25 score for a lucky keyword match could "drown out" 10 highly relevant semantic matches. RRF ignores the absolute scores and only cares about the relative order (rank). This makes it mathematically robust to outliers and "score-drift" in different retrieval engines.

### Q: When would you choose SPLADE over the standard BM25 + Dense Hybrid approach?

**Strong answer:**
I would choose SPLADE when I want to simplify my infrastructure. SPLADE produces a sparse vector that can be stored in many modern vector databases (like Milvus or Qdrant) alongside the dense vector. This allows the database to perform "Hybrid search" in a single pass without needing a separate Elasticsearch or BM25 index. However, I would stick to BM25 if my dataset has extremely rare, non-linguistic tokens (like unique serial numbers) that a neural model might not have seen during training.

---

## References
- Cormack et al. "Reciprocal Rank Fusion" (2009/2024 update)
- Formal et al. "SPLADE: Sparse Lexical and Expansion Model" (2021/2025)
- Qdrant. "Hybrid Search in Production" (2025)

---

*Next: [Reranking Strategies](06-reranking-strategies.md)*

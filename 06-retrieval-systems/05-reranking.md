# Reranking

Reranking is a second-stage retrieval step that re-scores candidates with a more accurate (but slower) model. This chapter covers reranking approaches, models, and implementation patterns.

## Table of Contents

- [Why Reranking](#why-reranking)
- [Reranking Architectures](#reranking-architectures)
- [Reranking Models](#reranking-models)
- [Implementation Patterns](#implementation-patterns)
- [When to Rerank](#when-to-rerank)
- [LLM-Based Reranking](#llm-based-reranking)
- [Production Considerations](#production-considerations)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Reranking

First-stage retrieval (embedding search) optimizes for recall. Reranking optimizes for precision.

### The Quality Gap

| Stage | Model | Speed | Quality |
|-------|-------|-------|---------|
| Embedding retrieval | Bi-encoder | Fast (ms) | Good |
| Reranking | Cross-encoder | Slow (100ms+) | Better |

**Why the gap exists:**
- Bi-encoders embed query and document independently
- Cross-encoders jointly process query and document
- Joint processing captures interactions bi-encoders miss

### Example

```
Query: "How to configure CUDA memory"

Document 1: "Configure GPU memory using CUDA_VISIBLE_DEVICES..."
Document 2: "Memory management in CUDA applications..."
Document 3: "Configure RAM allocation for machine learning..."

Bi-encoder scores (cosine similarity):
- Doc 1: 0.72
- Doc 2: 0.75  <-- Ranked first (wrong)
- Doc 3: 0.71

Cross-encoder scores (relevance):
- Doc 1: 0.91  <-- Ranked first (correct)
- Doc 2: 0.67
- Doc 3: 0.42
```

The cross-encoder sees that "CUDA memory" in the query relates to "GPU memory...CUDA" in Doc 1.

---

## Reranking Architectures

### Bi-Encoder vs Cross-Encoder

**Bi-Encoder (First Stage):**
```
Query ──► Encoder ──► Query Embedding ─┐
                                       ├─► Similarity
Document ──► Encoder ──► Doc Embedding ┘
```
- O(1) per document (embeddings pre-computed)
- Cannot see query-document interactions

**Cross-Encoder (Reranking):**
```
[Query, Document] ──► Encoder ──► Relevance Score
```
- O(n) per query (process each candidate)
- Sees full query-document context

### Two-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│  STAGE 1: Retrieval (Bi-Encoder)                               │
│                                                                │
│  Query ──► Embed ──► Top-K candidates (K=100)                  │
│                                                                │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 2: Reranking (Cross-Encoder)                            │
│                                                                │
│  For each candidate:                                           │
│    score = reranker([query, candidate])                        │
│                                                                │
│  Return Top-N by reranker score (N=5-10)                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Multi-Stage Pipeline

For very large corpora:

```
Stage 1: Sparse (BM25)      → Top 1000
Stage 2: Dense (Bi-encoder) → Top 100
Stage 3: Cross-encoder      → Top 10
```

Each stage trades speed for accuracy.

---

## Reranking Models

### Cross-Encoder Models

| Model | Size | Languages | Quality |
|-------|------|-----------|---------|
| ms-marco-MiniLM-L-6 | 22M | English | Good |
| bge-reranker-base | 278M | English | Very good |
| bge-reranker-v2-m3 | 568M | Multilingual | Excellent |
| Cohere Rerank v3 | API | Multilingual | Excellent |

### Using Cross-Encoders

```python
from sentence_transformers import CrossEncoder

# Load model
reranker = CrossEncoder('BAAI/bge-reranker-base')

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    # Create pairs
    pairs = [[query, doc] for doc in documents]
    
    # Score all pairs
    scores = reranker.predict(pairs)
    
    # Sort by score
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return scored_docs[:top_k]
```

### Cohere Rerank

```python
import cohere

co = cohere.Client(api_key="...")

def cohere_rerank(
    query: str,
    documents: list[str],
    top_k: int = 5
) -> list[dict]:
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k,
        return_documents=True
    )
    
    return [
        {
            "text": result.document.text,
            "score": result.relevance_score,
            "index": result.index
        }
        for result in response.results
    ]
```

### Model Selection Guide

| Use Case | Recommended Model | Notes |
|----------|-------------------|-------|
| English, self-hosted | bge-reranker-base | Good balance |
| Multilingual | bge-reranker-v2-m3 | Best open source |
| Low latency | MiniLM-L-6 | 4x faster |
| Highest quality | Cohere Rerank v3 | API, costly at scale |
| Large batches | Jina Reranker | Good throughput |

---

## Implementation Patterns

### Pattern 1: Basic Reranking

```python
class RerankedRetriever:
    def __init__(
        self,
        vector_db,
        embedding_model,
        reranker,
        retrieval_k: int = 50,
        rerank_k: int = 5
    ):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
    
    def search(self, query: str) -> list[Document]:
        # Stage 1: Retrieve candidates
        query_embedding = self.embedding_model.encode(query)
        candidates = self.vector_db.search(
            query_embedding,
            top_k=self.retrieval_k
        )
        
        # Stage 2: Rerank
        pairs = [[query, c.text] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Combine and sort
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:self.rerank_k]
```

### Pattern 2: Batched Reranking

```python
def batch_rerank(
    queries: list[str],
    candidates_per_query: list[list[str]],
    reranker,
    batch_size: int = 32
) -> list[list[tuple[str, float]]]:
    # Flatten all pairs
    all_pairs = []
    pair_mapping = []  # (query_idx, doc_idx)
    
    for q_idx, (query, candidates) in enumerate(zip(queries, candidates_per_query)):
        for d_idx, doc in enumerate(candidates):
            all_pairs.append([query, doc])
            pair_mapping.append((q_idx, d_idx))
    
    # Batch score
    all_scores = []
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        scores = reranker.predict(batch)
        all_scores.extend(scores)
    
    # Reconstruct per-query results
    results = [[] for _ in queries]
    for (q_idx, d_idx), score in zip(pair_mapping, all_scores):
        results[q_idx].append((candidates_per_query[q_idx][d_idx], score))
    
    # Sort each query's results
    for i in range(len(results)):
        results[i].sort(key=lambda x: x[1], reverse=True)
    
    return results
```

### Pattern 3: Async Reranking

```python
import asyncio

class AsyncReranker:
    def __init__(self, reranker, max_concurrent: int = 5):
        self.reranker = reranker
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def rerank_async(
        self,
        query: str,
        documents: list[str]
    ) -> list[tuple[str, float]]:
        async with self.semaphore:
            # Run reranking in thread pool
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self.reranker.predict([[query, doc] for doc in documents])
            )
            return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

---

## When to Rerank

### Cost-Benefit Analysis

| Factor | Without Reranking | With Reranking |
|--------|-------------------|----------------|
| Latency | 50-100ms | 150-300ms |
| Quality (NDCG) | 0.65 | 0.78 |
| Complexity | Simple | Moderate |
| Cost | Baseline | +API cost or +compute |

### Decision Framework

**Always rerank when:**
- Quality is critical (customer-facing, high-stakes)
- Retrieved candidates have similar scores
- Query is complex or multi-part
- Budget allows for latency increase

**Skip reranking when:**
- Latency budget is very tight (<100ms total)
- Retrieved candidates are clearly ranked
- Simple queries (single term lookups)
- Cost constrained at scale

### Optimal Candidate Count

How many candidates to retrieve before reranking:

```python
def optimize_candidate_count(test_set, retriever, reranker):
    """Find optimal retrieval_k for reranking."""
    results = {}
    
    for retrieval_k in [10, 20, 50, 100, 200]:
        ndcg_scores = []
        latencies = []
        
        for query, relevant_docs in test_set:
            start = time.time()
            
            # Retrieve
            candidates = retriever.search(query, top_k=retrieval_k)
            
            # Rerank to top 5
            reranked = reranker.rerank(query, candidates, top_k=5)
            
            latency = time.time() - start
            latencies.append(latency)
            
            ndcg = compute_ndcg(reranked, relevant_docs)
            ndcg_scores.append(ndcg)
        
        results[retrieval_k] = {
            "ndcg": mean(ndcg_scores),
            "latency_p99": percentile(latencies, 99)
        }
    
    return results

# Typical findings:
# K=20:  NDCG 0.72, latency 120ms
# K=50:  NDCG 0.76, latency 180ms  <-- Often sweet spot
# K=100: NDCG 0.77, latency 280ms  <-- Diminishing returns
```

---

## LLM-Based Reranking

### Using LLMs as Rerankers

LLMs can score relevance but are expensive:

```python
def llm_rerank(
    query: str,
    documents: list[str],
    model: str = "gpt-4o-mini"
) -> list[tuple[str, float]]:
    prompt = f"""Rate the relevance of each document to the query.
Query: {query}

Documents:
{format_documents(documents)}

For each document, output a relevance score from 0-10.
Format: DOC_NUM: SCORE
"""
    
    response = llm.generate(prompt)
    scores = parse_scores(response)
    
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

**Pros:**
- Can handle complex relevance judgments
- Understands nuance and context
- No separate model to maintain

**Cons:**
- Expensive at scale
- Slower than cross-encoders
- Non-deterministic

### Listwise vs Pointwise LLM Reranking

**Pointwise:** Score each document independently
```
For document: [doc text]
Query: [query]
Rate relevance 0-10: _
```

**Listwise:** Rank all documents together
```
Query: [query]
Rank these documents by relevance:
A: [doc1]
B: [doc2]
C: [doc3]
Output order: _
```

**Listwise is often better** because the LLM can compare documents directly.

### Sliding Window for Many Documents

```python
def sliding_window_rerank(
    query: str,
    documents: list[str],
    window_size: int = 10,
    step: int = 5
) -> list[str]:
    """Rerank many documents with LLM using sliding window."""
    ranked = list(range(len(documents)))
    
    for start in range(0, len(documents), step):
        window = ranked[start:start + window_size]
        
        # LLM ranks this window
        window_docs = [documents[i] for i in window]
        window_order = llm_listwise_rank(query, window_docs)
        
        # Update rankings
        for new_pos, old_idx in enumerate(window_order):
            ranked[start + new_pos] = window[old_idx]
    
    return [documents[i] for i in ranked]
```

---

## Production Considerations

### Latency Optimization

```python
class OptimizedReranker:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device)
        # Enable optimizations
        self.model.model.half()  # FP16
        
    def rerank(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        with torch.inference_mode():
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False
            )
        return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
```

**Optimization techniques:**
- FP16 inference: 2x speedup
- Batching: Amortize overhead
- ONNX export: 1.5-2x speedup
- TensorRT: 2-3x speedup (NVIDIA)
- Model distillation: 4x speedup with quality tradeoff

### Caching Reranker Results

```python
class CachedReranker:
    def __init__(self, reranker, cache_ttl: int = 3600):
        self.reranker = reranker
        self.cache = TTLCache(maxsize=10000, ttl=cache_ttl)
    
    def rerank(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        # Cache key includes query and doc hashes
        key = self._make_key(query, documents)
        
        if key in self.cache:
            return self.cache[key]
        
        result = self.reranker.rerank(query, documents)
        self.cache[key] = result
        return result
    
    def _make_key(self, query: str, documents: list[str]) -> str:
        doc_hash = hashlib.sha256(
            "".join(sorted(documents)).encode()
        ).hexdigest()[:16]
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{query_hash}:{doc_hash}"
```

### Fallback Strategy

```python
def rerank_with_fallback(
    query: str,
    candidates: list[Document],
    primary_reranker,
    timeout: float = 2.0
) -> list[Document]:
    try:
        # Try reranking with timeout
        result = timeout_call(
            primary_reranker.rerank,
            args=(query, candidates),
            timeout=timeout
        )
        return result
    except TimeoutError:
        # Fallback: return original order
        logger.warning("Reranker timeout, using original order")
        return candidates
    except Exception as e:
        logger.error(f"Reranker error: {e}")
        return candidates
```

---

## Interview Questions

### Q: Explain the difference between bi-encoders and cross-encoders.

**Strong answer:**
**Bi-encoders** embed query and document independently:
```
query_vec = encode(query)
doc_vec = encode(document)
score = similarity(query_vec, doc_vec)
```
- Pros: Documents pre-computed, O(1) retrieval
- Cons: No query-document interaction

**Cross-encoders** process query and document together:
```
score = encode([query, document])
```
- Pros: Sees interactions, higher quality
- Cons: O(n) per query, cannot pre-compute

**In practice:** Use bi-encoder for first-stage retrieval (speed), cross-encoder for reranking (quality). This gives the best of both.

### Q: How do you decide how many candidates to rerank?

**Strong answer:**
Tradeoff between quality and latency:

**Factors:**
- Reranker latency per document
- Total latency budget
- Quality improvement curve (usually diminishing returns)
- First-stage retrieval quality

**Process:**
1. Benchmark reranker latency per document
2. Calculate max candidates within latency budget
3. Test quality at different K values
4. Find elbow point (quality vs latency)

**Typical findings:**
- K=20-50 is often optimal
- Beyond K=100, quality gains are minimal
- Adjust based on first-stage retrieval quality

For a 200ms reranking budget with 4ms per document, I would rerank ~50 candidates.

### Q: When would you use LLM-based reranking?

**Strong answer:**
LLM reranking makes sense when:

1. **Complex relevance judgments:** Query requires understanding nuance, context, or multi-hop reasoning
2. **Low volume:** Cannot justify training/hosting a cross-encoder
3. **Highest quality required:** Legal, medical, safety-critical
4. **Already using LLM in pipeline:** Marginal cost lower

**Cautions:**
- Expensive at scale (10-100x cross-encoder)
- Slower (1-3s vs 100ms)
- Non-deterministic
- May require careful prompt engineering

**Production pattern:** Use cross-encoder normally, LLM for fallback on low-confidence reranking scores.

---

## References

- Nogueira and Cho. "Passage Re-ranking with BERT" (2019)
- BAAI BGE Reranker: https://huggingface.co/BAAI/bge-reranker-base
- Cohere Rerank: https://docs.cohere.com/docs/rerank
- Sun et al. "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (2023)

---

*Previous: [Hybrid Search](04-hybrid-search.md) | Next: [Query Processing](06-query-processing.md)*

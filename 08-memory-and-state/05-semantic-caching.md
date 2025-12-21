# Semantic Caching (Dec 2025)

In late 2025, caching has evolved from exact string matching to **Semantic Matching**. Semantic caching reduces costs by **30-70%** and reduces latency from seconds to milliseconds by reusing completions for "equivalent" queries.

## Table of Contents

- [Exact Cache vs. Semantic Cache](#vs)
- [The Semantic Matching Pipeline](#pipeline)
- [RedisVL and GPTCache](#tech-stack)
- [Evaluation: Hit Rate vs. Hallucinated Drift](#eval)
- [Multimodal Semantic Caching](#multimodal)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Exact Cache vs. Semantic Cache

| Feature | Exact Cache (Redis/Memcached) | Semantic Cache (RedisVL/Qdrant) |
|---------|-------------------------------|---------------------------------|
| **Key** | Hashed query string | Query embedding vector |
| **Match**| 100% string identity | Cosine Similarity > Threshold |
| **Efficiency**| Low (Minor typos break cache) | High (Understands intent) |
| **Risk** | Zero | Semantic Drift (Returning wrong answer) |

---

## The Semantic Matching Pipeline

1. **Embed**: The incoming query is converted into a vector (e.g., using `text-embedding-3-small`).
2. **Search**: Search the cache for the nearest neighbor.
3. **Threshold Check**: If `distance < 0.05` (very similar), return the cached result.
4. **LLM Verification (2025 nuance)**: For high-stakes queries, a tiny "Verifier Model" (e.g., GPT-4o-mini) checks if the cached response actually answers the new query.
5. **Update**: If no hit, call the LLM and store the new result in the vector cache.

---

## RedisVL and GPTCache

Late 2025's standard stack:
- **RedisVL**: Provides low-latency vector search directly within a Redis instance.
- **Hybrid Caching**: Using Redis for both metadata (keys) and vector payloads.
- **TTL**: Semantic caches should have a TTL (Time-To-Live). In 2025, we use **Dynamic TTL**: popular answers live longer, while "stale" information is evicted regularly.

---

## Multimodal Semantic Caching

With the rise of Gemini 3 and GPT-5.2 Native Omni, we now cache **Image and Audio queries**.
- **Visual Similarity**: Caching the description of an image if a semantically similar image was processed before.
- **Audio Fingerprinting**: Caging transcripts for similar voice commands.

---

## Interview Questions

### Q: What is "Semantic Drift" in caching, and how do you prevent it?

**Strong answer:**
Semantic Drift occurs when the similarity threshold is too loose (e.g., 0.8 instead of 0.95). A query like *"How do I fix my car?"* might match a cached response for *"How do I wash my car?"*. To prevent this, we use **Multi-Stage Validation**: 1) Vector similarity check, 2) **Entity-Match check** (ensures both queries involve "Car" and the same "Verb"), and 3) **Threshold Tightening**: for technical or medical queries, we require $>0.98$ similarity to return a cached result.

### Q: Why is a Semantic Cache sometimes *more* expensive than a raw LLM call at low volume?

**Strong answer:**
Because a semantic cache requires its own **Embedding API call** and **Vector Search query**. If the embedding model costs $0.02 and the search takes 100ms, and your primary LLM call is only $0.05 and takes 500ms, the relative savings are small. Semantic caching only becomes a significant win at **High Scale** (millions of requests) where the cache hit rate is high enough to offset the "Embedding Tax" and drastically reduce aggregate latency.

---

## References
- Redis. "RedisVL: Python Client for Redis Vector Library" (2025)
- Akiba et al. "GPTCache: A Library for Creating Semantic Cache" (2024/2025)
- Google Cloud. "Generative AI Caching Patterns" (2025)

---

*Next: [State Management Patterns](06-state-management-patterns.md)*

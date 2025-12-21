# Case Study: AI-Powered Recommendation Engine

## The Problem

A streaming platform with **50 million users** needs to build a recommendation system that combines collaborative filtering with LLM-generated explanations: "Because you enjoyed Inception, you might like Tenet for its mind-bending time mechanics."

**Constraints given in the interview:**
- Real-time recommendations (under 200ms p95)
- Must explain why each recommendation was made
- Cold-start handling for new users
- Privacy: cannot leak viewing history between users
- Daily active: 5M users, each viewing 10+ recommendation sets

---

## The Interview Question

> "Design a system that recommends movies AND explains the recommendation in natural language, at scale."

---

## Solution Architecture

```mermaid
flowchart TB
    subgraph Offline["Offline Pipeline (Daily)"]
        HISTORY[(Watch History)] --> EMBED[User Embedding<br/>Matrix Factorization]
        CATALOG[(Content Catalog)] --> CONTENT_EMBED[Content Embeddings]
        EMBED --> CANDIDATES[Candidate Generation<br/>ANN Index]
    end

    subgraph Online["Online Serving (Real-Time)"]
        USER[User Request] --> FETCH[Fetch User Embedding]
        FETCH --> ANN[ANN Search<br/>Top 100 Candidates]
        ANN --> RERANK[Reranker<br/>Cross-Encoder]
        RERANK --> TOP10[Top 10 Results]
    end

    subgraph Explain["Explanation Generation"]
        TOP10 --> BATCH[Batch Explanation Request]
        BATCH --> LLM[GPT-4o-mini<br/>Cached Explanations]
        LLM --> RESPONSE[Recommendations + Reasons]
    end
```

---

## Key Design Decisions

### 1. Why Not Just Use LLM for Everything?

**Answer:** Scale economics. Calling an LLM for 50M users × 10 recommendation sets/day = 500M LLM calls daily. At $0.001 per call, that is $500K/day. Instead:

| Component | Role | Cost per User/Day |
|-----------|------|-------------------|
| Embedding lookup | Fetch precomputed vector | $0.00001 |
| ANN search | Find candidates | $0.0001 |
| Cross-encoder rerank | Score top 100 | $0.001 |
| LLM explanation | Natural language | $0.005 |
| **Total** | | **$0.006** |

The LLM is only used for the final explanation, not the ranking itself.

### 2. Explanation Caching

**Answer:** Most explanations can be cached. "Because you watched Inception" applies to thousands of users. We cache explanations at the (content_pair, reason_type) level:

```python
cache_key = f"{source_movie}:{target_movie}:{reason_type}"
# Example: "inception:tenet:time_mechanics"

explanation = cache.get(cache_key)
if not explanation:
    explanation = generate_explanation(source_movie, target_movie, reason_type)
    cache.set(cache_key, explanation, ttl=86400)
```

Cache hit rate: 85%+ after warmup.

### 3. Cold-Start Handling

**Answer:** New users have no history for collaborative filtering. We use a **Hybrid Approach**:

```mermaid
flowchart LR
    NEW_USER[New User] --> CHECK{Has History?}
    CHECK -->|No| CONTENT[Content-Based<br/>Preferences Survey]
    CHECK -->|Yes, <10 items| HYBRID[Hybrid:<br/>Content + Collaborative]
    CHECK -->|Yes, >10 items| COLLAB[Full Collaborative<br/>Filtering]
    
    CONTENT --> RECS[Recommendations]
    HYBRID --> RECS
    COLLAB --> RECS
```

---

## The Personalized Explanation Challenge

The explanation must feel personal, not generic:

**Bad:** "Tenet is a popular thriller."
**Good:** "Because you enjoyed Inception's mind-bending plot, Tenet offers similar time-manipulation puzzles from the same director."

We achieve this by including user context in the prompt:

```python
prompt = f"""
Generate a 1-sentence explanation for why this user would enjoy {target_movie}.

User context:
- Recently watched: {recent_movies}
- Preferred genres: {genres}
- Dislikes: {dislikes}

Source movie that triggered this recommendation: {source_movie}
Reason category: {reason_type}

Explanation:
"""
```

---

## Latency Budget

| Stage | Target | Actual p95 |
|-------|--------|------------|
| User embedding lookup | 5ms | 3ms |
| ANN search (top 100) | 20ms | 15ms |
| Cross-encoder rerank | 50ms | 45ms |
| LLM explanation (cached) | 10ms | 8ms |
| LLM explanation (miss) | 500ms | 450ms |
| **Total (cache hit)** | **85ms** | **71ms** |
| **Total (cache miss)** | **575ms** | **513ms** |

To meet 200ms p95, we ensure 95%+ cache hit rate for explanations and generate explanations asynchronously for new content pairs.

---

## Interview Follow-Up Questions

**Q: How do you prevent the LLM from hallucinating facts about movies?**

A: The LLM receives a structured fact sheet for each movie (director, cast, themes, awards) as context. It can only use information from this sheet. We also have a post-generation validator that checks claims against our catalog metadata.

**Q: What if a user's taste changes rapidly?**

A: We use a **recency-weighted embedding update**. Recent watches are weighted 3x more than older ones. For real-time responsiveness, we maintain a "session embedding" that captures current-session behavior and blends it with the historical embedding.

**Q: How do you A/B test recommendation algorithms?**

A: We hash user_id to consistently assign users to experiment buckets. Each bucket can have different candidate generation, ranking, or explanation strategies. We track engagement metrics (click-through, watch time, skip rate) per bucket.

---

## Key Takeaways for Interviews

1. **LLMs for explanation, not ranking**: use traditional ML for scale, LLMs for personalization
2. **Cache aggressively**: explanations for content pairs are reusable across users
3. **Cold-start is a spectrum**: new users → content-based; some history → hybrid; full history → collaborative
4. **Latency budgets require cache hit rate targets**: design the cache around your latency SLA

---

*Related chapters: [Semantic Caching](../08-memory-and-state/05-semantic-caching.md), [Cost Optimization](../04-inference-optimization/07-cost-optimization-playbook.md)*

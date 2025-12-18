# Reranking Strategies (Dec 2025)

Reranking is the second stage of retrieval that re-scores a small set of candidates (Top 50-100) using a high-precision model. In late 2025, reranking is the bridge between "efficient search" and "perfect grounding."

## Table of Contents

- [The Bi-Encoder vs. Cross-Encoder Gap](#architecture)
- [Cross-Encoder Reranking](#cross-encoder)
- [Listwise Reranking (LLM-as-Reranker)](#listwise)
- [SLM Distillation (The 2025 Performance Win)](#slm)
- [Inference Time Tradeoffs](#tradeoffs)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Architecture Gap

Production retrieval uses a two-stage funnel:

1. **Stage 1 (Bi-Encoder / Retrieval)**: 
   - Uses pre-computed vectors.
   - **Scale**: Search 1 Billion docs.
   - **Cost**: Low (ms).
2. **Stage 2 (Cross-Encoder / Reranking)**:
   - Processes query and document *together* in a single pass.
   - **Scale**: Search Top 100 docs.
   - **Cost**: High (10-100ms).

---

## Cross-Encoder Reranking

A Cross-Encoder doesn't just calculate a dot product. It looks at the textual interaction between every token in the query and every token in the document.

- **2025 Model**: **BGE-Reranker-v2-M3**.
- **The "Lost in the Middle" Fix**: Rerankers are trained to prioritize relevant information regardless of its position in the chunk, ensuring that "middle" data is scored correctly before being sent to the final LLM.

---

## Listwise Reranking (LLM-as-Reranker)

Instead of scoring each document 1-by-1 (**Pointwise**), we send the top 10 documents to a model and ask it to order them (**Listwise**).

```markdown
# Prompt
Rank these 10 documents by relevance to the query: "$QUERY".
Output only the indices in order (e.g., [4, 1, 9...]).
```

**2025 Status**: Frontier models (like o1-mini or Sonnet 3.7) are extremely good at this, but it adds 1-2s of latency. It is only used for high-stakes enterprise search (Legal, Medical).

---

## SLM Distillation (The 2025 Performance Win)

To solve the latency problem, we now use **Distilled Small Language Models (SLMs)** for reranking.

- **Process**: Take a giant model (GPT-5), have it rerank 1 million pairs, and use those labels to "distill" a tiny 0.1B parameter model.
- **Result**: You get 95% of the reranking quality of a giant model with the latency of a standard CPU lookup ( < 10ms).

---

## Inference Time Tradeoffs

| Stage | Retrieval (K) | Rerank (N) | Latency | Quality |
|-------|---------------|------------|---------|---------|
| **Naive** | 5 | 0 | 50ms | Low |
| **Standard** | 50 | 5 | 150ms | High |
| **Enterprise**| 200 | 20 | 500ms | Max |

**Key Rule**: If you have a budget of 200ms, spend 50ms on retrieval and 150ms on reranking. Reranking Top 50 results provides a much higher ROI than retrieving more chunks from the vector DB.

---

## Interview Questions

### Q: Why is a Cross-Encoder fundamentally more accurate than a Bi-Encoder?

**Strong answer:**
A Bi-Encoder creates a single, static vector representtion for a document *before* any query is known. This loses the specific relationship between different parts of the text. A Cross-Encoder takes both the query and the document as a single input pair and uses the **Attention Mechanism** to compare them. It can see how specific words in the query change the meaning of words in the document (late interaction), allowing for much more nuanced relevance scoring than a simple mathematical similarity of two fixed vectors.

### Q: How do you handle reranking for extremely long queries (e.g., a whole paragraph)?

**Strong answer:**
Long queries present a "Token Budget" problem for cross-encoders, which often have 512 or 1024 token limits. In 2025, we use **Sliding Window Reranking** or **Query Summarization**. Alternatively, we use specialized models like **Jina-Reranker-v2**, which can handle 8k+ tokens. We can also perform a "First-Pass Rerank" using a very fast, short-context model and then a "Second-Pass Rerank" for the top 5 candidates using a high-context LLM.

---

## References
- BAAI. "BGE Reranker: A High-Precision Reranking Model" (2024)
- Cohere. "Rerank v3: Multilingual Reranking for Enterprise" (2024)
- Nogueira et al. "Multi-Stage Document Ranking with BERT" (2019/2025 update)

---

*Next: [GraphRAG](07-graph-rag.md)*

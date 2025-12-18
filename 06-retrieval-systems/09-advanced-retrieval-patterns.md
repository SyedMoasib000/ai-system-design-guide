# Advanced Retrieval Patterns (Dec 2025)

Beyond the basics, production RAG systems use specialized patterns to handle complex query-document gaps. In late 2025, these patterns are the "Secret Sauce" of high-precision search.

## Table of Contents

- [Query Decomposition (Multi-Query)](#query-decomposition)
- [Hypothetical Document Embeddings (HyDE)](#hyde)
- [Contextual Retrieval (The Anthropic Pattern)](#contextual)
- [Iterative Document Enrichment](#enrichment)
- [In-Context Reranking (The 2025 Standard)](#reranking)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Query Decomposition (Multi-Query)

Complex user queries are often "Compound Queries."
- **User**: "Compare our Q3 vs Q4 revenue and explain the drop."
- **Decomposition**:
  1. "Q3 Revenue 2025"
  2. "Q4 Revenue 2025"
  3. "Reasons for Q4 revenue variance"
- **Implementation**: Use an LLM to generate these 3 sub-queries, search the DB for ALL of them, and aggregate the context.

---

## Hypothetical Document Embeddings (HyDE)

Queries are short; documents are long. This "Asymmetry" causes retrieval failure.
- **Pattern**: 
  1. Take the user query.
  2. Ask the LLM: "Write a 1-paragraph hypothetical answer to this."
  3. **Embed the hypothetical answer** instead of the query.
- **Why?**: The hypothetical answer is in the same "Vector neighborhood" as the real documents, leading to much higher recall.

---

## Contextual Retrieval (The Anthropic Pattern)

 standardized by Anthropic in late 2024, this pattern solves **Context Dilution**.

- **The Problem**: A chunk might say "It costs $200," but without the header, we don't know "It" is a "Widget-X."
- **The Pattern**: During ingestion, for every 300-token chunk, have an LLM write a 50-token context string (e.g., "This chunk is about the pricing for Widget-X in the North American market").
- **Benefit**: Increases retrieval precision by 30-50% for fragmented data.

---

## Iterative Document Enrichment

Instead of just storing the raw document, we store "Enriched" meta-data.
- **Summary**: Store a 1-paragraph summary of the document.
- **Q&A Generation**: Generate 5 questions this document answers and embed those *with* the document.
- **Status**: By late 2025, most high-end RAG systems embed **"Questions"** rather than **"Answers"** to match the user's query intent.

---

## In-Context Reranking (The 2025 Standard)

With 2M context windows, we use **Rank-by-Context**.
1. Retrieve Top 100 docs.
2. Put all 100 in the context window.
3. Ask the model: "Read these 100 docs and identify the 5 most relevant. Then, use those 5 to answer."
- **Win**: This utilizes the model's **Long Context Reasoning** to perform reranking without needing a separate Cross-Encoder model.

---

## Interview Questions

### Q: Why is HyDE (Hypothetical Document Embedding) risky for some applications?

**Strong answer:**
HyDE relies on "Hallucinating" a baseline answer to find real data. If the user's query describes something non-existent or logically impossible, the LLM will still generate a hypothetical answer. This can pull in "Incorrect but Semantically Similar" data from the database, reinforcing the model's initial hallucination. In 2025, we mitigate this by using a **Hybrid approach**: retrieve once with the real query (Keyword) and once with the HyDE query, and use **RRF** to combine them.

### Q: What is the "Asymmetric Retrieval" problem?

**Strong answer:**
Asymmetric retrieval refers to the fact that user queries are usually short (3-10 words) while document chunks are long (300-500 words). These inhabit different statistical distributions in the vector space, leading to "Distance Bias." High-performance systems solve this using **Asymmetric Encoders** (one model for queries, one for docs) or **Query Expansion** (HyDE) to "inflate" the query into a document-like distribution.

---

## References
- Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, 2023/2024)
- Anthropic. "The Contextual Retrieval Playbook" (2024)
- LlamaIndex. "Query Transformation Cookbook" (2025)

---

*Next: [Agentic Systems](../07-agentic-systems/01-agents-fundamentals.md)*

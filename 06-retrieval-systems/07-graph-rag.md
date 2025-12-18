# GraphRAG (Dec 2025)

GraphRAG is the combination of **Knowledge Graphs (KG)** and **Retrieval-Augmented Generation**. While vector RAG is good at "finding a specific chunk," GraphRAG is designed for **Global Reasoning** across an entire dataset.

## Table of Contents

- [The Limitations of Vector RAG](#limitations)
- [GraphRAG Architecture (Extract-Build-Query)](#architecture)
- [Community Summarization (Microsoft Pattern)](#communities)
- [Entity-Relationship Retrieval](#retrieval)
- [When to Use GraphRAG](#when)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Limitations of Vector RAG

Vector RAG operates on "points" in space. This fails for questions like:
- *"What are the primary themes across all 500 employee reviews?"*
- *"Show me all connections between Project Alpha and the Q3 budget cuts."*

**The Problem**: Vector search finds "similar text," but it doesn't understand "connected entities."

---

## GraphRAG Architecture

A late 2025 GraphRAG pipeline consists of three phases:

1. **Extraction (VLB)**: An LLM scans the text and extracts **Entities** (People, Projects, Dates) and **Relationships** (e.g., "Person A *works on* Project B").
2. **Graph Construction**: The entities are stored as nodes and relationships as edges in a Graph Database (Neo4j, Memgraph).
3. **Querying**: 
   - **Local Search**: Find a node and its neighbors.
   - **Global Search**: Use **Community Summaries** to answer high-level questions.

---

## Community Summarization

 popularized by Microsoft, this technique involves:
1. Identifying clusters of related nodes (Communities) using graph algorithms (e.g., Leiden).
2. Generating a natural language summary for *each* community.
3. At query time, searching the **summaries** instead of the raw chunks.

**The Win**: This allows the model to answer "Big Picture" questions without reading 1M tokens.

---

## Entity-Relationship Retrieval

In 2025, we use **Hybrid Graph-Vector Search**.
- **Dense Pass**: Find the most similar nodes via embeddings.
- **Graph Pass**: Traverse the edges of those nodes to find relevant "supporting" info that might not be semantically similar to the query but is logically connected.

---

## When to Use GraphRAG

| Feature | Vector RAG | GraphRAG |
|---------|------------|----------|
| **Data Type** | Unstructured text | Highly connected data |
| **Query Type**| "Find X" | "Explain the relationship between X and Y" |
| **Scale** | Petabytes | Millions of entities |
| **Cost** | Low | High (Extraction is expensive) |

**2025 Recommendation**: Use GraphRAG for **Internal Knowledge Bases** (Wikis, Codebases, Legal repositories) where the connections between documents are as important as the content itself.

---

## Interview Questions

### Q: Why is the "Extraction" phase the bottleneck for GraphRAG?

**Strong answer:**
Knowledge Graph extraction is extremely token-intensive. To build a high-quality graph, you must process every document with a "Frontier" model to ensure you don't miss subtle entity connections. For a 10,000-page dataset, this can cost thousands of dollars in LLM API calls. In late 2025, we mitigate this by using **SLM-based Extraction** (Small Language Models) for the initial pass and only using giant models for "conflict resolution" between overlapping entities.

### Q: How does GraphRAG solve the "Context Window" limit for aggregate questions?

**Strong answer:**
For aggregate questions (e.g., "Summarize the sentiment of 1,000 documents"), a standard RAG system would have to feed 1,000 chunks into the context window, which is impossible or prohibitively expensive. GraphRAG solves this by **Pre-Summarization**. It hierarchically summarizes the clusters of information in the graph (Communities). When the user asks a global question, the system only retrieves the high-level community summaries, which are compact and rich in information, allowing the model to "see" the entire dataset through a condensed lens.

---

## References
- Edge et al. "From Local to Global: A GraphRAG Approach" (Microsoft Research, 2024)
- Neo4j. "Generative AI and Graph Databases" (2025)
- WhyHow AI. "Deterministic RAG with Knowledge Graphs" (2024)

---

*Next: [Agentic RAG](08-agentic-rag.md)*

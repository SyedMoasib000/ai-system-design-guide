# Embedding Models (Dec 2025)

Embedding models convert text into high-dimensional vectors. In late 2025, we have moved beyond "static vectors" to **Multi-Resolution & Late-Interaction** representations.

## Table of Contents

- [The Embedding Frontier (Matryoshka)]( #matryoshka)
- [Late Interaction (ColBERT v2)]( #late-interaction)
- [Binary and Int8 Quantization]( #quantization)
- [Model Selection Criteria]( #selection)
- [Multimodal Embeddings (Vision + Text)]( #multimodal)
- [Interview Questions]( #interview-questions)
- [References]( #references)

---

## The Embedding Frontier: Matryoshka Embeddings

Traditionally, if you embedded text into 1,536 dimensions, you were stuck using all 1,536 dimensions for search. 

**2025 Innovation: Matryoshka Represenation Learning (MRL)**
- Models are trained to "store" the most important info in the first few dimensions.
- **The Win**: You can embed at 1,536 dims, but index only the first **64 dims** for a "fast search" pass, then refine the top results with the full 1,536 dims.
- **Efficiency**: 20x reduction in memory/index size with <2% drop in accuracy.

---

## Late Interaction: ColBERT v2

Standard embeddings are "Bi-Encoders" (one vector per chunk). **ColBERT** (Contextualized Late Interaction over BERT) uses a "token-level" approach.

- **How**: Instead of 1 vector per chunk, ColBERT stores 1 vector **per token**.
- **Interaction**: At query time, the model compares every token in your query to every token in the documents (the "MaxSim" operation).
- **2025 Status**: ColBERT v2 is drastically compressed (PLAID indexing), making it feasible for production. It achieves much higher precision for "needle in a haystack" technical queries.

---

## Binary and Int8 Quantization

Storing `float32` vectors is expensive. In 2025, we use **In-Model Quantization**.

- **Binary Embeddings**: Convert vectors to 1s and 0s. 
  - **Memory**: 32x reduction.
  - **Speed**: Hamming distance (XOR operations) is 10x faster than Cosine similarity on modern CPUs.
- **Int8/Int4**: Supported natively by models like `text-embedding-3-small`.

---

## Model Selection Criteria (Dec 2025)

| Model | Provider | Features | Context |
|-------|----------|----------|---------|
| **Text-Embedding-4** | OpenAI | Matryoshka, Native Int8 | 32k |
| **Cohere Embed v3.5** | Cohere | Binary quantization, "Compressible" | 1M |
| **BGE-M3** | Open Source | Multilingual, Multi-granularity | 8k |
| **Jina-Embeddings-v3** | Jina AI | Late-interaction support | 128k |

---

## Multimodal Embeddings

In late 2025, "Text-only RAG" is dying. 
- **CLIP (2025 version)**: Embeds images and text into the *same* space.
- **Architecture**: You can search a library of schematics (images) using a natural language query ("Where is the emergency shutoff valve?").

---

## Interview Questions

### Q: What is the "Vocabulary Mismatch" problem in embeddings?

**Strong answer:**
Embeddings rely on the semantic space learned during training. If a user query uses a newer term (e.g., "DeepSeek-V3") that wasn't in the embedding model's training set, the model might assign it a generic "AI" vector, missing the specific nuances. In 2025, we solve this with **Hybrid Search** (using BM25 to catch the specific keyword) or **Cross-Encoder Reranking**, which is better at handling out-of-distribution vocabulary by looking at the query and document tokens simultaneously.

### Q: Why would you choose a Matryoshka model for a 1-billion-vector index?

**Strong answer:**
Scaling to 1 billion vectors with standard `float32` 1536-dim embeddings requires ~6TB of high-speed RAM for an HNSW index, which is prohibitively expensive. With a Matryoshka model, I can use the first 128 dimensions (Binary quantized) for the initial retrieval. This reduces the memory footprint by over 90%, allowing the "Top 1,000" candidates to be found on significantly cheaper hardware. I can then fetch the full-resolution vectors for just those 1,000 candidates to perform the final reranking.

---

## References
- Kusupati et al. "Matryoshka Representation Learning" (2022/2024 update)
- Khattab et al. "ColBERT v1 & v2: Efficient Late Interaction" (2021/2023)
- OpenAI. "Introducing New Embedding Models with Matryoshka Support" (2024)

---

*Next: [Vector Databases Comparison](04-vector-databases-comparison.md)*

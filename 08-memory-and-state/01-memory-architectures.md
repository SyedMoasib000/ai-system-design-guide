# Memory Architectures (Dec 2025)

In late 2025, LLM memory has evolved from "history buffers" to a **Three-Tiered Cognitive Architecture**. This hierarchy mimics human cognitive systems (L1-L3) to balance speed, cost, and recall capacity.

## Table of Contents

- [The Three-Tiered Hierarchy](#hierarchy)
- [Tier 1: Working Memory (Context)](#tier-1)
- [Tier 2: Episodic Memory (Events)](#tier-2)
- [Tier 3: Semantic Memory (Knowledge)](#tier-3)
- [Memory Consolidation Patterns](#consolidation)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Three-Tiered Hierarchy

| Tier | Type | Human Analogy | Technology | Latency |
|------|------|---------------|------------|---------|
| **L1** | Working Memory | Immediate focus | Context Window / KV Cache | <50ms |
| **L2** | Episodic Memory | Past experiences | Vector DB / Local Graph | 100-300ms |
| **L3** | Semantic Memory | General knowledge | Global Graph / SQL / Mem0 | >500ms |

---

## Tier 1: Working Memory (L1)

L1 is the **active focus** of the model. 
- **Context Window**: 128K - 2.5M tokens in late 2025 models.
- **KV Cache**: The GPU "RAM" that stores pre-computed keys and values.
- **Management Strategy**: **Sliding Windows** and **Prefix Caching** (vLLM/PagedAttention).
- **Redundancy Note**: We only keep the most recent turns and critical system instructions in L1.

---

## Tier 2: Episodic Memory (L2)

L2 stores "What happened previously" in this session or past sessions with this user.
- **Storage**: Vector databases (Pinecone, Weaviate, Qdrant).
- **Retrieval**: Semantic search. If the user asks "What did we talk about last Tuesday?", L2 provides the answer.
- **Pattern**: **Experience Replay**. Agents retrieve successful past trajectories to guide current decisions.

---

## Tier 3: Semantic Memory (L3)

L3 stores **Immutable Facts** and **Learned Rules**.
- **Knowledge Graphs**: Store relationships (e.g., `User` -- `WORKS_FOR` --> `Company_X`).
- **Mem0**: A managed service that extract facts (e.g., "User likes Dark Mode") and makes them available globally.
- **Truth Anchoring**: L3 acts as the "Ground Truth" when L1 and L2 provide conflicting information.

---

## Memory Consolidation Patterns

Memories move between tiers via **Consolidation**:
1. **Extraction**: An LLM "Reviewer" extracts facts from L1 at the end of a session.
2. **Indexing**: Facts are stored in L2 (as vectors) and L3 (as graph nodes).
3. **Decay**: Old, non-reinforced memories are moved from L2 to cold storage (L3) or deleted.

---

## Interview Questions

### Q: Why not just use a 2M token context window for all memory (L1-L3)?

**Strong answer:**
While possible, it is **Economically and Cognitively inefficient**. 
1. **Cost**: Calling a model with 1M+ tokens for every turn costs significantly more than a 10K token call with RAG-recalled context.
2. **Attention Dilution**: Even with "Long Context" models, "Lost in the Middle" remains a factor. If the context is cluttered with irrelevant historical turns, the model's reasoning on the *current* task degrades.
3. **Latency**: TTFT (Time to First Token) scales with context size due to KV cache loading. 
A staff-level architecture uses **Strategic Retrieval** to keep the context window lean and focused.

### Q: How do you handle "Privacy Leakage" in Tier 3 (Global Semantic Memory)?

**Strong answer:**
Tier 3 (Semantic Memory) must be **Sharded by Namespace**. Each user or organization gets a unique `namespace_id` in the vector DB and Knowledge Graph. We implement **RLS (Row Level Security)** at the database layer. Additionally, we use a **PII-Scrubbing Layer** during the Consolidation step to ensure that sensitive data (passwords, PII) never moves from the transient L1 context into the persistent L3 knowledge store.

---

## References
- Pack et al. "Generative Agents" (2023/2025 Context)
- OpenAI. "Context Window Optimization" (2025)
- Mem0 Documentation. "Dynamic Memory Management" (2025)

---

*Next: [Short-Term Context Management](02-short-term-context.md)*

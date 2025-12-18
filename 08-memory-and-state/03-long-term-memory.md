# Long-Term Memory (Dec 2025)

Long-term memory (L2 & L3) provides persistence across sessions. In late 2025, we have moved from simple "History RAG" to **Multi-Representation Stores** that combine Vector, Graph, and Relational data.

## Table of Contents

- [Episodic Memory (The Narrative)](#episodic)
- [Semantic Memory (The Knowledge)](#semantic)
- [Hybrid Vector-Graph Storage](#hybrid)
- [Memory Pruning and Decay](#pruning)
- [Privacy and Multi-Tenancy](#privacy)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Episodic Memory: The Personal Log

Episodic memory stores **Trajectories**—sequences of events and their outcomes.
- **Data Structure**: `(Timestamp, Interaction_ID, Trajectory_Summary, Embedding)`.
- **The Rationale**: If an agent successfully built a React component using a specific tool sequence last month, it should "Recall" that success when asked to build another one today.
- **Implementation Note**: We store the *Summary* for retrieval and the *Raw Logs* in cold storage (S3/GCS) for forensic analysis.

---

## Semantic Memory: The Fact Store

Semantic memory stores **Discovered Facts** about entities.
- **Entity Identification**: Using a "Fact Extraction Agent" to parse every user turn.
- **Example triplets**:
  - `(User_1, HAS_PREFERENCE, Dark_Mode)`
  - `(Company_X, USES_SDK, Stripe)`
- **Technology**: Knowledge Graphs (Neo4j, AWS Neptune) combined with relational tagging.

---

## Hybrid Vector-Graph Storage

In 2025, Staff-level engineers use **GraphRAG-style Memory**.
- **Vector Search** finds "Related" nodes.
- **Graph Traversal** finds "Connected" nodes.
- **The Win**: If I search for "Project Alpha," vector search finds the name, but graph traversal finds the 10 developers, the deadline, and the linked code repos.

---

## Memory Pruning and Decay

Memory is a liability if it grows unchecked.
- **Temporal Decay**: Older memories lose their "relevance score" unless frequently accessed.
- **Consolidation**: Merging 10 separate interactions about "billing" into one high-quality summary node.
- **Explicit Forgetting**: Honoring GDPR "Right to be Forgotten" by deleting all episodic and semantic clusters associated with a user ID.

---

## Privacy and Multi-Tenancy

> [!CAUTION]
> **Cross-Session Leakage** is the #1 security risk in global memory. 
> Ensure that the `user_id` is a hard partition key in your vector DB metadata. Never rely on the LLM to filter results by user.

---

## Interview Questions

### Q: How do you choose between a Vector DB and a Knowledge Graph for long-term memory?

**Strong answer:**
I use **Vector DBs** for **Episodic Context** (unstructured logs, past conversations) because I need a "Fuzzy" match on meaning. I use **Knowledge Graphs** for **Structural Semantic Knowledge** (relationships, attributes, hierarchies) because I need "Deterministic" traversal. In 2025, a production system uses a **Hybrid** approach: the vector index points to graph IDs, allowing the system to find the right "Starting Node" and then traverse for high-precision context.

### Q: What is "Catastrophic Forgetting" in the context of learned agentic memory?

**Strong answer:**
In fine-tuned agents, catastrophic forgetting happens when new training data wipes out old knowledge. In **Agentic Memory (RAG-based)**, it refers to **Index Overload**. If an agent adds 1,000 low-quality new "facts" to its memory, the retrieval precision drops, effectively making it "forget" the older, higher-quality facts because they are buried in noise. We mitigate this with **Quality-Weighted Retrieval**—memories with high "Verification Scores" from a supervisor are boosted over raw logs.

---

## References
- Neo4j. "Knowledge Graphs for Generative AI" (2025)
- Pinecone. "The Managed Memory Layer" (2025)
- GraphRAG. "Reasoning over Relationships" (2024/2025)

---

*Next: [Agentic Memory with Mem0](04-agentic-memory-mem0.md)*

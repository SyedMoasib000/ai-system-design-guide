# Agent Memory and State (Dec 2025)

Memory is what allows an agent to learn and maintain context over time. In late 2025, agent memory has evolved from simple "Chat History" to a **Multi-Tiered Cognitive Architecture** that includes persistent episodic and semantic layers.

## Table of Contents

- [The Memory Hierarchy](#hierarchy)
- [Short-Term: The Reasoning Trace](#short-term)
- [Episodic Memory: Past Experiences](#episodic)
- [Semantic Memory: The Persona](#semantic)
- [Mem0 and Personalization](#mem0)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Memory Hierarchy

Agents use a tiered approach to storage:

| Tier | Type | Technology | Purpose |
|------|------|------------|---------|
| **L1** | Working Memory | Context Window / KV Cache | Current task steps, local vars |
| **L2** | Episodic Memory | Vector DB / Graph | "What did I do last time?" |
| **L3** | Semantic Memory | SQL / Knowledge Graph | User preferences, "The Truth" |

---

## Short-Term: The Reasoning Trace

In 2025, we no longer just store the "Messages." we store the **State Object**.
- **The Scratchpad**: A dedicated section of the prompt where the agent "writes notes" to itself that are NOT shown to the user.
- **KV Cache Tiling**: For long-running agents, we use **Prefix Caching** to keep the "System Instruction" and "Standard Tools" warm in GPU memory, only swapping the dynamic task state.

---

## Episodic Memory: Past Experiences

Episodic memory stores "Runs" or "Trajectories."
- If an agent failed to scrape a website last Tuesday, episodic memory should prevent it from trying the same failing selector today.
- **Pattern**: When a task completes, summarize the "Lessons Learned" and store them in a vector DB. At the start of a new task, perform a **Self-Search** for similar previous tasks.

---

## Semantic Memory: The Persona

Semantic memory stores "Facts" about the user or the environment.
- *"The user prefers JSON output."*
- *"The production DB is offline between 3 AM and 4 AM."*

**2025 Best Practice**: Use a **Knowledge Graph** for semantic memory. Unlike vector search (which is fuzzy), a graph provides deterministic retrieval of entities and relationships (e.g., `User` -- `OWNER_OF` --> `Project_A`).

---

## Mem0 and Agentic Personalization

In late 2025, **Mem0** (and similar frameworks) has become the standard for "Smart Memory."
- It automatically extracts "User Insights" from conversations.
- It provides a "Memory API" that agents can call to `remember` or `forget` specific triplets of information.
- **Impact**: Agents feel "Alive" because they remember a detail you mentioned 3 months ago in a different session.

---

## Interview Questions

### Q: How do you handle "Conflicting Memories" in an agentic system?

**Strong answer:**
Conflicting memories (e.g., the user said "I like blue" last week but says "I like red" now) are handled via **Temporal Weighting** or **Explicit Disputing**. In my architecture, I assign a `timestamp` and a `confidence_score` to every memory triplet. If a new fact conflicts with an old one, the agent is prompted to "Resolve the Conflict" by asking the user for clarification or defaulting to the most recent timestamp. We also use **Decay Functions** where older, non-reinforced memories are eventually pruned from the active index.

### Q: Why is the "Context Window" alone insufficient for a staff-level Agent architecture?

**Strong answer:**
First, **Cost and Latency**: Filling 1M tokens of context for every turn is prohibitively expensive even with context caching. Second, **Signal-to-Noise**: Large context windows suffer from "In-context Learning" degradation—the model gets distracted by irrelevant historical turns. A Staff-level architecture uses **Selective Memory Retrieval** (RAG over history) to only pull in the 3-5 most relevant historical interactions, keeping the Reasoning Engine focused on the current sub-goal.

---

## References
- Mem0. "The Memory Layer for AI Agents" (2024/2025)
- Park et al. "Generative Agents: Interactive Simulacra of Human Behavior" (2023)
- LlamaIndex. "Managed Index for Agentic Memory" (2025)

---

*Next: [Planning and Decomposition](06-planning-and-decomposition.md)*

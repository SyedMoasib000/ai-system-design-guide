# Agentic RAG (Dec 2025)

Agentic RAG moves from a "Linear Pipeline" to a **"Reasoning Loop."** Instead of simply retrieving once, an agent decides *when* and *what* to retrieve to resolve a query.

## Table of Contents

- [Linear vs. Agentic RAG](#comparison)
- [Self-RAG (Self-Reflection)](#self-rag)
- [Corrective RAG (CRAG)](#crag)
- [Multi-Hop Reasoning Loops](#multi-hop)
- [Agentic Filtering and Plan Revision](#planning)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Linear vs. Agentic RAG

| Model | Linear RAG | Agentic RAG |
|-------|------------|-------------|
| **Structure** | Predetermined sequence | Dynamic loop |
| **Self-Correction** | None | High (Can re-retrieve) |
| **Query Complexity**| Simple (1-step) | Hard (Multi-step) |
| **Latency** | Low (Fixed) | Variable (Multiple turns) |

**2025 Principle**: Use Agentic RAG when the query requires "Synthesized Proof" rather than just a "Document Match."

---

## Self-RAG (Self-Reflection)

 popularized in 2024/2025, **Self-RAG** uses "Critic Tokens" to evaluate its own work.

1. **Retrieve**: Model pulls Top-K chunks.
2. **Evaluate**: Is the info relevant? (CRITIC: `Relevant`)
3. **Generate**: Is the answer supported? (CRITIC: `Supported`)
4. **Iterate**: If the answer isn't supported, the model *automatically* triggers a broader search.

---

## Corrective RAG (CRAG)

CRAG adds a "Reliability Layer" between retrieval and generation.

- **The Logic**: 
  - If retrieval is **Correct**: Direct generation.
  - If retrieval is **Ambiguous**: Use a Web-Search tool to supplement.
  - If retrieval is **Incorrect**: Discard context and use external search or fallback logic.

---

## Multi-Hop Reasoning Loops

For questions like "Who is the CEO of the company that acquired Figma?", the system must:
1. **Hop 1**: Search for "Who acquired Figma?" (Result: Adobe).
2. **Hop 2**: Search for "CEO of Adobe" (Result: Shantanu Narayen).

**Agentic Pattern**: The agent maintains a "State Object" and updates its "Sub-goal" after every retrieval until the chain is complete.

---

## Agentic Filtering and Plan Revision

In late 2025, agents use **Sub-Step Plans**.
- Instead of one big retrieval, the agent writes a plan: "First I will check our internal database for X, then I will look at the public API for Y."
- **Revised planning**: If Step 1 fails, the agent *rewrites* Step 2.

---

## Interview Questions

### Q: What is the "Reasoning-Retrieval Balance" in Agentic RAG?

**Strong answer:**
Every "Reasoning turn" in an agentic loop adds token cost and user latency. The goal of a production engineer is to find the "Retrieval Threshold." We use **Token-Budgeting** where we allow the agent only 3-5 "turns" before forcing a final answer. We also use **Speculative Retrieval**—where the agent predicts the next 2 steps it will take and retrieves for both simultaneously to reduce round-trip latency.

### Q: Why does Agentic RAG often lead to higher quality but lower "Reliability" (Determinism)?

**Strong answer:**
Agentic RAG is non-deterministic because the model is "Deciding" its path at every step. A small change in the user query might cause the agent to pick a different tool or search strategy, leading to a different answer format. In 2025, we mitigate this by using **Constrained Agent Frameworks** (like LangGraph or DSPy) where the "Graph of possible paths" is strictly defined, even if the choice *between* those paths is stochastic.

---

## References
- Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique" (2024/2025)
- Yan et al. "Corrective Retrieval Augmented Generation (CRAG)" (2024)
- LangChain. "Agentic RAG with LangGraph" (2025)

---

*Next: [Advanced Retrieval Patterns](09-advanced-retrieval-patterns.md)*

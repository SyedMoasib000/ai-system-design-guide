# Framework Selection Guide (Dec 2025)

The landscape of AI frameworks changed drastically in 2025. This guide provides the **Decision Matrix** for choosing your stack based on production requirements, team expertise, and system scale.

## Table of Contents

- [The 2025 Framework Landscape](#landscape)
- [The Decision Matrix](#matrix)
- [Build vs. Buy vs. Framework](#build-vs-buy)
- [Anti-Patterns to Avoid](#anti-patterns)
- [Staff-Level Recommendation](#recommendation)
- [Interview Questions](#interview-questions)

---

## The 2025 Framework Landscape

| Framework | Tier | Primary Value | Key Weakness |
|-----------|------|---------------|--------------|
| **LangGraph** | L1 (Core) | Precise state control | Complexity |
| **DSPy** | L1 (Core) | Reliability & Optimization | Upfront cost (Training) |
| **LlamaIndex**| L2 (Data) | Advanced Retrieval (RAG) | Logic flexibility |
| **CrewAI** | L3 (App) | Business process speed | Hides failures |
| **Semantic Kernel**| L3 (Enterprise)| Microsoft Ecosystem | Python maturity |

---

## The Decision Matrix

**Use this logic to select your stack:**

1. **Is it a pure RAG app?** -> **LlamaIndex**.
2. **Does it require long-running state/Human-in-the-loop?** -> **LangGraph**.
3. **Is high reliability (99%+) and cross-model portability critical?** -> **DSPy**.
4. **Are you a C#/.NET shop?** -> **Semantic Kernel**.
5. **Are you building high-level automations for business users?** -> **CrewAI**.

---

## Build vs. Buy vs. Framework

As a Staff Engineer, you must resist **Framework Bloat**.

- **Use a Framework** when it solves a **Non-Trivial Computer Science Problem** (e.g., State persistence, Bayesian prompt optimization, Vector-Graph linking).
- **Build Custom (Thin Wrapper)** when you are just making simple calls to an LLM. Frameworks add latency, update-churn, and debugging overhead that isn't worth it for a single-turn agent.

---

## Anti-Patterns to Avoid

1. **Framework Tunnelling**: Trying to force a complex logic flow into a framework that doesn't support it (e.g., using a pure RAG library for a coding agent).
2. **The Golden Hammer**: Using LangChain just because it's popular, when a 50-line Python script would be faster and cheaper.
3. **Ignoring Observability**: Deploying any framework without an LLOps layer (LangSmith/Phoenix).

---

## Staff-Level Recommendation (Dec 2025)

For a modern, production-grade agentic system:
- **Orchestration**: LangGraph (for state and loops).
- **Optimization**: DSPy (to compile prompts for different model tiers).
- **Retrieval**: LlamaIndex (for multi-stage RAG).
- **Observability**: LangSmith (for tracing and evaluation).

---

## Interview Questions

### Q: Why do we see a trend towards "Programming" (DSPy) instead of "Prompting"?

**Strong answer:**
**Industrialization**. Prompt engineering is "Alchemy": it is inconsistent and does not scale. Programming LLMs via Frameworks like DSPy allows us to treat AI as a **Software Engineering discipline**. We can apply CI/CD, unit testing (metrics), and automated optimization. This moves AI from "Nondeterministic Magic" to a **Predictable Component** of a larger distributed system, which is a requirement for any mission-critical production environment.

### Q: If you had to build a system that works across OpenAI, Anthropic, and local Llama models, how would you architect it?

**Strong answer:**
I would use **DSPy** for the prompt layer and **LangChain/LangGraph** for the orchestration layer. DSPy's **Signatures** allow me to decouple the task definition from the model's specific "Vibes." I would then use a **Universal Model Gateway** (like LiteLLM or an internal proxy) to handle the different API formats. This stack ensures that if I need to switch from GPT-4o to Claude Sonnet 4.5 for cost or latency reasons, I do not have to rewrite 50 prompts; I just re-compile or update the config.

---

## References
- Google Cloud. "Enterprise Generative AI Reference Architecture" (2025)
- Gartner. "Magic Quadrant for AI Application Frameworks" (2025)
- Thoughtworks. "Technology Radar: The Rise of Agentic Frameworks" (Nov 2024/2025)

---

*New Chapter: [Section 10: Document Processing](../10-document-processing/01-ocr-and-layout.md)*

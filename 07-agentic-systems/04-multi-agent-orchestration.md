# Multi-Agent Orchestration (Dec 2025)

Complex systems are rarely one agent. They are teams of specialized agents. In late 2025, orchestration has moved from "Blind Managers" to **Hierarchical Supervisors** and **Dynamic Swarms**.

## Table of Contents

- [Why Multi-Agent?](#why)
- [The Supervisor Pattern](#supervisor)
- [The Pipeline Pattern](#pipeline)
- [Swarms and Peer-to-Peer (P2P)](#swarms)
- [State Management in Agent Teams](#state)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Multi-Agent?

A single agent with 50 tools experiences **Cognitive Load**.
1. **Specialization**: A "Code Agent" can use a model optimized for Python, while a "Search Agent" uses a model optimized for RAG.
2. **Parallelism**: Multiple agents can work on independent sub-tasks simultaneously.
3. **Decoupled Evaluation**: You can evaluate the "Writer Agent" separately from the "Researcher Agent."

---

## The Supervisor Pattern (Hierarchical)

The most common enterprise pattern in 2025.

- **The Supervisor**: A high-reasoning model (o1-pro) that decomposes the user prompt and delegates to workers.
- **Workers**: Fast, cost-efficient models (Gemini Flash / GPT-4o-mini) that perform the work.
- **Reviewer**: A separate agent that validates the consolidated output against the supervisor's original plan.

**Architecture**: LangGraph is the dominant framework for implementing these state-aware hierarchical loops.

---

## Swarms (The OpenAI Pattern)

 popularized in late 2024, **Swarms** focus on "Handoffs."

- One agent "Hands off" the conversation to another.
- **Key concept**: `Handoff(TargetAgent)`.
- **Benefit**: No central "Manager" bottleneck. The conversation flows naturally between specialized entities.

---

## State Management

The biggest challenge in multi-agent systems is the **Shared Blackboard**.

1. **Local State**: Context only visible to a specific agent.
2. **Global State**: Shared memory (e.g., the final draft) visible to all.
3. **Write Conflicts**: When two agents try to modify the same Global State.
   - **2025 Best Practice**: Use **Transactional Handoffs**. An agent can only write to the global state when it "Owns" the lock.

---

## Peer-to-Peer (P2P) Debate

For high-accuracy tasks (e.g., Legal or Medical), we use **Agentic Debate**.
- **Agent A**: Proposes an answer.
- **Agent B**: Tries to find flaws in Agent A's answer.
- **Agent A**: Refines the answer based on B's critique.
- **Result**: Convergence on a higher-quality result than any single agent could produce.

---

## Interview Questions

### Q: What are the main failure modes of a "Supervisor" multi-agent architecture?

**Strong answer:**
The primary failure mode is **Decomposition Failure**. If the Supervisor agent breaks a task into sub-tasks that are logically inconsistent or have hidden dependencies, the workers will produce correct answers to the *wrong questions*. In 2025, we solve this with **Iterative Planning**—the Supervisor must get "Confirmation of sub-task feasibility" from the workers before they begin execution. Another failure is **Context Dilution**, where the global state becomes so bloated with worker logs that the Supervisor loses the "Big Picture."

### Q: How do you choose between a "Sequence of Chains" and a "Multi-Agent Graph"?

**Strong answer:**
I use a **Sequence of Chains** when the task is linear and deterministic (e.g., Extract -> Translate -> Summarize). I use a **Multi-Agent Graph** (like LangGraph) when the task is **Non-Linear** or requires **Conditional Loops**. For example, if the "Translate" step might fail and need to go back to "Extract" for more context, a static chain breaks, but a graph can self-correct by routing back to an earlier node.

---

## References
- Wu et al. "AutoGPT: An Autonomous GPT-4 Experiment" (Historical/2025 update)
- Li et al. "Camel: Communicative Agents for 'Mind' Exploration" (2023/2025)
- OpenAI. "Swarms Framework" (2024/2025)

---

*Next: [Agent Memory and State](05-agent-memory-and-state.md)*

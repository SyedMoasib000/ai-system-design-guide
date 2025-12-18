# Planning and Decomposition (Dec 2025)

Planning is the "System 2" component that allows agents to solve multi-stage problems without "wandering." In late 2025, we have moved from simple "Chain-of-Thought" to **Recursive Decomposition** and **Tree Search**.

## Table of Contents

- [The Planning Spectrum](#spectrum)
- [Static vs. Dynamic Planning](#static-vs-dynamic)
- [Chain-of-Thought (CoT) and o1 Reasoning](#cot)
- [Recursive Task Decomposition](#decomposition)
- [Tree Search (MCTS) for Agent Paths](#mcts)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Planning Spectrum

| Method | Strategy | Complexity | Best For |
|--------|----------|------------|----------|
| **Linear** | One step at a time | Low | Simple tools |
| **Branching** | If-Then-Else logic | Medium | Conditional flows |
| **Hierarchical** | Master-Plan -> Sub-Plans | High | Software engineering |
| **Search-Based** | Try multiple paths internally | Max | Scientific Research |

---

## Static vs. Dynamic Planning

### Static (Plan-and-Solve)
The agent writes a 10-step plan and follows it strictly.
- **Pros**: High performance, easy to parallelize.
- **Cons**: Brittle. If step 2 fails, steps 3-10 are useless.

### Dynamic (Adaptive)
The agent writes a plan, but **Re-evaluates** after every tool call.
- **2025 Best Practice**: Use **Checkpointed Planning**. The agent is forced to "Commit" its progress to a state store after every major sub-goal to allow for recovery and "Backtracking" if the plan fails.

---

## CoT and o1 Reasoning

In 2025, the model's internal "Thinking" window (Inference scaling) is used as a **Hidden Planner**.
- Instead of using a separate "Planner LLM," we use a reasoning model (o1/R1) to generate a "Mental Draft."
- This draft is translated into a **Task DAG (Directed Acyclic Graph)** that the orchestrator executes.

---

## Recursive Task Decomposition

For massive tasks (e.g., "Build a full-stack app"), we use **Sub-Agent Spawning**.
1. **Master Agent**: Decomposes "Project" into "Frontend," "Backend," and "DB."
2. **Sub-Agents**: Each receives a "Sub-Goal" and performs its own decomposition.
3. **Consolidation**: The Master Agent merges the results.

**Critical Nuance**: Each sub-agent is given a **Minimal Context** (only what it needs) to prevent token bloat and hallucination.

---

## Tree Search (MCTS)

For high-stakes decisions, we use **Monte Carlo Tree Search (MCTS)** within the agent loop.
- The agent "Simulates" 10 possible tool calls.
- A **Reward Model** (or a separate LLM prompt) scores each simulation.
- The agent follows the path with the highest reward.

---

## Interview Questions

### Q: How do you prevent an agent from "Infinite Recursion" during task decomposition?

**Strong answer:**
We implement **Decomposition Depth Limits** (usually 3 levels) and **Granularity Checks**. Before spawning a sub-agent, we ask the Supervisor model: "Is this task small enough to be solved by a single tool call?" If yes, we execute. If no, we decompose. We also use a **Global Controller** that tracks the total "Agent Count" to prevent a recursive bomb (fork bomb) that could drain the API budget.

### Q: Why is "Plan Revision" often more expensive than "Plan Generation"?

**Strong answer:**
Plan generation is a "Fresh Start." Plan revision requires **Context Re-evaluation**—the model must understand what was *already done*, why the *previous step failed*, and how to fix it without undoing previous successes. This requires a much higher "Reasoning Density." In production, we often use a larger model (e.g., Sonnet 3.7 or o1) for the **Revision** step, while using a smaller model for the initial plan generation.

---

## References
- Silver et al. "Mastering the game of Go with deep neural networks and tree search" (Applied to LLMs, 2024/2025)
- Wang et al. "Self-Consistency Improves Chain of Thought Reasoning" (2022/2025 update)
- LangGraph. "Multi-Agent Planning Patterns" (2025)

---

*Next: [Error Handling and Recovery](07-error-handling-and-recovery.md)*

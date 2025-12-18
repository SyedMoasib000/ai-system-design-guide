# Evaluating Agentic Systems (Dec 2025)

Evaluating agents is fundamentally different from evaluating RAG. While RAG is about "Accuracy," Agents are about **"Reliability," "Efficiency," and "Safety."** In late 2025, we use **Trajectory Benchmarks** and **LLM-as-Judge** for multi-step reasoning.

> [!NOTE]
> For standard RAG evaluation (Retrieval vs. Generation metrics), see [06-retrieval-systems/09-advanced-retrieval-patterns.md](../06-retrieval-systems/09-advanced-retrieval-patterns.md) and Section 14. This chapter focuses specifically on the *Execution Path* of an agent.

## Table of Contents

- [The Evaluation Shift](#shift)
- [Trajectory Benchmarks (The GOLD Standard)](#benchmarks)
- [Key Metrics: Success, Cost, and Duration](#metrics)
- [LLM-as-Judge for Step Quality](#judge)
- [Production Evaluation (A/B Testing Agents)](#production)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Evaluation Shift

| Metric | RAG App | Agentic App |
|--------|---------|-------------|
| **Unit of Eval** | Single Response | The **Trajectory** (All steps) |
| **Success Criteria**| Groundedness/Faithfulness | Task Completion / Logical Soundness |
| **Complexity** | Low (Text similarity) | High (Tool state validation) |

---

## Trajectory Benchmarks

In 2025, we evaluate the **"Path to the Result."**
1. **Optimal Path**: The shortest sequence of tools to solve the task.
2. **Agent Path**: The actual steps taken.
3. **The Score**: `Efficiency = (Optimal Steps / Agent Steps)`. A score of `0.2` means the agent meandered or looped excessively.

**Common Benchmarks**:
- **SWE-bench**: Fixing GitHub issues (Code Agency).
- **WebArena**: Navigating menus and forms (Browser Agency).
- **GAIA**: General tool-use tasks (Assistant Agency).

---

## Key Metrics

### 1. Task Success Rate (TSR)
The percentage of tasks where the final state is correct.
> [!IMPORTANT]
> A "Correct Answer" via a "Wrong Path" is a score of 0 in senior production settings.

### 2. Action Success Rate (ASR)
The percentage of individual tool calls that returned valid data (not errors or hallucinations).

### 3. Unit Cost per Task
Total tokens + infrastructure cost (Sandboxes, API calls) per completed goal.

---

## LLM-as-Judge for Step Quality

We use a stronger model (e.g., o1-pro) to review the **Reasoning Log** of a smaller agent.
- **Thought Quality**: Did the agent's logic for using Tool X follow from Observation Y?
- **Redundancy Check**: Did the agent repeat a search it just performed?
- **Feedback Loop**: This "Judge" output is then used for **DPO (Direct Preference Optimization)** to align the agent's future behavior.

---

## Production Evaluation

In late 2025, we use **Shadow Execution**.
1. **V1 Agent** responds to the user.
2. **V2 (Experimental) Agent** runs the same query in a "Hidden Sandbox."
3. **The Comparison**: We compare the two trajectories. If V2 consistently solves tasks in fewer steps without safety violations, we promote it to production.

---

## Interview Questions

### Q: How do you evaluate an agent when the environment is non-deterministic (e.g., the web)?

**Strong answer:**
We use **Mock Environments** or **Snapshotted States**. For high-fidelity testing, we use a containerized browser that resets to a clean state for every test run. We then compare the agent's trajectory against a **Reference Trace**. If the environment is truly live, we use **State-Based Verification**—instead of comparing the text, we check the external world's state (e.g., "Is there a new row in the database with the correct values?").

### Q: Why is "Meandering" (taking too many steps) a critical failure in Staff-level Agent design?

**Strong answer:**
Meandering leads to three failures: 1) **Cost**: Every step is an LLM call; 2) **Latency**: Every step adds 2-5 seconds; 3) **Entropy**: The longer the trajectory, the higher the chance of the agent encountering a weird edge case that triggers a hallucination. In 2025, we implement **Step Budgets**. If an agent doesn't solve a task in 10 steps, we terminate it and escalate to a human to prevent a "Token Leak."

---

## References
- Jimenez et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2024/2025 update)
- Microsoft Research. "AgentBench: A Comprehensive Benchmark for AI Agents" (2024)
- RAGAS. "Agentic Evaluation Module" (2025)

---

*Next: [Memory Architectures](../08-memory-and-state/01-memory-architectures.md)*

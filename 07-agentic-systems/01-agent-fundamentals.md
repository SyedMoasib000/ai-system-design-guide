# Agent Fundamentals (Dec 2025)

Agents are LLM-powered systems that move beyond "chat" into "autonomous problem solving." In late 2025, the definition of an agent has shifted from simple ReAct loops to **Closed-Loop Reasoning Systems** that utilize built-in "System 2" thinking.

## Table of Contents

- [The Agent Formula](#formula)
- [System 1 (LLM) vs. System 2 (Reasoning Model)](#systems)
- [Agency Levels (Autonomous Spectrum)](#levels)
- [Core Components](#components)
- [The Agent Lifecycle](#lifecycle)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Agent Formula

Modern agency is often described as:
`Agent = Reasoning Model + Tool Use + Persistent Memory + Environment Feedback`

**2025 Nuance**: In 2023, agents were "wrappers." In 2025, agents are increasingly **Integrated**. Frontier models (like OpenAI o1 or DeepSeek R1) have the "Thinking" process baked into the pre-training, making the agent loop more stable and less prone to "stalling."

---

## System 1 vs. System 2 Thinking

Architecting an agent requires choosing the right "Thinking Mode":

| Mode | Cognitive Type | Analogy | 2025 Stack |
|------|----------------|---------|------------|
| **System 1** | Fast, intuitive, reactive | Reflexes | GPT-4o / Sonnet 3.5 |
| **System 2** | Slow, logical, planning | Deliberation | o1 / R1 / Reasoning Loops |

**The Design Pattern**: Use System 1 models for "Fast UI" and "Routing." Use System 2 models for "Decision Gates" and "Complex Planning."

---

## Agency Levels

Not every autonomous system is an "Agent." We categorize them by the **Level of Agency**:

1. **L0: Scripted Chains**: Fixed sequence (e.g., standard LangChain).
2. **L1: Tool-Enabled**: Model picks a tool but doesn't plan.
3. **L2: ReAct Agent**: Simple loop of "Thought -> Action -> Observation."
4. **L3: Autonomous Planner**: Decomposes a goal into a graph of sub-tasks.
5. **L4: Ambient Agent**: Runs in the background, intervenes only when necessary.

---

## Core Components

### 1. The Reasoning Model (The Executive)
The CPU of the agent. It determines the "Path to Success."

### 2. Tools (The Limbs)
Interfaces (APIs, Browsers, DBs) that allow the agent to affect the world.
> [!Note]
> In late 2025, standard Tool-Use is being replaced by the **Model Context Protocol (MCP)** for standardized tool interoperability.

### 3. Memory (The Experience)
- **Short-term**: Context window (KV Cache).
- **Long-term**: Vector DBs or persistent state (e.g., Mem0).

---

## The Agent Lifecycle

1. **Intake**: Receive user goal.
2. **Decomposition**: Break goal into sub-steps.
3. **Execution**: Call tools and handle results.
4. **Reflection**: Evaluate if the observation got the agent closer to the goal.
5. **Completion**: Synthesize final proof for the user.

---

## Interview Questions

### Q: Why is a "Reasoning Model" (like o1) better for agency than a standard LLM?

**Strong answer:**
Standard LLMs (System 1) predict the *very next token* based on pattern matching. When they encounter an error in a tool call, they often hallucinate a fix instead of admitting the failure. Reasoning Models use **Chain-of-Thought (CoT)** during inference. They "think" through multiple hidden turns before outputting a response. For an agent, this means higher **Path Reliability**—the model is significantly less likely to enter an infinite loop or try the same failing action twice because it has already simulated the failure internally.

### Q: How do you prevent "Agentic Drift" in long-running tasks?

**Strong answer:**
Agentic Drift occurs when the sub-steps take the agent so far from the original goal that it loses context. In 2025, we solve this with **Goal Anchoring**. We include the "Original Objective" as a pinned system message and use a **Secondary Observer Model** (a smaller, cheaper model) to score every agent action against the original objective. If the score drops below a threshold, the agent is forced to "re-plan" from the root.

---

## References
- Kahneman, D. "Thinking, Fast and Slow" (applied to AI, 2025)
- OpenAI. "Learning to Reason with LLMs" (2024)
- DeepSeek. "R1: Cold-Start Data for Reasoning" (2025)

---

*Next: [Reasoning Loops: ReAct and Beyond](02-reasoning-loops-react-and-beyond.md)*

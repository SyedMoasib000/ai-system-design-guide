# Error Handling and Recovery (Dec 2025)

Agents fail in non-deterministic ways. In late 2025, error handling has moved from "Try-Catch blocks" to **Agentic Self-Correction** and **Stateful Rollbacks**.

## Table of Contents

- [The Taxonomy of Agent Failures](#fail-types)
- [Self-Correction Loops](#correction)
- [Stateful Rollbacks (Checkpointing)](#rollbacks)
- [The "Stuck in a Loop" Fix](#stuck)
- [Graceful Degradation](#degradation)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Taxonomy of Agent Failures

1. **Hallucinated Tools**: Calling a tool that doesn't exist.
2. **Schema Violation**: Passing the wrong arguments to a real tool.
3. **Environment Error**: Tool exists, but the external API is down.
4. **Logical Stall**: The agent performs the same failing action repeatedly (The ReAct Loop of Death).

---

## Self-Correction Loops

In 2025, errors are treated as **Tokens of Information**.

- **Pattern**: When a tool fails, the error message is NOT just logged; it is fed back to the model as a prompt: *"Action failed with error: X. Reflect on why this happened and provide an alternative strategy."*
- **Reasoning Models (o1/R1)**: These models excel at this because they "internalize" the error during their hidden Chain-of-Thought, leading to a much higher one-shot recovery rate.

---

## Stateful Rollbacks (Checkpointing)

For long-running agents, an error in Step 9 shouldn't crash the whole project.

- **Checkpoints**: High-reliability systems (using LangGraph or similar) save the "State Snapshot" to a DB after every successful tool call.
- **The Rollback**: If the agent enters a logical stall, the supervisor agent can **Reset common-state** to Step 5—the last "Safe" state—and force a different path.

---

## The "Stuck in a Loop" Fix

Infinite loops are the #1 cost-sink in agentic systems.

**The 2025 Solution**: **Counter-Based Intervention**.
1. If the same `(Tool, Args)` tuple is seen 3 times in one session, the orchestrator interrupts the model.
2. It injects a mandatory **"Pivot Instruction"**: *"You have tried searching for 'X' three times. This path is dead. You MUST try a different tool or admit you are stuck."*

---

## Graceful Degradation

If the high-reasoning agent (o1) keeps failing, we fall back to:
- **Simplified Agent**: A smaller model with fewer, more reliable tools.
- **RAG-only Mode**: Disable actions and just provide a conceptual answer based on the knowledge base.
- **Human Escalation**: (See the next chapter).

---

## Interview Questions

### Q: Why is traditional "Exception Handling" (Try/Catch) insufficient for Agentic Systems?

**Strong answer:**
In traditional software, an exception is a "Stop" command. In an agentic system, the model is the "Driver." If the system just stops, the user task fails. We use **Error Injection** instead of Exception Handling. We catch the exception at the platform level and transform it into a **Synthesized Observation** for the model. This allows the model to "Reason" around the failure. A TRY/Catch only fixes the code; Error Injection allows the model to fix the **Plan**.

### Q: How do you handle "Silent Failures" (Where the tool returns 200 OK but the data is wrong)?

**Strong answer:**
Silent failures are the most dangerous. We implement **Output Validation Agents**. For critical steps, we don't just accept the tool output. We pipe the output to a "Verifier Agent" (often a smaller, faster model) whose only job is to check: *"Does this tool output actually answer the query provided?"* If the Verifier says "No," it triggers a self-correction loop as if it were a hard error.

---

## References
- LangGraph. "Persistence and Checkpointing" (2025)
- Shinn et al. "Reflexion: Learning from Errors" (2024 update)
- Microsoft. "Managing Hallucinations in Agentic Systems" (2025)

---

*Next: [Human-in-the-Loop Patterns](08-human-in-the-loop-patterns.md)*

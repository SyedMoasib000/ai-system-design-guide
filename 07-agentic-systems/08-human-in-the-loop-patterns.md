# Human-in-the-Loop Patterns (Dec 2025)

No agent is 100% reliable. **Human-in-the-Loop (HITL)** is the bridge that ensures safety and accuracy in high-stakes environments. In late 2025, we have moved beyond "Approval Buttons" to **Co-Reasoning** and **Interrupt-Based Steering**.

## Table of Contents

- [The HITL Spectrum](#spectrum)
- [Interrupts and Breakpoints](#interrupts)
- [Time-Travel Debugging (State Editing)](#time-travel)
- [Co-Reasoning (Shared Scratchpads)](#co-reasoning)
- [Confidence-Based Escalation](#escalation)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The HITL Spectrum

| Pattern | Agent Autonomy | Human Role | Best For |
|---------|---------------|------------|----------|
| **Human-in-command** | Low | Drives every step | High-risk Legal/Medical |
| **Human-as-filter** | Medium | Approves/Edits final output | Content Generation |
| **Human-as-backup** | High | Only intervenes on error | Customer Support |
| **Human-on-the-loop** | Max | Audits logs after completion | High-volume analysis |

---

## Interrupts and Breakpoints

In 2025 architectures (like LangGraph), we use **Deterministic Breakpoints**.

- **The Pattern**: The system is hardcoded to "Pause" before a specific sensitive tool is called (e.g., `execute_purchase` or `delete_user`).
- **The Decision**: The environment waits for a user to send an `approve` or `reject` signal.
- **State Preservation**: The agent's reasoning state is "Frozen" in the DB until the human acts.

---

## Time-Travel Debugging (State Editing)

Standard agents are "One-way." If they make a mistake in Step 3, the session is usually ruined.
- **2025 Innovation**: **State Injection**. A human reviewer can "Go back" to the state at Step 3, edit the agent's observation or thought, and then "Resume" execution.
- **Impact**: It allows humans to "Steer" the agent off a bad path without starting from zero.

---

## Co-Reasoning (Shared Scratchpads)

Instead of the human being a "Judge," they become a **"Partner."**
- The agent shows its **Scratchpad** (Internal Thinking) to the human.
- Characterized as: *"I am planning to use Tool A because of Fact B. Does that seem right to you?"*
- **Benefit**: Catching reasoning errors *before* they translate into actions.

---

## Confidence-Based Escalation

Using models that support "Logprobs" or built-in reasoning steps, we calculate an **Uncertainty Score**.

- If the score exceeds a threshold, the agent **Automatically Pauses** and sends a notification to a human operator.
- **Example**: An agent trying to resolve a complex billing dispute realizes the user's intent is ambiguous. It stops and says: *"I'm not 100% sure how to handle this specific refund case. One moment while I get a human expert to look at this."*

---

## Interview Questions

### Q: How do you design an HITL system that doesn't "Fatigue" the human operator?

**Strong answer:**
We use **Threshold Tuning**. We don't ask for approval on every action. We only trigger HITL for: 1) High-risk "Writing" tools, 2) Low-confidence reasoning steps, or 3) Actions that violate a "Policy" set by the business. Additionally, we provide the human with a **Contextual Summary**—instead of the whole log, we show them a 1-sentence "Diff" of what the agent wants to do. This reduces the "Review cognitive load" from minutes to seconds.

### Q: What is the "Over-Reliance" risk in HITL, and how do you mitigate it?

**Strong answer:**
Over-reliance happens when humans start clicking "Approve" without reading the logs. We mitigate this with **Forced Review Checkpoints** (e.g., the human MUST edit at least one word in the proposed plan) or **Synthetic Error Injections** (intentionally showing the human a "wrong" plan 1% of the time to see if they catch it). If they pass the "Trap," they continue; if they fail, they are flagged for additional training.

---

## References
- Wu et al. "Co-reasoning: Human-AI Collaboration Patterns" (2025)
- LangChain. "Human-in-the-loop in LangGraph" (2024/2025)
- Anthropic. "Designing for Safety and Human Oversight" (2024)

---

*Next: [Agentic Security and Sandboxing](09-agentic-security-and-sandboxing.md)*

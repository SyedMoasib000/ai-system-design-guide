# AutoGen and CrewAI (Dec 2025)

In late 2025, while LangGraph dominates the "low-level" orchestration layer, **AutoGen** and **CrewAI** have carved out out niches as **High-Level Agentic Frameworks**. They focus on "Collaborative AI" where the main abstraction is the **Role-Playing Agent**.

## Table of Contents

- [CrewAI: The Manager Perspective](#crewai)
- [AutoGen: The Developer Perspective](#autogen)
- [Swarms and Peer-to-Peer Communication](#swarms)
- [Framework Comparison Matrix](#comparison)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## CrewAI: The Manager Perspective

CrewAI is built around the concept of a **Process**.
- **Role-Based Agents**: You define a "Researcher," a "Writer," and a "Manager."
- **Tasks**: Explicit goals with specific outputs.
- **Process Orchestration**: Sequential, Hierarchical, or Consensual (Consensus-based).

**2025 Use Case**: CrewAI is the best framework for **Automating White-Collar Workflows** (e.g., generating a full marketing campaign from a single prompt) where the structure of the team is well-understood.

---

## AutoGen: The Developer Perspective

Microsoft's AutoGen is built for **Dynamic Conversation**.
- **Conversable Agents**: Every agent is a "Chat participant."
- **Code Execution**: Integrated support for running code in sandboxes (E2B/Docker).
- **GroupChat Manager**: A specialized agent that decides *who speaks next*.

**2025 Use Case**: AutoGen excels at **Joint Logic Generation** and **Self-Healing Code execution**, where agents need to iterate on a technical problem.

---

## Swarms and P2P

In late 2025, both frameworks have adopted **Swarm Patterns**.
- **The Handoff**: Instead of a central supervisor, agents "Hand off" the conversation to the most relevant expert.
- **Example**: A "Sales Agent" realizes the user is asking a technical question and hands off the thread to the "Support Agent."

---

## Framework Comparison Matrix

| Feature | CrewAI | AutoGen | LangGraph |
|---------|--------|---------|-----------|
| **Core Abstraction** | Task/Process | Conversation | State/Graph |
| **Ease of Use** | High (Declarative) | Medium | Low (Imperative) |
| **Control** | Low | Medium | High |
| **Best For** | Business Automations | Collaborative Logic | Complex Tool-Use |

---

## Interview Questions

### Q: When would you use CrewAI instead of LangGraph?

**Strong answer:**
**Speed vs. Precision**. I use **CrewAI** when I need to stand up a team of agents for a standard process (like content generation or data analysis) very quickly. It provides high-level abstractions for "Planning" and "Cooperation" out of the box. I switch to **LangGraph** when I need **Granular Control** over every state transition, multi-turn human-in-the-loop triggers, or complex error-recovery logic that doesn't fit into the "Role-playing team" metaphor.

### Q: How does AutoGen handle "Infinite Loops" where agents keep talking to each other without solving the task?

**Strong answer:**
We use **Termination Conditions** and **Max Conversational Turns**. In 2025, we also implement a \"Critic Agent\" whose only job is to detect if the conversation is stagnant. If the Critic detects circularity, it triggers a `UserProxy` to interrupt or force-switches the `GroupChatManager` to a different reasoning path. We also monitor **Token Velocity**: if an agent pair uses 100K tokens in 2 minutes without progress, we kill the session automatically.

---

## References
- CrewAI. "The Multi-Agent Process Engine" (2025)
- Microsoft Research. "AutoGen: Enabling Next-Gen LLM Applications" (2025)
- OpenAI Swarm. "Lightweight Multi-Agent Orchestration" (2024 tech report)

---

*Next: [Framework Selection Guide](08-framework-selection-guide.md)*

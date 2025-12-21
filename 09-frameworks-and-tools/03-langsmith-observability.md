# LangSmith Observability (Dec 2025)

In 2023, LLM observability was "logging strings." In late 2025, it is **Full Trajectory Debugging** and **Automated Evaluation Pipelines**. LangSmith is the industry standard for this "LLMOps" layer.

## Table of Contents

- [The Observability Pyramid](#pyramid)
- [Tracing and Trajectories](#tracing)
- [Unit Testing for LLMs (Datasets)](#datasets)
- [Automated Evaluators (LLM-as-Judge)](#evaluators)
- [Managing Deployment: A/B Testing](#ab-testing)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Observability Pyramid

1. **Top (Value)**: Is the user task getting completed? (Success Rate)
2. **Middle (Flow)**: Which agent node is the bottleneck? (Latency/Cost per node)
3. **Bottom (Raw)**: What were the exact prompt/completion pairs? (Traces)

---

## Tracing and Trajectories

LangSmith automatically captures every node in a **LangGraph** or **Chain**.
- **Metadata Tagging**: In 2025, we tag every trace with `user_id`, `model_tier`, and `is_canary`.
- **The Debugger**: You can \"Play back\" a trace in the LangSmith UI, modifying the prompt and seeing how the response changes. This works without re-running the entire application.

---

## Unit Testing for LLMs (Datasets)

Building an LLM app without a **Dataset** is "vibe-based development."
- **Gold Standard Datasets**: A collection of `(Input, Expected_Output)` pairs.
- **2025 Workflow**: Whenever a user provides negative feedback, that interaction is automatically pumped into a "Correction Dataset" for future testing.

---

## Automated Evaluators

You cannot manually check 1,000 log entries every morning.
- **LLM-as-Judge**: Using a superior model (o1/R1) to score the production model on categories like **Tone**, **Accuracy**, and **Safe Action execution**.
- **Custom Evaluators**: Python functions that check for regex patterns, JSON schema validity, or Toxicity scores.

---

## A/B Testing in 2025

LangSmith allows for **Experiment Branching**.
- Run 2% of traffic on a new "System Prompt" version.
- Compare the **Success Rate** and **Token Cost** in real-time.
- Automatically roll back if the failure rate exceeds a threshold.

---

## Interview Questions

### Q: Why is "Trace Attribution" critical for Staff-level engineers?

**Strong answer:**
In complex multi-agent systems, the final output might be bad, but the error happened 10 steps ago in a "Researcher" node. Without **Trace Attribution**, you're just guessing where to fix the prompt. Attribution allows me to see the **Line of Reasoning**. I can see that the "Researcher" failed to find the right URL, which led to the "Summarizer" hallucinating. This allows for **Targeted Optimization** instead of broad "Prompt Engineering."

### Q: How do you justify the cost of an observability platform like LangSmith?

**Strong answer:**
The cost is offset by **Developer Productivity** and **Token Efficiency**. A single day of an engineer "guessing" why a model is failing costs significantly more than a monthly subscription. Moreover, by using LangSmith to find "Meandering" agents (those taking too many steps), I can optimize the graphs to reduce the average number of steps from 8 to 5, which directly results in a **30-40% reduction in LLM API bills**.

---

## References
- LangChain Team. "LangSmith: The Unified Evaluation Platform" (2025)
- Microsoft. "Tracing and Debugging Multi-Agent Systems" (2025)
- Weights & Biases. "Integrating LLOps into the CI/CD Pipeline" (2024/2025)

---

*Next: [LlamaIndex and Data-Centric AI](04-llamaindex.md)*

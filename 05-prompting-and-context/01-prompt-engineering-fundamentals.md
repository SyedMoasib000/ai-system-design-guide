# Prompt Engineering Fundamentals

Prompt engineering is the design of inputs to steer LLM behavior. In 2025, it has evolved from "trial and error" to a disciplined architectural practice.

## Table of Contents

- [The Core Philosophy (Intent + Constraint)](#core-philosophy)
- [The Instruction Hierarchy](#instruction-hierarchy)
- [Role Prompting (Dec 2025 Standard)](#role-prompting)
- [Instruction Clarity and Delimiters](#clarity)
- [Zero-Shot vs. Few-Shot Efficiency](#zero-vs-few)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Core Philosophy: Intent + Constraint

Effective prompting is about maximizing **Intent Disclosure** while minimizing **Output Variance**.

1. **Intent**: Precisely what the model should do.
2. **Constraint**: Exactly what the model should *avoid* (Safety, Tone, Format).

**2025 Principle**: "Prompting is Programming in Natural Language." Treat your prompts like code (Version control, Unit tests).

---

## The Instruction Hierarchy

Production systems use a tiered message structure:

| Role | Responsibility | 2025 Nuance |
|------|----------------|-------------|
| **System** | High-level rules, persona, safety. | Stickiest for frontier models (H-rank). |
| **Developer** | Technical overrides (e.g., formatting). | Newest role for "un-opinionated" models. |
| **User** | The specific, dynamic query. | Susceptible to injection; must be isolated. |
| **Assistant**| History of previous turns. | Source of "recency bias." |

---

## Role Prompting (Dec 2025 Standard)

Assigning a persona is no longer just "You are a teacher." It is a **Capabilities Anchor**.

- **Weak**: "You are a coder."
- **Strong**: "You are a Staff Software Engineer at a Tier-1 tech company specializing in high-concurrency Rust systems. You prioritize memory safety and zero-cost abstractions."

**Why it works**: It focuses the model's attention on the specific subset of its training data related to that high-level expertise, reducing irrelevant hallucinations.

---

## Instruction Clarity and Delimiters

Models in 2025 process massive contexts. Delimiters help the model distinguish between instructions and data.

```markdown
# Instructions
Analyze the following text for PII.

# Data to Analyze
--- START OF USER DATA ---
$USER_INPUT_HERE
--- END OF USER DATA ---

# Output Schema
{ "pii_found": boolean, "types": [] }
```

**Delimiters to use**: XML tags (`<context>`, `</context>`), Markdown headers (`#`), or triple quotes (`"""`).

---

## Zero-Shot vs. Few-Shot Efficiency

| Aspect | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| **Latency** | Lowest (Short prompt) | Higher (Example tokens) |
| **Accuracy**| Variable | High (Format stability) |
| **Use Case**| Simple chat, Summarization | Specific formatting, Subtle logic |

**2025 Strategy**: If the model is a "Frontier Reasoning" model (o1, DeepSeek-R1), use **Zero-Shot + Clear Chain-of-Thought**. If it's a small model (8B), use **Few-Shot** to ground it.

---

## Interview Questions

### Q: Why do system prompts carry more weight than user prompts in modern LLMs?

**Strong answer:**
System prompts are typically prioritized by the model's architectural training (RLHF) and may be injected into a special "instruction-only" embedding space in some architectures. From a design perspective, the system prompt defines the "Constitution" of the interaction. If a user prompt contradicts a system prompt (e.g., asking for a bomb recipe), a well-aligned model is trained to prioritize the system's "Safety Constraint" over the user's "Task Intent."

### Q: What is the "Step-by-Step" prompt optimization?

**Strong answer:**
In 2022, "Think step by step" was a magic phrase to trigger Chain-of-Thought (CoT). In 2025, we use **Programmatic CoT**. Instead of a vague phrase, we provide explicit reasoning milestones: "1. Identify the core problem. 2. List the constraints. 3. Propose 3 solutions. 4. Select the best one and justify." This provides a "deterministic path" for the model's internal attention, leading to much more reliable outputs for production agents.

---

## References
- OpenAI. "Prompt Engineering Guide" (2024-2025)
- Anthropic. "Claude Prompt Engineering Documentation" (2024)
- Google DeepMind. "The Power of Prompting" (2023)

---

*Next: [Few-Shot and In-Context Learning](02-few-shot-and-icl.md)*

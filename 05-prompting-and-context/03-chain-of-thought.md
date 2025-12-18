# Chain-of-Thought (CoT)

Chain-of-Thought (CoT) is the technique of encouraging an LLM to generate intermediate reasoning steps before providing a final answer. In late 2025, this has evolved from a simple prompt phrase to the CORE architectural feature of reasoning models.

## Table of Contents

- [The CoT Revolution](#cot-revolution)
- [Zero-Shot vs. Programmatic CoT](#zero-vs-programmatic)
- [The Rise of "Thinking" Models (o1, DeepSeek-R1)](#thinking-models)
- [Self-Correction and Verification](#self-correction)
- [When CoT Fails (Over-thinking)](#over-thinking)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The CoT Revolution

Standard LLMs are "Next Token Predictors." For complex math or logic, a single pass is often insufficient. CoT provides the "Scribble Pad" (Working Memory) for the model to work through sub-problems.

**The Formula**: `Input -> Reasoning (Chain) -> Output`

---

## Zero-Shot vs. Programmatic CoT

| Technique | Trigger Phrase | Efficiency | Use Case |
|-----------|----------------|------------|----------|
| **Zero-Shot CoT** | "Let's think step by step." | High | Ad-hoc queries. |
| **Few-Shot CoT** | (Provided examples with logic) | Higher Stability | Production pipelines. |
| **Programmatic CoT** | "1. Analyze X. 2. Verify Y. 3. Resolve Z." | **Best for Agents** | Complex multi-tool tasks. |

---

## The Rise of "Thinking" Models (Dec 2025)

Models like **OpenAI o1** and **DeepSeek-R1** have CoT "baked in" via Reinforcement Learning (RL).

1. **System-Level CoT**: The model doesn't just "print" reasoning; it has a dedicated "Thinking Window."
2. **Hidden CoT**: In many enterprise versions, the reasoning chain is hidden from the user but verifiable by the system to prevent prompt injection or "thought leakage."
3. **Scaling Law**: These models follow the **Inference Scaling Law**—the longer they "think," the better they solve hard problems ($o1$ can solve gold-medal IMO math given enough time).

---

## Self-Correction and Verification

In 2025, we no longer trust a single Chain-of-Thought. We use **Self-Verification**.

```markdown
# Process
1. Generate Answer A via CoT.
2. Critique: "Are there any errors in the logic above?"
3. If errors: "Correct the logic and provide Answer B."
```

**The 2025 Nuance**: This is now integrated into **Execution-Verified CoT** for coding, where the model writes the logic, runs the code, and corrects itself if the code fails.

---

## When CoT Fails (Over-thinking)

CoT is not a silver bullet. For simple tasks, it adds:
1. **Latency**: More tokens = slower response.
2. **Cost**: You pay for every "thought" token.
3. **Over-thinking**: The model might hallucinate complexity where none exists (e.g., explaining why 2+2=4 for 3 paragraphs).

---

## Interview Questions

### Q: Why does CoT improve performance on mathematical word problems?

**Strong answer:**
CoT improves performance by aligning the model's computational complexity with the task's logical complexity. In a standard single-pass generation, the model must predict the final answer token based on limited local information. With CoT, the model "breaks" the problem into smaller, auto-regressive steps. Each step uses the previous step's output as context, allowing the model's attention mechanism to focus on one sub-problem at a time (e.g., first adding the apples, then subtracting the oranges), reducing the "cognitive load" of the single-pass prediction.

### Q: How do you handle CoT in a production environment where latency is critical?

**Strong answer:**
We use a **Hybrid Reasoning Architecture**:
1. **Tier 1 (Fast)**: A classifier identifies if the query needs deep reasoning.
2. **Tier 2 (Condensed CoT)**: We prompt the model with "Be concise in your reasoning," or use "Knowledge Distillation" where we train a smaller model to produce *only* the final answer while benefitting from a teacher's CoT-style pretraining. 
3. **Tier 3 (Streaming)**: We stream the CoT to the user (if transparent) or a background process so the system can begin "pre-processing" the final result as it appears.

---

## References
- Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2023)
- OpenAI. "Learning to Reason with LLMs" (2024)

---

*Next: [Tree-of-Thought](04-tree-of-thought.md)*

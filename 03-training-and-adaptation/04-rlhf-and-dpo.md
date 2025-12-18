# RLHF and DPO (Alignment)

Alignment is the process of ensuring an LLM's behavior matches human values and instructions. In 2025, the field has moved from traditional RLHF to more efficient and scalable methods like DPO and Online RL.

## Table of Contents

- [The Alignment Problem](#the-alignment-problem)
- [RLHF: The Foundation](#rlhf-foundation)
- [DPO: Direct Preference Optimization](#dpo)
- [Online Alignment (Dec 2025 Standard)](#online-alignment)
- [Alignment for Reasoning Models](#alignment-for-reasoning)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Alignment Problem

Pretrained models are "knowledgeable but uncontrolled." They may:
1. Generate harmful content (Safety).
2. Fail to follow instructions (Instruction Following).
3. Hallucinate wildly (Factuality).

Alignment creates "Reward Models" and "Policy Updates" to steer the model.

---

## RLHF: The Foundation

Reinforcement Learning from Human Feedback (RLHF) involves three steps:
1. **SFT**: Supervised Fine-Tuning.
2. **Reward Model (RM)**: Train a model on `(Prompt, Winning_Response, Losing_Response)` to predict human scores.
3. **PPO (Proximal Policy Optimization)**: Use the RM to provide a "reward signal" to the LLM via Reinforcement Learning.

**2025 Nuance**: Traditional RLHF is now considered too complex/unstable for most teams due to the overhead of training a separate Reward Model and the instability of PPO.

---

## DPO: Direct Preference Optimization

DPO is the **2024-2025 industry standard**. It eliminates the Reward Model.

### How it Works:
DPO uses the LLM itself as the Reward Model by mathematically deriving the optimal policy directly from preference data.
- **Goal**: Maximize the probability of the "winning" response and minimize the "losing" response, relative to a fixed "reference model."

### The 2025 Multi-Stage Alignment Pattern:
1. **Base SFT**: 5k-10k high-quality samples.
2. **DPO Step 1**: Alignment for instruction following.
3. **DPO Step 2**: Alignment for safety and specific tone.

---

## Online Alignment (Dec 2025 Standard)

**The Problem with Offline DPO**: It only learns from static data. If the model improves beyond that data, it hits a ceiling.

**The Solution: Online DPO (or RLOO)**:
1. The model generates 4-8 responses to a prompt.
2. A **Judge Model** (e.g., GPT-5.2) or a **Rule-based Reward** (e.g., Code Execution) ranks them in real-time.
3. The model updates its policy immediately based on this "Online" feedback.

---

## Alignment for Reasoning Models (o1/DeepSeek-R1 style)

Aligning "Thinking" models requires a shift from **Response Preference** to **Process Preference**.

| Feature | Standard Alignment | Reasoning Alignment |
|---------|-------------------|---------------------|
| Reward Target | The final answer | The **Chain of Thought (CoT)** |
| Reward Signal | Helpful/Safe | **Correctness + Conciseness** |
| Method | Human Ranking | Rule-based (e.g., "Did the code run?") |

**Principal-level Nuance**: "Verification-based RL" is the secret to 2025 frontier models. Instead of humans saying what is better, we use hard verifiable outcomes (Math answers, Code test cases) as the reward signal.

---

## Interview Questions

### Q: Why is DPO often preferred over RLHF/PPO in 2025?

**Strong answer:**
DPO is preferred primarily due to its simplicity and stability. PPO requires maintaining four models in memory (Policy, Reference, Value, and Reward), which is extremely VRAM-intensive. Furthermore, PPO is notoriously sensitive to hyperparameters and often suffers from "reward hacking" or sudden collapse. DPO treats alignment as a simple classification problem on preference pairs, making it much more robust, easier to tune, and significantly cheaper to run.

### Q: What is the risk of "Alignment Tax"?

**Strong answer:**
The "Alignment Tax" refers to the decline in a model's raw capabilities (e.g., coding, creative writing, or logical reasoning) after it is aligned for safety or specific personas. Because the model is being forced to prioritize safety or adherence to a specific style, it may become "too cautious" or lose the nuance it learned during pretraining. Modern 2025 techniques like **Steerable Alignment** and **DPO-with-KL-penalty** aim to minimize this by ensuring the model's policy doesn't drift too far from the original pretrained distribution.

---

## References
- Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- OpenAI. "Learning to Reason with LLMs" (2024)

---

*Next: [Knowledge Distillation](05-knowledge-distillation.md)*

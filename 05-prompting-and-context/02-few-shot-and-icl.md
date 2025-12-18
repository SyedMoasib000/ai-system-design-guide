# Few-Shot and In-Context Learning (ICL)

In-Context Learning (ICL) is the ability of an LLM to learn a new task simply by seeing examples in the prompt, without any weight updates. In 2025, maximizing ICL efficiency is the key to prompt stability.

## Table of Contents

- [The Anatomy of a Few-Shot Example](#anatomy)
- [How many examples? (The 2025 Standard)](#how-many)
- [Dynamic Example Selection](#dynamic-selection)
- [The Importance of Labelling Nuance](#labelling)
- [Advanced ICL: Analogy and Retraining-lite](#advanced-icl)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Anatomy of a Few-Shot Example

A high-quality example consists of three parts:
1. **Input**: A realistic sample of potential user data.
2. **Reasoning (Optional)**: A short explanation of *why* the output is what it is.
3. **Output**: The "Gold Standar" result.

```markdown
User: "The weather is okay, but the flight was late."
Reasoning: The user is neutral about the weather but negative about the service.
Sentiment: Mixed
```

---

## How many examples? (The 2025 Standard)

| Model Size | Sweet Spot | Scaling Behavior |
|------------|------------|------------------|
| **Small (8B)** | 5 - 10 | Gains continue until ~20 examples. |
| **Medium (70B)**| 3 - 5 | Plateaus early; more examples increase latency. |
| **Frontier (405B)**| 1 - 2 | Highly capable; "Instruction Following" usually suffices. |

**The 2025 Rule**: If you need more than 20 examples to get a stable output, your task is likely too complex for the model, or you should consider **Fine-tuning**.

---

## Dynamic Example Selection

In production RAG or Classification, don't use the same static examples for every user.
**The Dynamic Pattern:**
1. User provides a query.
2. Search a "Vector DB of Gold Examples" for the 3 most **semantically similar** cases.
3. Inject those 3 specific cases into the prompt.

**Result**: Drastically higher accuracy because the model sees "local" patterns relevant to the current user.

---

## The Importance of Labelling Nuance

Frontier models are sensitive to **Distribution Bias** in examples.
- If you provide 5 "Positive" examples and 1 "Negative," the model will bias toward "Positive."
- **The 2025 Fix**: Always use **Label Balancing**. Ensure your few-shot examples roughly mirror the expected output distribution or are perfectly balanced (1:1).

---

## Advanced ICL: Analogy and "Few-Shot CoT"

**Analogy Prompting**: Instead of saying "Do X," provide an analogy. 
"Translate this code like a translator would move a poem from French to English—preserving the soul (logic) but changing the syntax."

**Few-Shot CoT**: Providing 2 examples where the reasoning is explicit. This "primes" the model's attention to focus on logic rather than just mimicking the output string.

---

## Interview Questions

### Q: Why not just provide all 50 examples we have in the prompt?

**Strong answer:**
There are three primary reasons:
1. **Context Window Latency**: Every example adds tokens, increasing the "Prefill" time and the cost per request.
2. **Attention Dilution**: Even with 128k context, models can "lose" specific constraints if buried under too much irrelevant data (the "lost-in-the-middle" effect).
3. **Overfitting**: Providing too many narrow examples can cause the model to mimic the *format* of the examples too strictly, losing its general capability to handle edge cases outside that set.

### Q: What is "Label Bias" in In-Context Learning?

**Strong answer:**
Label bias occurs when the model predicts a specific label more frequently simply because it appeared more often in the few-shot examples or because it appeared at the end of the list. In 2025, we mitigate this by:
1. Shuffling the order of examples for different requests.
2. Ensuring an equal number of positive/negative/neutral samples.
3. Using "Permutation Testing" during prompt development to ensure the model responds to the content, not the order.

---

## References
- Brown et al. "Language Models are Few-Shot Learners" (2020)
- Min et al. "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" (2022)

---

*Next: [Chain-of-Thought](03-chain-of-thought.md)*

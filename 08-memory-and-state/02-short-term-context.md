# Short-Term Context Management (Dec 2025)

Short-term context (L1 Memory) is the high-speed interface where reasoning happens. In late 2025, managing this context is no longer about "message lists," but about **KV Cache Optimization** and **Dynamic Context Allocation**.

## Table of Contents

- [The Context Lifecycle](#lifecycle)
- [KV Cache Tiling and PagedAttention](#paged-attention)
- [Prefix Caching (System Prompt Preservation)](#prefix-caching)
- [Sliding Windows vs. Summarization](#sliding-vs-summary)
- [Contextual Compression (Selective Dropping)](#compression)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Context Lifecycle

Context goes through three stages:
1. **Intake**: User query + recent history + system instructions.
2. **Processing**: The GPU computes the KV cache for the new tokens.
3. **Eviction**: Removing old tokens to make room for new ones once the limit is reached.

---

## KV Cache Tiling (Dec 2025 Tech)

Modern inference engines (vLLM, TensorRT-LLM) use **PagedAttention**.
- **Idea**: Instead of allocating a contiguous block of GPU memory for the context, memory is broken into **Blocks** (pages).
- **Efficiency**: Reduces memory fragmentation by **60-80%**, allowing for significantly larger batch sizes and longer context windows on the same hardware.

---

## Prefix Caching

This is the **Holy Grail of Latency** in 2025.
- **The Problem**: Every time an agent calls an LLM, it sends the same 2,000-token System Prompt + 50 Tool Schemas. This wastes compute.
- **The Solution**: **Persistent Prefix Caching**. The server keeps the KV cache for the "Static" part of the prompt (the prefix) in memory.
- **Result**: You only pay for (and wait for) the compute on the *new* part of the message.

---

## Sliding Windows vs. Summarization

| Method | Mechanics | Pro | Con |
|--------|-----------|-----|-----|
| **Sliding Window** | Keep last N tokens exactly. | High fidelity for recent. | "Dory" effect (forgets start). |
| **Summarization** | Compress old turns into text. | Preserves "Key Facts". | Loses nuance/formatting. |
| **Hybrid (2025)** | Keep last 10 turns + 1 summary. | Best of both worlds. | Slightly higher complexity. |

---

## Contextual Compression

Late 2025 models support **Prompt Hardening**.
- **Selective Dropping**: Automatically stripping out irrelevant "Thought" blocks from previous turns to save space.
- **Token Pruning**: Using a smaller model to rewrite a long user message into a 50% shorter, equivalent prompt before sending it to the "Reasoning" model.

---

## Interview Questions

### Q: What is the difference between "Model Context Window" and "Application Context Window"?

**Strong answer:**
The **Model Context Window** is the hard limit defined by the architecture (e.g., 128K for GPT-4o). The **Application Context Window** is a configuration set by the engineer (e.g., 16K limit) to manage **Latency and Cost**. In production, we rarely use the full model window for every turn because attention overhead increases with context size, leading to slower generation. We use a **Buffer Zone** to leave space for the model's new response.

### Q: How does "Prefix Caching" change how you design System Prompts?

**Strong answer:**
It forces me to move **Static content to the front** and **Dynamic content to the back**. In 2023, we often put the user's name or date at the very top. In 2025, that breaks the prefix cache. I now put the "Immutable Rules" and "Tool Schemas" at the beginning and the "User Context" (which changes every turn) at the end. This ensures the first 5,000 tokens are identical across all users, maximizing cache hits on the inference server.

---

## References
- vLLM Team. "PagedAttention: Software-Defined Memory for LLM Serving" (2024/2025)
- NVIDIA. "Optimizing Inference with TensorRT-LLM" (2025)
- Anthropic. "Prompt Caching: Scale while reducing costs" (2024/2025)

---

*Next: [Long-Term Memory](03-long-term-memory.md)*

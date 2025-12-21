# Context Engineering

Context engineering is the science of filling the LLM's finite "working memory" with the most valuable tokens. In 2025, with context windows reaching 2M+ tokens, the focus has shifted from "fitting data" to "ranking relevance."

## Table of Contents

- [The Long Context Paradigm (2M+ Tokens)](#long-context)
- [Lost-in-the-Middle (2025 Update)](#lost-in-the-middle)
- [Context Budgeting & Token Awareness](#budgeting)
- [Prompt Caching Economics](#prompt-caching)
- [Contextual Compression (RAD-L)](#compression)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Long Context Paradigm (2M+ Tokens)

Models like Gemini 1.5 Pro and Claude Sonnet 4.5 have multi-million token context windows. 

**2025 Insight**: "Context is the new RAG." 
For datasets under 100,000 documents, it is often more accurate and faster to put the entire dataset in the context window than to use an external vector database. This is called **"In-Context RAG."**

---

## Lost-in-the-Middle (2025 Update)

In 2023, models lost accuracy for information in the middle of the prompt. 
**The 2025 Status**: Frontier models are significantly better, but the **Attention Gradient** still exists.
- **The Best Practice**: Place critical instructions and gold-standard examples at the **very beginning** and the **very end** of your prompt. The middle should contain the "raw" data or knowledge chunks.

---

## Context Budgeting & Token Awareness

Every token costs money and increases TTFT (Time to First Token).

| Component | Budget (Tokens) | Why? |
|-----------|-----------------|------|
| **System Prompt** | 500 - 1,000 | Core logic and persona. |
| **History** | 2,000 - 5,000 | Conversational "State." |
| **Data/Search** | 10k - 1M | Depends on task depth. |
| **Output Reserve**| 1,000 - 4,000 | Must reserve space for reasoning. |

---

## Prompt Caching Economics

In late 2025, almost all major providers (OpenAI, DeepSeek, Anthropic) support **Prefix Caching**.

- **The Crossover**: If you reuse a 100k token context (e.g., a codebase) for more than 2 requests, the caching discount effectively makes it cheaper than RAG.
- **Cache Hits**: $0.05 / 1M tokens.
- **Cache Misses**: $5.00 / 1M tokens.

**The Architectural Choice**: Design your system to keep the "System Prompt + Base Knowledge" static to maintain a 100% cache hit rate.

---

## Contextual Compression (RAD-L)

For extremely long contexts (10M+), we use **Reasoning-Aware Deletion (RAD-L)**.
- **How**: A tiny auxiliary model (0.1B) scans the text and removes "filler" words, common linguistic patterns, and irrelevant sections *before* the prompt is sent to the giant frontier model.
- **Benefit**: Reduces prompt size by 20-50% with <1% drop in accuracy.

---

## Interview Questions

### Q: When would you choose Long Context over RAG in 2025?

**Strong answer:**
I choose Long Context when high-fidelity retrieval and cross-document reasoning are critical. RAG suffers from "Retrieval Gap"—if your vector search misses the relevant chunk, the model never sees it. Long Context (up to 2M tokens) provides 100% recall. Specifically, I'd use it for codebase analysis, legal document review, and multi-file financial auditing. I'd stick to RAG for dynamic web-scale data or billion-document datasets that exceed any context window.

### Q: How do you handle the high TTFT associated with million-token prompts?

**Strong answer:**
The primary solution is **Context Caching**. By caching the heavy document on the GPU cluster, the model doesn't have to "re-read" (prefill) the entire 1M tokens for every turn. The TTFT for a cached prompt is nearly the same as for a 1k token prompt. Additionally, for non-cached requests, I would use **Streaming Prefill**, where the model generates an initial summary or "Thought" while it is still processing the latter half of the massive context.

---

## References
- Liu et al. "Lost in the Middle" (2023/2024 update)
- DeepSeek. "Prompt Caching in Production" (2025)
- Google. "Gemini 1.5: Unlocking Multi-modal Context" (2024)

---

*Next: [Structured Generation](06-structured-generation.md)*

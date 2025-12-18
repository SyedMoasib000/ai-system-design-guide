# Batching Strategies

Batching is the primary lever for increasing LLM throughput and reducing cost. In 2025, serving frameworks have moved beyond simple request-level batching to sub-token, iteration-level orchestration.

## Table of Contents

- [Static vs. Dynamic Batching](#static-vs-dynamic)
- [Continuous Batching (The 2025 Standard)](#continuous-batching)
- [In-Flight Batching (Prefill-Decode Fusion)](#in-flight-batching)
- [Chunked Prefill & RAD-O](#chunked-prefill)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Static vs. Dynamic Batching

In traditional ML (Classification), we use **Static Batching** where all requests must be the same size and start/end together. This is inefficient for LLMs due to variable response lengths.

---

## Continuous Batching (Iteration-level)

Continuous batching (pioneered by Orca and vLLM) allows new requests to join the batch and finished requests to leave at the end of every individual token generation step.

| Aspect | Static Batching | Continuous Batching |
|--------|-----------------|---------------------|
| **Join/Leave** | Only at start/end | Any iteration |
| **GPU Utilization**| Low (waiting for longest) | High (always saturated) |
| **Throughput** | 1x | **4x - 10x** |
| **Latency** | Highest for shortest | Balanced |

---

## In-Flight Batching (Prefill-Decode Fusion)

Previously, serving engines processed a batch of "Prefill" (heavy compute) OR a batch of "Decode" (heavy memory). 
**In-Flight Batching** (TensorRT-LLM) allows mixing them:
- 1 request is in the Prefill phase.
- 15 requests are in the Decode phase.
- **Benefit**: The Prefill request utilizes the GPU's idle compute cores while the Decode requests utilize the memory bandwidth.

---

## Chunked Prefill & RAD-O (Dec 2025)

Massive context prompts (1M+ tokens) can hang a batch for seconds during the Prefill phase, causing "stalls."

**The 2025 Fix: Chunked Prefill**
Instead of prefilling 128k tokens at once, the engine breaks the prefill into smaller chunks (e.g., 4k tokens each) and interleaves them with the ongoing Decode steps of other users. This maintains a steady **TPOT** even when heavy requests arrive.

---

## Interview Questions

### Q: Why is Continuous Batching superior to Static Batching for LLMs?

**Strong answer:**
Static batching forces all requests in a batch to wait for the longest generation to complete (the "longest tail" problem). If one user asks for 500 tokens and another for 5 tokens, the GPU remains idle for the 5-token user for 495 cycles. Continuous batching allows the 5-token user's request to exit the GPU immediately after its last token, freeing up VRAM and compute slots for a new request from the queue. This maximizes "Tokens per Second" across the entire hardware cluster.

### Q: What is a "stall" in LLM serving, and how does Chunked Prefill mitigate it?

**Strong answer:**
A "stall" occurs when a massive new request arrives and its Prefill phase (which is compute-hungry) takes 2-3 seconds to complete. During this time, the GPU is so busy with the prefill that it cannot generate tokens for existing users in the "Decode" phase, causing their TPOT to spike. Chunked Prefill breaks that 3-second prefill into small 200ms "chunks," processing one chunk and then doing one round of decoding for everyone else, before returning to the next prefill chunk. This ensures a consistent, smooth experience for all users.

---

## References
- Yu et al. "Orca: A Distributed Serving System for [Transformer] Models" (2022)
- NVIDIA. "TensorRT-LLM: In-Flight Batching" (2023)
- vLLM Project. "Iteration-Level Scheduling" (2023)

---

*Next: [PagedAttention](05-paged-attention.md)*

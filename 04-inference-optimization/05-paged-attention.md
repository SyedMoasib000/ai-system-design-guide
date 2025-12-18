# PagedAttention

PagedAttention is the foundational algorithm behind high-throughput serving engines in 2025. It solves the "Memory Fragmentation" problem that previously limited LLM scalability.

## Table of Contents

- [The Contiguous Memory Problem](#contiguous-memory)
- [How PagedAttention Works](#how-it-works)
- [Managing Virtual Memory (Block Manager)](#block-manager)
- [KV Cache Sharing (Copy-on-Write)](#sharing)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Contiguous Memory Problem

Standard deep learning frameworks allocate memory in large, contiguous blocks. 
For an LLM request, you might pre-allocate memory for a `max_sequence_length` of 8192 tokens.

**The Waste:**
1. **Internal Fragmentation**: If the user only generates 10 tokens, 99.9% of that reserved block is wasted.
2. **External Fragmentation**: Memory is broken into gaps too small for a new "large block," even if total free memory is high.

---

## How PagedAttention Works (vLLM)

PagedAttention draws inspiration from Virtual Memory in Operating Systems.

1. **Tokens to Blocks**: The KV cache for a request is broken into small, fixed-size **Blocks** (e.g., 16 tokens per block).
2. **Logical vs. Physical**: The model thinks it's attending to a contiguous sequence (Logical memory), but the blocks are scattered throughout VRAM (Physical memory).
3. **The Lookup Table**: A **Block Table** maps logic indices to physical addresses.

**Primary Benefit**: Memory waste drops from ~60-80% down to **less than 4%**.

---

## Managing Virtual Memory (Block Manager)

 Serving frameworks in 2025 (vLLM, SGLang) act as "mini-OSs" for GPUs.

- **Allocation**: When a new request starts, the Block Manager assigns it a set of empty physical blocks.
- **Eviction**: If VRAM is full, the manager can "swap" inactive KV blocks to CPU RAM and bring them back when needed (Paged Swap).

---

## KV Cache Sharing (Copy-on-Write)

PagedAttention enables effortless sharing of "Common Prefixes."

**The Scenario**: 100 users are chatting with the same 5,000-token system prompt.
- **Traditional**: Store that 5,000-token KV cache 100 times (**500k tokens** in VRAM).
- **PagedAttention**: Store it **once** via the Block Table and have all 100 users point to the same physical blocks.
- **Copy-on-Write**: If a user generates a unique token, a new block is created just for them, while the shared blocks remain unchanged.

---

## Interview Questions

### Q: Why does PagedAttention significantly increase throughput?

**Strong answer:**
PagedAttention increases throughput by allowing for much larger **batch sizes**. Because it eliminates internal and external memory fragmentation, we can pack many more requests into the same GPU VRAM. In traditional serving, we might only fit 4 requests because we have to "reserve" max-length blocks; with PagedAttention, we can fit 20-30 requests because we only use memory for the tokens that actually exist. Larger batches lead to better GPU utilization and significantly higher aggregate tokens per second.

### Q: Explain the "Block Table" in the context of vLLM.

**Strong answer:**
The Block Table is a mapping structure that bridges the Gap between the model's expectation of contiguous data and the physical reality of scattered memory. Each entry in the table corresponds to a "Logical Block" of tokens. It stores the physical address of the GPU memory where that block's key and value tensors are stored. This allows the framework to dynamically allocate and free memory in small chunks, enabling advanced features like prefix sharing and efficient multi-threading.

---

## References
- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- vLLM Documentation. "PagedAttention Logic" (2024)

---

*Next: [Serving Infrastructure](06-serving-infrastructure.md)*

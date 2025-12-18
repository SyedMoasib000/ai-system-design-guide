# Inference Fundamentals

Inference is the process of generating predictions from a trained model. In late 2025, inference optimization has shifted from "simple speedups" to "architectural efficiency" to handle reasoning-heavy workloads.

## Table of Contents

- [The Two Phases of Inference](#two-phases)
- [Bottlenecks: Compute-Bound vs. Memory-Bound](#bottlenecks)
- [Performance Metrics: TTFT and TPOT](#metrics)
- [Hardware-Enabled Optimizations (FP8)](#hardware-optimizations)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Two Phases of Inference

LLM inference is not a single operation; it consists of two distinct computational phases.

### 1. The Prefill Phase (Prompt Processing)
The model processes the entire input prompt in a single pass.
- **Computation**: High-parallelism matrix multiplications.
- **Bottleneck**: **Compute-bound** (limited by GPU TFLOPS).
- **Time Complexity**: $O(N)$ where $N$ is input length (but parallelized).

### 2. The Decode Phase (Token Generation)
The model generates tokens one by one, where each token depends on the previous ones.
- **Computation**: Sequential processing, one row of the weight matrix at a time.
- **Bottleneck**: **Memory-bound** (limited by memory bandwidth).
- **Time Complexity**: $O(M)$ where $M$ is output length (sequential).

---

## Bottlenecks: Compute-Bound vs. Memory-Bound

Understanding where your system is bottlenecked is critical for choosing the right optimization.

| Phase | Bottleneck | Why? | Primary Optimization |
|-------|------------|------|----------------------|
| **Prefill** | Compute (FLOPs) | Parallel processing saturates the GPU's arithmetic units. | FlashAttention, FP8/FP16 precision. |
| **Decode** | Memory Bandwidth | Weights must be loaded from VRAM for *every single token*. | Quantization (4-bit), GQA, Batching. |

**The 2025 Insight: Memory Wall**
As models grow larger, memory bandwidth (HBM3) has not scaled as fast as compute (TFLOPS). This makes the Decode phase the primary target for production optimization.

---

## Performance Metrics (Dec 2025)

| Metric | Full Form | Goal | Importance |
|--------|-----------|------|------------|
| **TTFT** | Time To First Token | < 200ms | User-perceived responsiveness. |
| **TPOT** | Time Per Output Token | < 30ms | Reading speed and conversational flow. |
| **Throughput** | Tokens/Second (Agg) | Maximize | Determining cost per query. |
| **Latency** | End-to-End Time | < 2.0s | Total turn-around for the agent. |

---

## Hardware-Enabled Optimizations (FP8)

In late 2025, **FP8 (8-bit Floating Point)** is the native precision for inference on H100 and B200 GPUs.

- **Benefit**: 2x faster than FP16/BF16 with negligible (<0.1%) accuracy loss.
- **How it works**: Uses a smaller mantissa and larger exponent than Int8, allowing it to represent the dynamic range of LLM activations more accurately without complex calibration.

**Principal-level Nuance**: Serving frameworks now use **Dynamic FP8 Scaling**, which adjust the quantization scales per-layer to prevent outliers from degrading the entire model's logic.

---

## Interview Questions

### Q: Why is LLM generation slower than classification?

**Strong answer:**
Classification is a "Prefill-only" task; it processes the entire input and produces a single output in one parallel pass, making it compute-optimal. LLM generation, however, is **auto-regressive**. Each token depends on the previous one, forcing a sequential "Decode" loop. Because each step in this loop is memory-bound (loading Gigabytes of weights to produce Milligrams of data), the system spends most of its time waiting for memory transfers rather than doing math.

### Q: How do you optimize TTFT vs. TPOT?

**Strong answer:**
To optimize **TTFT**, you must optimize the Prefill phase: use FlashAttention-3, increase compute parallelism (Tensor Parallelism), or use Prefix Caching to skip the prefill entirely for common prompts. 
To optimize **TPOT**, you must optimize the Memory Bandwidth during Decode: use quantization (4-bit weights) to reduce the data moved from VRAM, use Grouped Query Attention (GQA) to reduce KV cache size, or use Speculative Decoding to generate multiple tokens per memory load.

---

## References
- Pope et al. "Efficiently Scaling Transformer Inference" (2022)
- NVIDIA. "Transformer Engine Documentation" (2024)
- vLLM Blog. "Understanding LLM Inference Latency" (2023)

---

*Next: [KV Cache and Context Caching](02-kv-cache-and-context-caching.md)*

# Quantization Deep Dive

Quantization is the process of reducing the precision of model weights (e.g., from 16-bit to 4-bit) to save memory and increase inference speed. In 2025, this is the primary tool for deploying large models on consumer hardware.

## Table of Contents

- [The Precision-Performance Tradeoff](#precision-performance)
- [Quantization Methods (NF4, GPTQ, AWQ)](#methods)
- [GGUF vs. EXL2 (Late 2025 Standard)](#formats)
- [KV Cache Quantization (The VRAM Saver)](#kv-cache)
- [Quantization-Aware Fine-Tuning](#qaft)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Precision-Performance Tradeoff

Traditional models use **BF16** (16-bit). Quantization seeks to reduce this to **8-bit (FP8)**, **4-bit (Int4/NF4)**, or even **1.5-bit (BitNet)**.

| Precision | Bits | Weight size (8B Model) | Quality Loss | GPU Compatibility |
|-----------|------|------------------------|--------------|-------------------|
| **BF16** | 16 | 16 GB | 0% (Baseline) | All Modern |
| **FP8** | 8 | 8 GB | < 1% | H100 / B200 / RTX 4090 |
| **4-bit (NF4)**| 4 | 5 GB | 1-2% | All Modern |
| **2-bit** | 2 | 2.5 GB | 10-15% | Research / Specialized |

---

## Quantization Methods (Dec 2025)

### 1. NF4 (NormalFloat4)
The gold standard for fine-tuning (QLoRA). It assumes weights follow a normal distribution and maps them to a set of 16 values.

### 2. AWQ (Activation-aware Weight Quantization)
Instead of quantizing all weights equally, AWQ identifies the **1% of "salient" weights** that are most important for quality and keeps them in higher precision.
- **Pro**: Better accuracy than GPTQ.

### 3. FP8 (The 2025 Multi-Node Standard)
Hardware-native quantization supported by Nvidia's Transformer Engine.
- **Why it wins**: It provides the speed of Int8 but with the dynamic range of Float16, making it stable for both training and inference.

---

## GGUF vs. EXL2

### GGUF (llama.cpp)
- **Deployment**: CPU + GPU offloading. 
- **Pros**: Cross-platform (Mac, Linux, Windows), single file, highly portable.
- **Cons**: Slower than pure GPU formats.

### EXL2 (ExLlamaV2)
- **Deployment**: GPU-only (Nvidia).
- **Pros**: The **fastest 4-bit format in 2025**. Significant performance boost over AutoGPTQ/AWQ.
- **Cons**: Inflexible (Nvidia only).

---

## KV Cache Quantization (The VRAM Saver)

In long-context RAG (1M+ tokens), the **KV Cache** often consumes more VRAM than the model weights themselves.

- **BF16 KV Cache**: 2M tokens ≈ 32GB VRAM (on 8B model).
- **FP8/Int4 KV Cache**: 2M tokens ≈ 8GB - 16GB VRAM.

**2025 Nuance**: Modern serving frameworks (vLLM, SGLang) now support **Streaming Quantization** where the KV cache is compressed on-the-fly, allowing 4x higher concurrency on the same GPU.

---

## Quantization-Aware Training (QAT)

Instead of quantizing a model *after* it's trained (Post-training Quantization), QAT simulates quantization *during* the training process.
- **Result**: The model learns to compensate for the lost precision.
- **Status in 2025**: Mandatory for models smaller than 3B parameters to remain useful.

---

## Interview Questions

### Q: Why do we use NF4 instead of standard Float4 for QLoRA?

**Strong answer:**
Standard Float4 has a fixed grid that doesn't map well to the actual distribution of LLM weights, which typically follow a zero-centered normal distribution. NF4 (NormalFloat4) is a data type that is mathematically optimized so that each quantization bin contains an equal number of values from the normal distribution. This prevents "clustering" of weights and ensures that the model preserves as much information (entropy) as possible, leading to significantly higher accuracy than standard 4-bit integers.

### Q: How does AWQ differ from GPTQ?

**Strong answer:**
GPTQ is a "Layer-wise" quantization method that minimizes the mean squared error of the weights. AWQ (Activation-aware Weight Quantization) is "input-aware." It identifies which weights are the most "salient" based on the actual activation values seen during a small calibration run. By preserving only these important weights (usually 1%) in higher precision and quantizing the rest, AWQ achieves better perplexity than GPTQ, especially for smaller models or more aggressive quantization (e.g., 3-bit).

---

## References
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)

---

*Next: [Inference Fundamentals](../04-inference-optimization/01-inference-fundamentals.md)*

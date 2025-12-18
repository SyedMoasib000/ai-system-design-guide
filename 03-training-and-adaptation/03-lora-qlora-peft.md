# LoRA, QLoRA, and PEFT

Parameter-Efficient Fine-Tuning (PEFT) has become the industry standard for adapting LLMs in 2025. This chapter covers the mechanics and advanced variants of LoRA and other PEFT methods.

## Table of Contents

- [The PEFT Revolution](#the-peft-revolution)
- [LoRA Mechanics](#lora-mechanics)
- [QLoRA: 4-bit Fine-Tuning](#qlora)
- [Advanced Variants (DoRA, Vera, RS-LoRA)](#advanced-variants)
- [Multi-LoRA Serving (Adapters)](#multi-lora-serving)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The PEFT Revolution

In 2025, full fine-tuning of frontier models (GPT-4+, Llama 4 405B) is economically unfeasible for most enterprises. PEFT allows:
1. **Memory Efficiency**: Train 70B models on a single A100.
2. **Speed**: 2x faster training by updating <1% of weights.
3. **Modularity**: Swap "skills" (adapters) onto a shared base model without reloading weights.

---

## LoRA Mechanics

LoRA (Low-Rank Adaptation) injects trainable rank-decomposition matrices into the transformer layers.

```python
# The LoRA Equation for a Weight Matrix W:
h = Wx + (BA)x * (alpha/r)
```
- **W**: Pretrained weights (Frozen, Gradient = None)
- **A, B**: LoRA adapters (Trainable)
- **r**: Rank (e.g., 8, 16, 64)
- **alpha**: Scaling factor (typically 2 * rank)

### 2025 Principal Nuance: Target Modules
Historically, we only targeted query/value projections (`q_proj`, `v_proj`).
**The 2025 Standard**: Target **all** linear layers (`q, k, v, o, gate, up, down`) for maximum stability and performance, even at lower ranks.

---

## QLoRA: 4-bit Fine-Tuning

QLoRA pushes efficiency further by quantizing the base model to 4-bit (NF4) while maintaining 16-bit gradients.

| Optimization | Method | Benefit |
|--------------|--------|---------|
| **NF4 Quantization** | Normalized Float 4 | Better info density than standard Int4 |
| **Double Quant** | Quantizing the quant constants | Saves ~0.5 GB VRAM per model |
| **Paging** | Unified Memory (Nvidia) | Prevents OOM by spilling to CPU RAM |

---

## Advanced Variants (Dec 2025)

### 1. DoRA (Weight-Decomposed Low-Rank Adaptation)
DoRA decomposes the weight update into **Magnitude** and **Direction**.
- **Result**: Learns 2x faster than LoRA and performs closer to full fine-tuning.
- **Why it wins**: It allows the model to adjust how much it changes vs. what it's changing independently.

### 2. Vera (Vector-based Random Aggregation)
Instead of low-rank matrices `A` and `B`, Vera uses fixed random projections with a small trainable vector.
- **Efficiency**: Reduces adapter size by **10x** compared to LoRA.
- **Use Case**: Massive-scale Multi-LoRA serving.

### 3. RS-LoRA (Rank-Stabilized LoRA)
Uses a scaling factor of `alpha / sqrt(r)`.
- **Benefit**: Allows you to increase rank (to 256+) without the model becoming unstable or requiring a lower learning rate.

---

## Multi-LoRA Serving (Adapters)

In 2025, production systems serve one base model (e.g., Llama 4 70B) and dynamically swap adapters in the same batch.

```python
# vLLM/LMCache Multi-LoRA Pattern:
# Request 1 -> Base + Finance_Adapter
# Request 2 -> Base + Legal_Adapter
# Request 3 -> Base + Medical_Adapter
```
**The Tech**: **Continuous Batching + PagedAttention v3** allows serving 100+ adapters with only a 5-10% latency overhead compared to the base model.

---

## Interview Questions

### Q: Why is the LoRA alpha parameter usually set to 2x the rank?

**Strong answer:**
The `alpha` parameter is a scaling factor for the LoRA update. When we initialize LoRA matrices, B is usually zero-initialized, and A is random. As we train, the update size depends on the rank `r`. By setting `alpha=2r` (or any constant), we ensure that if we decide to change the rank later (e.g., from 8 to 16), we don't need to retune the learning rate. The scaling factor `alpha/r` normalizes the update magnitude relative to the learning rate.

### Q: What is DoRA, and why would you use it over standard LoRA?

**Strong answer:**
DoRA (Weight-Decomposed Low-Rank Adaptation) is a 2024 technique that separates the pretrained weight updates into magnitude and direction components, similar to Weight Normalization. While standard LoRA updates magnitude and direction simultaneously, DoRA allows them to be learned independently. Empirically, DoRA shows much better convergence and higher accuracy, often matching full-parameter fine-tuning even at low ranks, making it the preferred choice for high-stakes domain adaptation in 2025.

---

## References
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Liu et al. "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

---

*Next: [RLHF and DPO](04-rlhf-and-dpo.md)*

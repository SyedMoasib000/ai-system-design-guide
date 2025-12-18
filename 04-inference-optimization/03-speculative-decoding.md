# Speculative Decoding

Speculative decoding is a 2024-2025 breakthrough that allows large Models (LLMs) to generate multiple tokens per forward pass, effectively breaking the memory-bandwidth bottleneck for sequential decoding.

## Table of Contents

- [The Core Concept](#the-core-concept)
- [Draft-Verify Paradigm](#draft-verify)
- [Medusa & Multi-Token Heads (Dec 2025)](#medusa)
- [Lookahead Decoding](#lookahead-decoding)
- [Hardware-Aware Speculation](#hardware-aware)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Core Concept

LLM decoding is memory-bound: loading 140GB of weights (70B model) to produce a single 2-byte token is inefficient. 
**Speculative Decoding** uses a cheaper method to "guess" the next $N$ tokens and uses the large model to verify them all in a single parallel "Prefill-style" pass.

---

## Draft-Verify Paradigm

1. **Drafting**: A small, fast "Draft Model" (e.g., 1B or 7B) generates $K$ candidate tokens.
2. **Verification**: The large "Target Model" processes all $K$ tokens at once.
3. **Acceptance**: The target model's logits are used to accept or reject candidates. If token $i$ is rejected, all tokens after it are discarded.

| Model | Size | Speed | Latency per token |
|-------|------|-------|-------------------|
| **Draft** | 1B | Fast | 5ms |
| **Target**| 70B| Slow | 50ms |
| **Speculative**| - | **Fast**| **15ms - 25ms** |

**Net Result**: 2x to 3x speedup in wall-clock time with **zero loss in quality**.

---

## Medusa & Multi-Token Heads (Dec 2025)

In late 2025, the industry is moving away from separate draft models (which add VRAM overhead) toward **Medusa Heads**.

- **What it is**: Extra "heads" (small linear layers) attached to the last layer of the target model.
- **How it works**: Instead of predicting just token $t+1$, Head 1 predicts $t+1$, Head 2 predicts $t+2$, and so on.
- **Benefit**: No second model needed; 2.5x speedup with minimal VRAM increase.

---

## Lookahead Decoding

A 2025 alternative that uses the model's own past hidden states to find recurring patterns (n-grams) to "look ahead" and predict future tokens.
- **Best For**: Structured data, code, and highly repetitive technical writing.

---

## Hardware-Aware Speculation

Frontier serving frameworks (vLLM, TensorRT-LLM) now use **Dynamic Draft Lengths**.
- If the GPU is underutilized (small batch), the system increases the number of draft tokens ($K$).
- If the GPU is saturated (large batch), it decreases $K$ to prioritize throughput over individual request latency.

---

## Interview Questions

### Q: Why doesn't Speculative Decoding work well for high-temperature creative writing?

**Strong answer:**
Speculative decoding relies on the "Draft Model" being able to accurately predict what the "Target Model" would say. In high-temperature creative writing, the probability distribution is "flatter," and the model is encouraged to pick less-likely tokens. This leads to a very low **Acceptance Rate** (the draft model's guesses are frequently rejected). When a guess is rejected, the target model's parallel pass was wasted compute, and the system falls back to standard sequential decoding, adding the overhead of the draft model's latency.

### Q: How does Medusa differ from traditional Speculative Decoding?

**Strong answer:**
Traditional speculative decoding requires a separate, smaller model (the Draft Model) which takes up extra VRAM and requires its own KV cache management. Medusa, instead, adds multiple "heads" to the base model's final hidden state. Each head is trained to predict a different offset (e.g., next token, next+1, next+2). This eliminates the need for a second model and minimizes the communication overhead between steps, as all "guesses" are generated within the same base model architecture during a single forward pass.

---

## References
- Chen et al. "Accelerating Transformer Decoding via Speculative Decoding" (2023)
- Cai et al. "Medusa: Simple LLM Acceleration via Multiple Decoding Heads" (2024)
- Fu et al. "Lookahead Decoding" (2024)

---

*Next: [Batching Strategies](04-batching-strategies.md)*

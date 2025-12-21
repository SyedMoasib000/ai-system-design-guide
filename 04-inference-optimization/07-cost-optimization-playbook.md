# Cost Optimization Playbook

In late 2025, AI costs are no longer "magic." They are measurable, predictable, and highly optimizable. This chapter covers the strategies to reduce inference costs by 10x without sacrificing quality.

## Table of Contents

- [The Unit Economics of AI](#unit-economics)
- [Model Cascading (Efficiency Tiers)](#model-cascading)
- [Small Language Models (SLMs)](#slms)
- [Spot Instance Strategies](#spot-instances)
- [The "Token Tax" Optimization](#token-tax)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Unit Economics of AI

In 2025, we measure success by **Tokens per Dollar ($)**.

| Component | Cost Driver | Optimization |
|-----------|-------------|--------------|
| **Compute** | GPU Time ($/hr) | Better utilization (Batching). |
| **VRAM** | KV Cache Size | GQA, Quantization. |
| **Network** | Payload Size | Compression, Local serving. |
| **API** | Per-token pricing | Caching, Model selection. |

---

## Model Cascading (Efficiency Tiers)

The most effective cost-saving strategy is to use the **cheapest model capable of the task.**

**The 2025 Cascade Pattern:**
1. **Classifier**: A tiny model (0.5B) determines query complexity ($0.00).
2. **Tier 1 (SLM)**: 90% of queries (greetings, simple Q&A) go to an 8B model ($).
3. **Tier 2 (Frontier)**: 9% of queries (complex reasoning) go to a 405B/GPT-5.2 model ($$$).
4. **Tier 3 (Reasoning)**: 1% of queries (expert-level) go to o1-style models ($$$$$).

**Net result**: 80% cost reduction vs. sending all traffic to Tier 2.

---

## Small Language Models (SLMs) for Production

In late 2025, 3B-8B models (Llama 4 8B, Gemini 3 Flash) are as capable as 2023's GPT-4.
- **Use Case**: Entity extraction, sentiment analysis, simple RAG.
- **Cost**: 100x cheaper to run than frontier models.
- **Latency**: < 100ms response times.

---

## Spot Instance Strategies (Dec 2025)

For non-real-time workloads (batch processing, data extraction), use **GPU Spot Instances** (AWS Spot, Azure Spot, Lambda Labs).

- **Risk**: GPU can be reclaimed with 30-sec notice.
- **The 2025 Mitigation**: **Live KV-Cache Migration**. Serving frameworks can now stream the KV cache of ongoing requests to another node as soon as the "Reclamation Signal" is received, ensuring no work is lost.

---

## The "Token Tax" Optimization

- **System Prompt Caching**: Hard-code common prefixes to get 90% discounts.
- **Output Truncation**: Strictly limit `max_tokens`.
- **Negative Prompting**: "Don't be wordy" saves ~15% in output tokens (and thus cost).

---

## Interview Questions

### Q: How do you justify the cost of an AI system to a CFO?

**Strong answer:**
I focus on the **ROI of Efficiency.** First, I implement "Model Cascading" to ensure that 90% of our traffic is handled by sub-cent-per-million-token models. Second, I implement "Semantic Caching" to prevent paying for the same answer twice. Third, I set up "Inference Quotas" and "Chargeback Models" so each business unit is accountable for their usage. By treating AI as a "Commodity Resource" with tiered pricing, we can transition from "unbounded experimentation" to a "predictable OpEx" model.

### Q: When is a self-hosted individual GPU cluster cheaper than an API?

**Strong answer:**
The "Crossover Point" usually happens at **constant high throughput.** If your application has a baseline of 5-10 requests per second, 24/7, the fixed cost of an H100 reservation becomes cheaper than the variable token cost of an API. However, if your traffic is "spiky" or heavily weighted toward business hours, API providers are usually cheaper because they allow you to "pay for the silence" during off-peak hours. In 2025, for most enterprises, the break-even is around 500 million tokens per month for a 70B-tier model.

---

## References
- Google Cloud. "Cost Optimization for Generative AI" (2024)
- Anyscale. "LLM Inference: API vs. Self-Hosted Costs" (2024)

---

*Next: [Prompt Engineering Fundamentals](../05-prompting-and-context/01-prompt-engineering-fundamentals.md)*

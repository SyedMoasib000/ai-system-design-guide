# Model Taxonomy

This chapter provides a comprehensive guide to the model landscape as of December 2025, covering model families, capabilities, and selection criteria for production systems.

## Table of Contents

- [Model Categories](#model-categories)
- [Frontier Models](#frontier-models)
- [Open Source Models](#open-source-models)
- [Specialized Models](#specialized-models)
- [Embedding Models](#embedding-models)
- [Model Selection Framework](#model-selection-framework)
- [Capability Comparison](#capability-comparison)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Model Categories

### By Capability Level

| Tier | Characteristics | Examples | Use Case |
|------|-----------------|----------|----------|
| Frontier | State-of-the-art, largest, most capable | GPT-4o, Claude 3.5 Opus, Gemini Ultra | Complex reasoning, high-stakes |
| Strong | Near-frontier, good balance | Claude 3.5 Sonnet, GPT-4o-mini, Gemini 1.5 Pro | Production workloads |
| Fast | Optimized for speed and cost | Claude 3.5 Haiku, GPT-4o-mini, Gemini 1.5 Flash | High-volume, low-latency |
| Small | Local deployment, edge | Llama 3 8B, Mistral 7B, Gemma 2B | Self-hosted, privacy |

### By Access Type

| Type | Characteristics | Examples |
|------|-----------------|----------|
| Proprietary API | Closed weights, API access | OpenAI, Anthropic, Google |
| Open Weights | Downloadable, restrictive license | Llama 3, Mistral |
| Fully Open | Open weights and training data | OLMo, Pythia |

### By Modality

| Type | Inputs | Outputs | Examples |
|------|--------|---------|----------|
| Text-only | Text | Text | GPT-4, Llama |
| Multimodal | Text + Images | Text | GPT-4V, Claude 3.5, Gemini |
| Vision-Language | Images | Text | LLaVA, BLIP-2 |
| Audio | Audio | Text/Audio | Whisper, Gemini |

---

## Frontier Models

### GPT-4o (OpenAI)

| Attribute | Value |
|-----------|-------|
| Context Window | 128K tokens |
| Input Cost | $2.50 / 1M tokens |
| Output Cost | $10 / 1M tokens |
| Multimodal | Text, Image, Audio |
| Strengths | General reasoning, code, instruction following |
| Released | May 2024 |

**Best for:** Complex reasoning, code generation, multimodal tasks
**Considerations:** Cost at scale, no self-hosting option

### Claude 3.5 Sonnet (Anthropic)

| Attribute | Value |
|-----------|-------|
| Context Window | 200K tokens |
| Input Cost | $3 / 1M tokens |
| Output Cost | $15 / 1M tokens |
| Multimodal | Text, Image |
| Strengths | Long context, coding, nuanced understanding |
| Released | June 2024 |

**Best for:** Code, long documents, nuanced tasks
**Considerations:** Slightly more conservative outputs

### Claude 3.5 Opus (Anthropic)

| Attribute | Value |
|-----------|-------|
| Context Window | 200K tokens |
| Input Cost | $15 / 1M tokens |
| Output Cost | $75 / 1M tokens |
| Multimodal | Text, Image |
| Strengths | Highest capability, complex tasks |
| Released | 2024 |

**Best for:** Most demanding tasks, research, complex analysis
**Considerations:** 5x cost of Sonnet

### Gemini 1.5 Pro (Google)

| Attribute | Value |
|-----------|-------|
| Context Window | 1M tokens (2M in preview) |
| Input Cost | $1.25 / 1M tokens |
| Output Cost | $5 / 1M tokens |
| Multimodal | Text, Image, Audio, Video |
| Strengths | Massive context, video understanding |
| Released | February 2024 |

**Best for:** Very long documents, video analysis
**Considerations:** Latency can be higher

### Model Comparison: Frontier Tier

| Model | Reasoning | Code | Long Context | Cost | Speed |
|-------|-----------|------|--------------|------|-------|
| GPT-4o | ★★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| Claude 3.5 Sonnet | ★★★★★ | ★★★★★ | ★★★★★ | ★★★ | ★★★★ |
| Gemini 1.5 Pro | ★★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★ |
| Claude 3.5 Opus | ★★★★★ | ★★★★★ | ★★★★★ | ★ | ★★★ |

---

## Open Source Models

### Llama 3 Family (Meta)

| Model | Parameters | Context | License | Notes |
|-------|------------|---------|---------|-------|
| Llama 3 8B | 8B | 8K | Llama License | Best small model |
| Llama 3 70B | 70B | 8K | Llama License | Strong open model |
| Llama 3.1 405B | 405B | 128K | Llama License | Frontier-competitive |

**Strengths:**
- Excellent quality-to-size ratio
- Large community and tooling
- Extensive fine-tuned variants

**License note:** Llama license allows commercial use with restrictions for large deployments (700M+ monthly users).

### Mistral Family

| Model | Parameters | Context | License | Notes |
|-------|------------|---------|---------|-------|
| Mistral 7B | 7B | 32K | Apache 2.0 | Sliding window attention |
| Mixtral 8x7B | 47B (12B active) | 32K | Apache 2.0 | MoE architecture |
| Mistral Large | API | 128K | Proprietary | Via API only |

**Strengths:**
- Efficient architecture (GQA, sliding window)
- Apache 2.0 license for base models
- Strong for size

### Qwen Family (Alibaba)

| Model | Parameters | Context | License | Notes |
|-------|------------|---------|---------|-------|
| Qwen 2.5 7B | 7B | 128K | Apache 2.0 | Excellent multilingual |
| Qwen 2.5 72B | 72B | 128K | Apache 2.0 | Near-frontier quality |

**Strengths:**
- Excellent multilingual (especially CJK)
- Strong coding capability
- Permissive license

### Open Model Comparison

| Model | Quality | Inference Speed | License Freedom | Community |
|-------|---------|-----------------|-----------------|-----------|
| Llama 3.1 70B | ★★★★★ | ★★★ | ★★★★ | ★★★★★ |
| Mixtral 8x7B | ★★★★ | ★★★★ | ★★★★★ | ★★★★ |
| Qwen 2.5 72B | ★★★★★ | ★★★ | ★★★★★ | ★★★ |
| Mistral 7B | ★★★ | ★★★★★ | ★★★★★ | ★★★★ |

---

## Specialized Models

### Coding Models

| Model | Base | Specialization | Best For |
|-------|------|----------------|----------|
| Codestral | Mistral | Code generation | Full-stack coding |
| StarCoder2 | Open | Code completion | IDE integration |
| DeepSeek Coder | DeepSeek | Code, math | Algorithmic problems |
| CodeLlama | Llama 2 | Code | General coding |

### Math and Reasoning

| Model | Approach | Best For |
|-------|----------|----------|
| DeepSeek-Math | Specialized training | Mathematical reasoning |
| WizardMath | Fine-tuned | Word problems |
| Phi-3 | Small but capable | Resource-constrained reasoning |

### Long Context Models

| Model | Context | Notes |
|-------|---------|-------|
| Gemini 1.5 Pro | 1M | Production, video support |
| Claude 3.5 | 200K | Reliable long-context |
| GPT-4 Turbo | 128K | Good long-context |
| Llama 3.1 | 128K | Open source option |

---

## Embedding Models

### API Embedding Models

| Model | Dimensions | Max Tokens | MTEB | Cost/1M |
|-------|------------|------------|------|---------|
| OpenAI text-embedding-3-large | 3072 | 8191 | 64.6 | $0.13 |
| OpenAI text-embedding-3-small | 1536 | 8191 | 62.3 | $0.02 |
| Voyage-3 | 1024 | 32000 | 67.8 | $0.06 |
| Cohere embed-v3 | 1024 | 512 | 66.4 | $0.10 |
| Google text-embedding-004 | 768 | 2048 | 66.1 | $0.025 |

### Open Source Embedding Models

| Model | Dimensions | Max Tokens | MTEB | Notes |
|-------|------------|------------|------|-------|
| BGE-large-en-v1.5 | 1024 | 512 | 63.9 | Instruction-tuned |
| E5-large-v2 | 1024 | 512 | 62.4 | Microsoft |
| GTE-large | 1024 | 512 | 63.1 | Alibaba |
| Nomic-embed-text-v1.5 | 768 | 8192 | 62.3 | Long context, open |

### Embedding Selection Guide

| Requirement | Recommended | Why |
|-------------|-------------|-----|
| Best quality | Voyage-3 or text-embedding-3-large | Highest MTEB |
| Cost-efficient | text-embedding-3-small | $0.02/1M |
| Self-hosted | BGE-large or GTE-large | Open, high quality |
| Long documents | Nomic or Voyage-3 | 8K+ context |
| Multilingual | Cohere embed-v3 | Built for multilingual |

---

## Model Selection Framework

### Decision Tree

```
What is your primary constraint?

├── Cost → Use smaller model, consider open source
│   ├── Very cost sensitive → GPT-4o-mini, Claude Haiku
│   └── Moderate budget → Claude Sonnet, GPT-4o
│
├── Quality → Use frontier models
│   ├── Highest quality → Claude Opus, GPT-4o
│   └── Near-frontier → Claude Sonnet, Gemini Pro
│
├── Latency → Use fast models
│   ├── <100ms response → GPT-4o-mini, Claude Haiku
│   └── <500ms response → Any with good inference
│
├── Self-hosting → Use open models
│   ├── Maximum capability → Llama 3.1 405B, Qwen 72B
│   ├── Good balance → Llama 3.1 70B, Mixtral
│   └── Edge/mobile → Llama 3 8B, Mistral 7B, Phi-3
│
└── Privacy → Self-host or use on-prem
    └── Choose open models with appropriate license
```

### Cost Comparison at Scale

Assume 1M requests/day, 1K input + 500 output tokens average:

| Model | Input Cost/Day | Output Cost/Day | Total/Month |
|-------|----------------|-----------------|-------------|
| GPT-4o | $2,500 | $5,000 | $225,000 |
| Claude 3.5 Sonnet | $3,000 | $7,500 | $315,000 |
| GPT-4o-mini | $150 | $300 | $13,500 |
| Claude Haiku | $250 | $625 | $26,250 |
| Self-hosted 70B* | - | - | ~$50,000 |

*Self-hosted assumes 4x H100 GPUs for 70B model

### Quality vs Cost Matrix

```
                       High Quality
                            ▲
                            │
    Claude Opus ●           │           ● GPT-4o
                            │
    Claude Sonnet ●         │
                            │
                            │           ● Gemini Pro
    ─────────────────────────┼───────────────────────►
                            │                      Low Cost
    Mixtral ●               │
                            │
    Llama 70B ●             │           ● GPT-4o-mini
                            │
    Mistral 7B ●            │           ● Claude Haiku
                            │
                       Low Quality
```

---

## Capability Comparison

### Benchmark Performance (Approximate, December 2025)

| Model | MMLU | HumanEval | GSM8K | MT-Bench |
|-------|------|-----------|-------|----------|
| GPT-4o | 87.2 | 90.2 | 95.3 | 9.3 |
| Claude 3.5 Sonnet | 88.7 | 92.0 | 96.4 | 9.1 |
| Gemini 1.5 Pro | 85.9 | 84.1 | 92.5 | 8.9 |
| Llama 3.1 405B | 88.0 | 89.1 | 94.2 | 8.8 |
| Llama 3.1 70B | 82.0 | 80.5 | 88.8 | 8.2 |
| GPT-4o-mini | 82.0 | 87.0 | 93.2 | 8.5 |
| Claude Haiku | 75.2 | 75.9 | 88.0 | 8.1 |
| Mistral 7B | 62.5 | 26.9 | 52.2 | 6.8 |

*Benchmark scores are approximate and may vary. Always verify with official sources.*

### Task-Specific Recommendations

| Task | Recommended Models | Why |
|------|-------------------|-----|
| Complex reasoning | Claude Sonnet, GPT-4o | Highest capability |
| Code generation | Claude Sonnet, Codestral | Strong code understanding |
| Long document Q&A | Gemini 1.5 Pro, Claude | Large context windows |
| High-volume API | GPT-4o-mini, Claude Haiku | Cost and speed |
| Image understanding | GPT-4o, Gemini, Claude | Vision capability |
| Self-hosted production | Llama 3.1 70B | Quality-cost balance |
| Edge deployment | Mistral 7B, Llama 8B | Size constraints |
| Multilingual | Qwen 2.5, Claude, Gemini | Multilingual training |

---

## Interview Questions

### Q: How would you select a model for a production RAG system?

**Strong answer:**
I would evaluate models across these dimensions:

**1. Quality requirements:**
- Test on representative queries
- Measure answer correctness
- Evaluate instruction following

**2. Cost analysis:**
```
Monthly cost = requests/day * 30 * avg_tokens * rate
```
Calculate for top candidates and compare.

**3. Latency requirements:**
- If <200ms TTFT needed: Consider GPT-4o-mini, Claude Haiku
- If quality is paramount: Accept higher latency with frontier models

**4. Operational requirements:**
- Self-hosting: Choose open models
- Compliance: Consider on-prem options
- Multi-region: Check provider availability

**5. Practical selection:**
- Start with Claude Sonnet or GPT-4o for prototyping
- A/B test cheaper alternatives on production traffic
- Consider model routing for cost optimization

### Q: Explain the tradeoffs between proprietary and open source models.

**Strong answer:**
| Factor | Proprietary (OpenAI, Anthropic) | Open Source (Llama, Mistral) |
|--------|--------------------------------|------------------------------|
| Quality | Generally higher | Catching up rapidly |
| Cost | Per-token pricing | Compute + ops |
| Control | Limited | Full |
| Privacy | Data goes to provider | Stays on-prem |
| Updates | Automatic | Manual |
| Customization | Limited fine-tuning | Full fine-tuning |
| Ops overhead | None | Significant |
| Latency control | Limited | Full |

**When to choose proprietary:**
- Speed to production
- Limited ML ops capacity
- Need highest quality
- Willing to accept vendor dependency

**When to choose open source:**
- Data privacy requirements
- Cost sensitive at scale
- Need fine-tuning control
- Have ML infrastructure expertise

### Q: What is the difference between Llama, Mistral, and Qwen families?

**Strong answer:**
| Aspect | Llama (Meta) | Mistral | Qwen (Alibaba) |
|--------|-------------|---------|----------------|
| Origin | Meta AI | Mistral AI (France) | Alibaba |
| Key innovation | Scaling, efficiency | MoE, sliding window | Multilingual, long context |
| License | Llama License | Apache 2.0 (base) | Apache 2.0 |
| Sizes available | 8B, 70B, 405B | 7B, 8x7B, API | 7B, 14B, 72B |
| Best at | General, code | Efficiency, Europe | Multilingual, CJK |
| Community | Largest | Growing | Large in Asia |
| Commercial use | With restrictions | Free | Free |

**Selection heuristic:**
- Default choice: Llama (best community, tooling)
- Need efficiency or MoE: Mistral
- Multilingual or CJK: Qwen

---

## References

- OpenAI Platform: https://platform.openai.com/docs/models
- Anthropic Documentation: https://docs.anthropic.com/
- Google AI: https://ai.google.dev/
- Meta Llama: https://llama.meta.com/
- Mistral AI: https://mistral.ai/
- LMSYS Chatbot Arena: https://chat.lmsys.org/
- Hugging Face Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

---

*Next: [Capability Assessment](02-capability-assessment.md)*

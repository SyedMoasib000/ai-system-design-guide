# Model Taxonomy

This chapter provides a comprehensive guide to the model landscape as of December 2025, covering model families, capabilities, and selection criteria for production systems.

## Table of Contents

- [Model Categories](#model-categories)
- [Frontier Models](#frontier-models)
- [Open Source Models](#open-source-models)
- [Specialized Models](#specialized-models)
- [Embedding Models](#embedding-models)
- [Model Selection Framework & Semantic Routing](#model-selection-framework)
- [Sovereign AI & Data Residency](#sovereign-ai-and-data-residency)
- [Capability Comparison](#capability-comparison)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Model Categories

### By Capability Level

| Tier | Characteristics | Examples | Use Case |
|------|-----------------|----------|----------|
| **Ultra (New)** | Super-intelligent reasoning, agentic mastery | GPT-5.2, Claude 4.5 Opus, Gemini 3.0 Pro | Full autonomous agents, R&D |
| Frontier | State-of-the-art general intelligence | Claude Sonnet 4.5, GPT-4o, Gemini 2.0 Pro | Complex reasoning, production |
| Strong | High efficiency, near-frontier | DeepSeek V3.2, GPT-5.2-mini | Scalable business logic |
| Fast | Sub-100ms, cost-optimized | Gemini 3 Flash, Claude 4.5 Haiku, o4-mini | High-volume streaming, UI |
| Small | Private, edge, specialized | Llama 4 8B, Nemotron 3 Nano, Phi-4 | Local privacy, specific tasks |

### By Reasoning Mode (New for 2025)

| Mode | Capability | Models | Use Case |
|------|------------|--------|----------|
| **Standard** | Fast, intuitive response | GPT-4o, Claude 3.5 Sonnet | Chat, simple extraction |
| **Thinking** | Internal CoT before output | o1, Claude Sonnet 4.5 (Reasoning), DeepSeek-R1 | Math, code debugging |
| **Hybrid** | User-controllable reasoning depth | Claude Sonnet 4.5 | Variable complexity tasks |

---

## Frontier Models

### GPT-5.2 (OpenAI)

| Attribute | Value |
|-----------|-------|
| Context Window | 512K tokens (Native) |
| Input Cost | $5.00 / 1M tokens |
| Output Cost | $20 / 1M tokens |
| Multimodal | Native Omni (Text, Vision, Audio, Video) |
| Highlights | Agentic tool-calling, long-horizon planning |
| Released | December 2025 |

**Best for:** Professional knowledge work, long-running agents, complex planning.
**Considerations:** High cost per token; optimized for agentic workflows.

### Claude 4.5 Opus (Anthropic)

| Attribute | Value |
|-----------|-------|
| Context Window | 400K tokens |
| Input Cost | $15 / 1M tokens |
| Output Cost | $75 / 1M tokens |
| Multimodal | Text, Vision (SoTA layout analysis) |
| Highlights | Surpassed 80% on SWE-bench Verified |
| Released | November 2025 |

**Best for:** Autonomous software engineering, complex legal/medical analysis.
**Considerations:** Most expensive model in the market; benchmark leader.

### Claude Sonnet 4.5 (Anthropic)

| Attribute | Value |
|-----------|-------|
| Context Window | 200K tokens |
| Input Cost | $3 / 1M tokens |
| Output Cost | $15 / 1M tokens |
| Highlights | Hybrid Reasoning (toggle between fast/deep) |
| Released | September 2025 |

**Best for:** Coding, front-end web development, "Claude Code" agent.

### Gemini 3.0 Pro (Google)

| Attribute | Value |
|-----------|-------|
| Context Window | 2.5M tokens (Native) |
| Input Cost | $1.25 / 1M tokens |
| Output Cost | $5 / 1M tokens |
| Multimodal | Parallel processing of Video/Audio/Text |
| Highlights | Near human-like responsiveness in voice |
| Released | Late 2025 |

**Best for:** Massive codebase ingestion, real-time multimodal interaction.

### Model Comparison: Frontier Tier (Dec 2025)

| Model | Reasoning | Coding | Context | Agentic | Cost |
|-------|-----------|--------|---------|---------|------|
| GPT-5.2 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★★ | $$$ |
| Claude 4.5 Opus | ★★★★★ | ★★★★★ | ★★★★ | ★★★★★ | $$$$$ |
| Gemini 3.0 Pro | ★★★★ | ★★★★ | ★★★★★ | ★★★★ | $$ |
| Claude Sonnet 4.5 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | $$$ |

### Production Heritage & Maturity

While the "Ultra" tier (GPT-5.2, Claude 4.5) represents the bleeding edge, many enterprise systems remain on the **Battle-Tested Frontier** models for standard production workloads.

| Model Family | Production Since | Maturity Note |
|--------------|------------------|---------------|
| **GPT-4o** | May 2024 | Most mature ecosystem; lowest latency variance; highest rate limits. |
| **Claude 3.5 Sonnet** | June 2024 | Gold standard for tool-use reliability and structured output in 2024-2025. |
| **Gemini 1.5 Pro** | May 2024 | Pioneer of mass-scale context; highly stable for multi-hour video analysis. |
| **o1-preview** | Sept 2024 | First to prove "Inference-time Scaling" in production at scale. |

**Why stay on "Older" Frontier models?**
1. **Consistency**: New models often have "release-window" latency spikes and behavior shifts.
2. **Cost Efficiency**: Providers often discount the previous generation significantly as new ones launch.
3. **Guardrail Tuning**: Security and moderation layers for 4o/3.5 are significantly more refined and "broken-in."

---

## Open Source Models

### Llama 4 Family (Meta)

| Model | Parameters | Context | License | Notes |
|-------|------------|---------|---------|-------|
| Llama 4 8B | 8B | 128K | Llama 4 | Best-in-class small model |
| Llama 4 70B | 70B | 128K | Llama 4 | Strongest open-weight reasoning |
| Llama 4 405B | 405B | 256K | Llama 4 | Frontier-omni competitive |

**Strengths:**
- Native multimodality (Text/Vision) across all sizes
- Significantly improved tool-use and JSON follow
- 128K context window is now the standard minimum

### Nemotron 3 Family (Nvidia)

| Model | Parameters | Architecture | Notes |
|-------|------------|--------------|-------|
| Nemotron 3 Ultra | 500B | MoE | Optimized for agentic systems |
| Nemotron 3 Super | 120B | MoE | High throughput, low latency |
| Nemotron 3 Nano | 4B | Dense | On-device agentic tasks |

**Strengths:**
- Highest throughput for agentic loops
- Integrated "SteerLM" for real-time tone adjustment

### DeepSeek Family

| Model | Parameters | Status | Notes |
|-------|------------|--------|-------|
| DeepSeek-V3.2 | 671B (MoE) | Frontier | GPT-4o level performance at 1/5 cost |
| DeepSeek-R1 | 671B | Reasoning | Competitive with o1/Claude Sonnet 4.5 |
| DeepSeek-V3.2-Spec | 671B | Ultra | Heavyweight logic focus |

---

## Specialized Models

### Coding Mastery (late 2025)

| Model | Specialization | Why it wins |
|-------|----------------|-------------|
| **Claude 4.5 Opus** | Software Engineering | Highest SWE-bench Verified score (80.9%) |
| **GPT-5.2** | Agentic Coding | Best at architectural refactoring |
| **DeepSeek-Coder-V3** | Algorithmic code | Best price-to-performance for IDEs |

### Reasoning & Math

| Model | Approach | Best For |
|-------|----------|----------|
| **OpenAI o3** | High-compute reasoning | Autonomous tool use in complex envs |
| **DeepSeek-R1** | RL-based thinking | Logical inference, competitive math |
| **Nemotron-Reasoning** | Open reasoning | Self-hosted logical tasks |

### Long Context (2M+)

| Model | Window | Recall Performance |
|-------|--------|--------------------|
| **Gemini 3.0 Pro** | 2.5M | 99%+ Needle-in-a-Haystack |
| **Gemini 3 Flash** | 1.0M | Fast, cost-effective long context |
| **Llama 4 70B** | 128K | Reliable mid-length open context |

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

### Semantic Routing

In 2025, static decision trees are being replaced by **Semantic Routers**.
- **How it works**: A small, fast embedding model vectorize the query. If it matches a "known easy" cluster, it's routed to a cheap model (e.g., Gemini 3 Flash). If it hits an "agentic/logic" cluster, it goes to o3 or Claude 4.5.
- **Benefit**: Automates cost-optimization without hardcoded rules.

---

## Sovereign AI and Data Residency

**The 2025 Regulatory Shift:**
Enterprises now prioritize "Sovereign AI" where data and weights must stay within specific borders (GDPR, India, Saudi Arabia, etc.).

| Solution | Provider | Use Case |
|----------|----------|----------|
| **Azure Sovereign Cloud** | Microsoft | Dedicated infra in 40+ regions |
| **AWS Sovereign Cloud** | Amazon | Physically isolated VPCs |
| **OCI Sovereign AI** | Oracle | Dedicated OCI regions for govt/finance |
| **Private Llama 4** | Meta (Self-host) | Maximum data sovereignty |

**Tradeoff**: Sovereign clouds often carry a **20-30% premium** over standard global regions but are mandatory for finance and government sectors.

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

## Capability Comparison

### Benchmark Performance (Approximate, December 2025)

| Model | MMLU | HumanEval | GSM8K | SWE-bench (Ver) |
|-------|------|-----------|-------|-----------------|
| **Claude 4.5 Opus** | 91.2 | 96.0 | 98.2 | 80.9% |
| **GPT-5.2** | 90.5 | 94.2 | 97.8 | 76.5% |
| **Claude Sonnet 4.5** | 89.2 | 93.0 | 97.1 | 72.0% |
| **Gemini 3.0 Pro** | 88.5 | 89.1 | 95.2 | 68.4% |
| **DeepSeek-R1** | 87.9 | 91.0 | 96.5 | 65.2% |
| **Llama 4 405B** | 88.8 | 90.5 | 95.8 | 66.8% |
| **GPT-5.2-mini** | 84.5 | 88.0 | 94.2 | 55.0% |
| **Gemini 3 Flash** | 83.0 | 85.2 | 92.5 | 48.0% |

*Benchmark scores are approximate based on late 2025 releases. Always verify with official technical reports.*

### Task-Specific Recommendations (Dec 2025)

| Task | Recommended Models | Why |
|------|-------------------|-----|
| **Software Engineering** | Claude 4.5 Opus, Claude Sonnet 4.5 | "Claude Code" integration, SoTA SWE-bench |
| **Complex Reasoning** | o3, DeepSeek-R1, Claude Sonnet 4.5 | Native "Thinking" modes for multi-step logic |
| **Autonomous Agents** | GPT-5.2, Claude 4.5 Opus | Best at tool-use reliability and planning |
| **Long Context RAG** | Gemini 3.0 Pro (2.5M) | Unmatched context depth and recall |
| **Real-time Multimodal** | Gemini 3 Flash, GPT-4o | Parallel vision/audio/text streams |
| **High-volume API** | Gemini 3 Flash, o4-mini | Lowest cost per token in class |
| **Private Production** | Llama 4 70B, Nemotron 3 | High capability with local control |
| **Logic/Debug** | o1, o3, DeepSeek-R1 | Spend compute at test-time for accuracy |

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

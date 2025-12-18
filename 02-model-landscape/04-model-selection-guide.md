# Model Selection Guide

A practical framework for choosing the right LLM for your use case, considering capability, cost, latency, and operational factors.

## Table of Contents

- [Selection Framework](#selection-framework)
- [Capability Comparison](#capability-comparison)
- [Use Case Mapping](#use-case-mapping)
- [Cost Analysis](#cost-analysis)
- [Operational Considerations](#operational-considerations)
- [Multi-Model Strategies](#multi-model-strategies)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Selection Framework

### Decision Tree (Dec 2025)

```
Start Here
    │
    ├── Need autonomous agents / long-horizon planning?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ GPT-5.2 / Claude│
    │            │                              │ 4.5 Opus / o3   │
    │            │                              └─────────────────┘
    │            │
    ├── Need best software engineering / coding?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ Claude 4.5 Opus │
    │            │                              │ Claude 3.7 Sonnet│
    │            │                              └─────────────────┘
    │            │
    ├── Need to process massive context (>1M)?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ Gemini 3.0 Pro  │
    │            │                              │ (2.5M context)  │
    │            │                              └─────────────────┘
    │            │
    ├── Cost-sensitive high volume?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ Gemini 3 Flash /│
    │            │                              │ o4-mini         │
    │            │                              └─────────────────┘
    │            │
    └── Default: Production Choice
                 ▼
        ┌─────────────────┐
        │ Claude 3.7 Sonnet│
        │ GPT-5.2-mini    │
        └─────────────────┘
```

### Key Selection Factors

| Factor | Weight | Considerations |
|--------|--------|----------------|
| **Agentic Reliability** | High | Tool-calling accuracy, multi-step planning |
| **Context Recall** | High | Needle-in-a-haystack performance at 1M+ |
| **Rate Limit Ceiling** | High | **(Principal Nuance)**: Can the provider handle your P99 throughput without 429 errors? |
| **Ecosystem Maturity** | High | Production track record, SDK support, and Enterprise SLA |
| **Cost / Output Token** | Medium | Agentic loops consume 5x-10x more tokens |

---

## Capability Comparison

### Frontier Model Comparison (December 2025)

| Model | Strengths | Cons | Context | Best For |
|-------|-----------|------|---------|----------|
| **GPT-5.2** | Agentic planning, native omni | High cost | 512K | Multi-agent systems |
| **Claude 4.5 Opus** | SoTA Software Engineering | Expensive | 400K | Complex codebases |
| **Claude 3.7 Sonnet** | Hybrid Reasoning depth | High peak latency | 200K | General production |
| **Gemini 3.0 Pro** | 2.5M context, multimodal | Latency spikes | 2.5M | Large data ingestion |
| **o3** | Extreme logic/reasoning | High cost/latency | 128K | Math, complex debug |

### Budget Model Comparison

| Model | Cost (per 1M input/output) | Quality | Context | Best For |
|-------|----------------------------|---------|---------|----------|
| **Gemini 3 Flash** | $0.05 / $0.20 | Frontier-tier | 1M | High-volume RAG |
| **o4-mini** | $0.10 / $0.40 | Excellent | 128K | Fast reasoning tasks |
| **Llama 4 8B** | Self-hosted (H100/L40) | Strong | 128K | On-device, private |

### Open Source Models

| Model | Parameters | Quality | Best For |
|-------|------------|---------|----------|
| **Llama 4 70B** | 70B | Frontier-competitive | Universal open choice |
| **Nemotron 3 Ultra** | 500B MoE | Agentic mastery | Scalable open agents |
| **DeepSeek V3.2** | 671B MoE | Ultra performance | Lowest TCO for frontier quality |

---

## Use Case Mapping

### By Application Type (Dec 2025)

| Use Case | Recommended Models | Rationale |
|----------|-------------------|-----------|
| **Autonomous Dev** | Claude 4.5 Opus, Claude 3.7 | "Claude Code" agentic mastery & verified coding |
| **Enterprise RAG** | Gemini 3.0 Pro, Gemini 3 Flash | 2.5M context removes retrieval complexity |
| **customer Support** | Gemini 3 Flash, GPT-5.2-mini | Near-zero latency with strong reasoning |
| **Reasoning / Debug** | o3, DeepSeek-R1 | Best at "Thinking" mode for code/logic |
| **Video/Multimodal** | Gemini 3.0 Pro, GPT-5.2 | Native interleaved multimodal processing |
| **Private Agent** | Llama 4 70B, Nemotron 3 | Strongest open-weight agentic planning |

### By Constraint

| Constraint | Approach |
|------------|----------|
| **Max latency < 100ms** | Gemini 3 Flash, o4-mini, or self-hosted Nano models |
| **Context > 1M tokens** | Gemini 3.0 Pro (native 2.5M) |
| **Zero-data Leakage** | Llama 4 70B on internal VPC |
| **Complex Tool Use** | Claude 4.5 Opus or GPT-5.2 (best planning accuracy) |

---

## Cost Analysis

### Cost Modeling (Dec 2025)

| Model | Input / 1M | Output / 1M | Notes |
|-------|------------|-------------|-------|
| **GPT-5.2** | $5.00 | $20.00 | Agentic premium |
| **Claude 4.5 Opus** | $15.00 | $75.00 | Specialized engineering |
| **Claude 3.7 Sonnet** | $3.00 | $15.00 | Balanced choice |
| **Gemini 3.0 Pro** | $1.25 | $5.00 | Best value frontier |
| **Gemini 3 Flash** | $0.05 | $0.20 | RAG-at-scale winner |
| **o4-mini** | $0.10 | $0.40 | Logic-on-a-budget |

### Cost Comparison Example

Assume 1M queries/month, 1K input tokens + 500 output tokens per query:

| Volume | GPT-5.2 | Claude Sonnet | Gemini 3 Pro | Gemini 3 Flash |
|--------|---------|---------------|--------------|----------------|
| 10K queries/mo | $150 | $105 | $37.50 | $1.50 |
| 1M queries/mo | $15,000 | $10,500 | $3,750 | $150 |

*2025 Insight: Gemini 3 Flash has effectively commoditized RAG, making long-context processing cheaper than traditional vector search infra at scale.*

---

## Operational Considerations

### Rate Limits and Quotas

| Provider | Tier | RPM | TPM |
|----------|------|-----|-----|
| OpenAI (Tier 1) | Basic | 500 | 30K |
| OpenAI (Tier 5) | Enterprise | 10K | 10M |
| Anthropic (Tier 1) | Basic | 50 | 40K |
| Anthropic (Tier 4) | Enterprise | 4K | 400K |

### Reliability Patterns

```python
class ReliableModelClient:
    def __init__(self):
        self.providers = {
            "primary": OpenAIClient(),
            "fallback1": AnthropicClient(),
            "fallback2": GoogleClient()
        }
    
    async def generate(self, prompt: str) -> str:
        for name, client in self.providers.items():
            try:
                return await client.generate(prompt)
            except RateLimitError:
                continue
            except ServiceError:
                continue
        
        raise AllProvidersUnavailable()
```

### Abstraction Layer

```python
class LLMClient:
    """Unified interface for multiple providers."""
    
    def __init__(self, config: dict):
        self.default_model = config["default_model"]
        self.clients = self._init_clients(config)
    
    async def generate(
        self,
        messages: list[dict],
        model: str = None,
        **kwargs
    ) -> str:
        model = model or self.default_model
        client = self._get_client(model)
        
        # Normalize request format
        normalized = self._normalize_request(messages, kwargs)
        
        # Call provider
        response = await client.generate(**normalized)
        
        # Normalize response
        return self._normalize_response(response)
    
    def _normalize_request(self, messages: list[dict], kwargs: dict) -> dict:
        # Handle differences between providers
        # OpenAI uses 'messages', Anthropic uses 'messages' with different format
        pass
```

---

## Multi-Model Strategies

### Model Routing

```python
class ModelRouter:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.models = {
            "simple": "gpt-4o-mini",
            "complex": "claude-3.5-sonnet",
            "code": "claude-3.5-sonnet",
            "long_context": "gemini-1.5-pro",
            "reasoning": "o1-mini"
        }
    
    async def route(self, query: str, context_length: int) -> str:
        # Classify query complexity
        query_type = await self.classifier.classify(query)
        
        # Override for long context
        if context_length > 100_000:
            return self.models["long_context"]
        
        return self.models[query_type]
```

### Cascade Pattern (2025 Refinement)

**The Logic**: Never use a 70B model for a task a 1B model can do. Use a "Router" to score confidence.

```python
class ModelCascade:
    """The 'Efficiency First' Pattern."""
    
    async def generate_optimized(self, query: str):
        # 1. Draft check (SLM / Classifier)
        if is_simple_intent(query):
            return await gpt4o_mini.generate(query)
            
        # 2. Main Generation (Efficient model)
        response = await claude_sonnet.generate(query)
        
        # 3. Validation / Escalate
        if needs_verification(response):
            return await o3.generate(f"Verify this: {response}")
            
        return response
```

**Principal-level Tip:** Implement "Semantic Fallback" where you don't just retry the same model on error, but immediately jump to a larger model or a different provider (OpenAI -> Anthropic) to avoid correlated failures.

---

## Interview Questions

### Q: How do you choose between GPT-4o, Claude, and Gemini for a production application?

**Strong answer:**

"My selection depends on specific requirements:

**For most production workloads**, I default to Claude 3.5 Sonnet or GPT-4o. Both are excellent general-purpose models. Sonnet has a slight edge on coding, GPT-4o has better ecosystem integration.

**For long-context applications**, Gemini 1.5 Pro is the clear winner with 1-2 million token context. If I need to process entire codebases or very long documents, Gemini is my choice.

**For cost-sensitive high-volume**, GPT-4o-mini or Claude Haiku. These are 10-20x cheaper and handle straightforward tasks well.

**My practical approach:**
1. Prototype with Sonnet or GPT-4o to validate the use case
2. Evaluate on MY specific task, not just benchmarks
3. Build abstraction layer so I can switch easily
4. Optimize costs by routing simpler requests to cheaper models

I never rely solely on benchmark scores. A model that ranks lower on MMLU might excel on my domain."

### Q: When would you self-host vs use API providers?

**Strong answer:**

"It is a tradeoff of control vs operational burden.

**Use APIs when:**
- Volume under 1M queries/month (cost crossover)
- Need latest models immediately
- Team lacks GPU infrastructure expertise
- Variable workload hard to capacity plan
- Time-to-market is critical

**Self-host when:**
- Data cannot leave infrastructure (compliance)
- Volume exceeds 10M queries/month (cost savings)
- Need latency under 100ms P99
- Need custom model weights or fine-tuning
- Full control over model behavior

**Hybrid often works best:**
- Self-host for high-volume predictable workloads
- API for spikes and specialized models
- API as fallback when self-hosted fails

Hidden costs of self-hosting: GPU procurement, engineering time, model updates, monitoring. Factor in 1-2 dedicated engineers for infrastructure."

---

## References

- OpenAI API: https://platform.openai.com/
- Anthropic API: https://docs.anthropic.com/
- Google AI: https://ai.google.dev/
- LMSys Leaderboard: https://chat.lmsys.org/

---

*Next: [Fine-Tuning Guide](../03-fine-tuning/01-when-to-fine-tune.md)*

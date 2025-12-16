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

### Decision Tree

```
Start Here
    │
    ├── Need frontier capability (best quality)?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ GPT-4o / Claude │
    │            │                              │ 3.5 Sonnet / o1 │
    │            │                              └─────────────────┘
    │            │
    ├── Cost is primary constraint?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ GPT-4o-mini /   │
    │            │                              │ Claude Haiku    │
    │            │                              └─────────────────┘
    │            │
    ├── Need very long context (>200K)?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ Gemini 1.5 Pro  │
    │            │                              │ (1-2M context)  │
    │            │                              └─────────────────┘
    │            │
    ├── Need to self-host / data stays local?
    │   └── Yes ─────────────────────────────────────────┐
    │   └── No ──┐                                       │
    │            │                                       ▼
    │            │                              ┌─────────────────┐
    │            │                              │ Llama 3 / Qwen  │
    │            │                              │ / Mistral       │
    │            │                              └─────────────────┘
    │            │
    └── Default: General purpose
                 ▼
        ┌─────────────────┐
        │ GPT-4o / Claude │
        │ 3.5 Sonnet      │
        └─────────────────┘
```

### Key Selection Factors

| Factor | Weight | Considerations |
|--------|--------|----------------|
| Task performance | High | Benchmark on YOUR specific task |
| Cost per token | Medium-High | Input + output, volume-based |
| Latency | Variable | TTFT, throughput, streaming |
| Context window | Variable | Required context size |
| Data residency | High (regulated) | Where data is processed |
| Ecosystem | Medium | Tools, integrations, support |
| Reliability | High | Uptime, rate limits |

---

## Capability Comparison

### Frontier Model Comparison (December 2025)

| Model | Strengths | Weaknesses | Context | Best For |
|-------|-----------|------------|---------|----------|
| **GPT-4o** | Balanced, good reasoning, multimodal | Higher cost than mini | 128K | General purpose, production |
| **Claude 3.5 Sonnet** | Excellent coding, instruction following | Slightly slower | 200K | Code, complex instructions |
| **Claude 3.5 Opus** | Best reasoning, nuance | Expensive, slower | 200K | High-stakes decisions |
| **Gemini 1.5 Pro** | Massive context, cost-effective | Slightly less reliable | 2M | Long-context applications |
| **o1** | Strong reasoning, math | Very expensive, slow | 128K | Complex reasoning, research |

⚠️ Verify current pricing and capabilities - model landscape changes rapidly.

### Budget Model Comparison

| Model | Cost (per 1M tokens) | Quality | Context | Best For |
|-------|---------------------|---------|---------|----------|
| **GPT-4o-mini** | $0.15 / $0.60 | Very good | 128K | High-volume, cost-sensitive |
| **Claude 3.5 Haiku** | $0.25 / $1.25 | Very good | 200K | Fast responses, cost-sensitive |
| **Gemini 1.5 Flash** | $0.075 / $0.30 | Good | 1M | Very high volume, long context |

### Open Source Models

| Model | Parameters | Quality | Best For |
|-------|------------|---------|----------|
| **Llama 3.1 70B** | 70B | Near-frontier | General purpose self-hosted |
| **Llama 3.1 405B** | 405B | Frontier-level | Maximum open source quality |
| **Qwen 2.5 72B** | 72B | Excellent | Multilingual, math |
| **Mistral Large** | 123B | Very good | European data residency |
| **DeepSeek-V3** | 671B MoE | Frontier-competitive | Cost-effective inference |

---

## Use Case Mapping

### By Application Type

| Use Case | Recommended Models | Rationale |
|----------|-------------------|-----------|
| **Customer support** | Claude Haiku, GPT-4o-mini | High volume, cost-sensitive |
| **Code generation** | Claude Sonnet, GPT-4o | Best coding capabilities |
| **Document analysis** | Gemini 1.5 Pro | Long context for full documents |
| **Creative writing** | Claude Sonnet, GPT-4o | Quality and style |
| **Data extraction** | GPT-4o-mini, Claude Haiku | Structured output, high volume |
| **Research/reasoning** | o1, Claude Opus | Complex multi-step reasoning |
| **Embeddings** | text-embedding-3-large, Voyage | Specialized for retrieval |

### By Constraint

| Constraint | Approach |
|------------|----------|
| **Max latency < 500ms** | Use smaller models, optimize prompts |
| **Cost < $0.001/query** | GPT-4o-mini, aggressive caching |
| **Data cannot leave infra** | Self-hosted Llama 3.1, Qwen |
| **100% uptime required** | Multi-provider with failover |
| **Regulatory compliance** | Audit trail, approved providers |

---

## Cost Analysis

### Cost Modeling

```python
class CostModeler:
    # Prices per 1M tokens (December 2025, verify current)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 0.25, "output": 1.25},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }
    
    def estimate_monthly_cost(
        self,
        model: str,
        queries_per_month: int,
        avg_input_tokens: int,
        avg_output_tokens: int
    ) -> float:
        pricing = self.PRICING[model]
        
        total_input = queries_per_month * avg_input_tokens
        total_output = queries_per_month * avg_output_tokens
        
        input_cost = (total_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def compare_models(
        self,
        queries_per_month: int,
        avg_input_tokens: int,
        avg_output_tokens: int
    ) -> dict:
        comparisons = {}
        for model in self.PRICING:
            comparisons[model] = self.estimate_monthly_cost(
                model, queries_per_month, avg_input_tokens, avg_output_tokens
            )
        return dict(sorted(comparisons.items(), key=lambda x: x[1]))
```

### Cost Comparison Example

| Volume | GPT-4o | GPT-4o-mini | Claude Sonnet | Gemini Pro |
|--------|--------|-------------|---------------|------------|
| 10K queries/mo | $125 | $7.50 | $180 | $62.50 |
| 100K queries/mo | $1,250 | $75 | $1,800 | $625 |
| 1M queries/mo | $12,500 | $750 | $18,000 | $6,250 |

*Assumes 1K input tokens + 500 output tokens per query*

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

### Cascade Pattern

```python
class ModelCascade:
    """Try cheap model first, escalate if needed."""
    
    async def generate(self, query: str) -> str:
        # Try cheap model first
        cheap_response = await self.cheap_model.generate(query)
        confidence = await self.score_confidence(cheap_response)
        
        if confidence > 0.8:
            return cheap_response
        
        # Escalate to better model
        return await self.expensive_model.generate(query)
```

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

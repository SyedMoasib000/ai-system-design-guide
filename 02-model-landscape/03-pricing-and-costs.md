# Pricing and Costs

Understanding the cost structure of LLM systems is essential for production planning. This chapter covers pricing models, cost optimization strategies, and total cost of ownership analysis.

## Table of Contents

- [Pricing Models](#pricing-models)
- [Current API Pricing](#current-api-pricing)
- [Cost Calculation](#cost-calculation)
- [Cost Optimization Strategies](#cost-optimization-strategies)
- [Context Caching Economics](#context-caching-economics)
- [Self-Hosting & GPU Cloud Arbitrage](#self-hosting-economics)
- [Total Cost of Ownership](#total-cost-of-ownership)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Pricing Models

### Token-Based Pricing

Most LLM APIs charge per token:

```
Cost = (input_tokens × input_rate) + (output_tokens × output_rate)
```

**Key observations:**
- Output tokens cost 2-5x more than input tokens
- Pricing varies significantly by model tier
- Some providers offer batch discounts

### Tiered Pricing

Some providers offer volume discounts:

| Tier | Monthly Spend | Discount |
|------|---------------|----------|
| Standard | $0 - $5K | 0% |
| Growth | $5K - $50K | 10-20% |
| Enterprise | $50K+ | Custom negotiation |

### Commitment-Based Pricing

Pre-purchase tokens at discounted rates:

```
Standard: $2.50 / 1M input tokens
Committed (1-year): $2.00 / 1M input tokens (20% savings)
```

---

## Current API Pricing

### December 2025 Pricing ⚠️

*Verify current values: Pricing in late 2025 has shifted toward 'Inference-time Scaling' (Thinking tokens).*

#### OpenAI
| Model | Input / 1M | Output / 1M | Thinking Tier |
|-------|------------|-------------|---------------|
| **GPT-5.2** | $5.00 | $20.00 | Included |
| **o3** | $15.00 | $60.00 | High-compute |
| **GPT-5.2-mini**| $0.10 | $0.40 | None |
| **o4-mini** | $0.15 | $0.60 | Low-compute |

#### Anthropic (The "Claude 4" Generation)
| Model | Input / 1M | Output / 1M | Reasoning |
|-------|------------|-------------|-----------|
| **Claude 4.5 Opus** | $15.00 | $75.00 | Native SoTA |
| **Claude Sonnet 4.5**| $3.00 | $15.00 | Hybrid (CoT) |
| **Claude 4.5 Haiku** | $0.20 | $1.00 | None |

#### Google (The "Gemini 3" Generation)
| Model | Input / 1M | Output / 1M | Context |
|-------|------------|-------------|---------|
| **Gemini 3.0 Pro** | $1.25 | $5.00 | 2.5M tokens |
| **Gemini 3 Flash** | $0.05 | $0.20 | 1M tokens |

#### Embedding Models (Dec 2025)
| Model | Cost / 1M tokens | Dimension |
|-------|------------------|-----------|
| **text-embedding-4** | $0.10 | 4096 |
| **Voyage-3** | $0.06 | 1024 |
| **Cohere embed-v4** | $0.10 | 1024 |

> [!IMPORTANT]
> **Inference-time Compute Costs:** For models with "Thinking" or "Reasoning" modes (o1, o3, Claude Sonnet 4.5), you are often charged for **internal thinking tokens** even if they are not shown to the user. This can increase total request cost by 3x-10x for logic-heavy tasks.

---

## Cost Calculation

### Basic Cost Formula

```python
def calculate_request_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 0.25, "output": 1.25},
    }
    
    rates = pricing[model]
    cost = (
        (input_tokens / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"]
    )
    return cost
```

### Example Cost Calculations

**Scenario 1: RAG Chatbot**
```
Per request:
- System prompt: 500 tokens
- Retrieved context: 2,000 tokens
- User message: 100 tokens
- Response: 300 tokens

Input: 2,600 tokens, Output: 300 tokens

GPT-4o cost: (2600 × $2.50 + 300 × $10) / 1M = $0.0095 per request

At 10,000 requests/day:
Daily: $95
Monthly: $2,850
```

**Scenario 2: Document Summarization**
```
Per document:
- Document: 8,000 tokens
- Summary: 500 tokens

GPT-4o cost: (8000 × $2.50 + 500 × $10) / 1M = $0.025

1,000 documents: $25
10,000 documents: $250
```

### Monthly Cost Projection

```python
def project_monthly_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str
) -> dict:
    per_request = calculate_request_cost(
        avg_input_tokens, avg_output_tokens, model
    )
    
    daily = per_request * requests_per_day
    monthly = daily * 30
    yearly = monthly * 12
    
    return {
        "per_request": per_request,
        "daily": daily,
        "monthly": monthly,
        "yearly": yearly
    }

# Example
costs = project_monthly_cost(
    requests_per_day=50000,
    avg_input_tokens=2000,
    avg_output_tokens=400,
    model="gpt-4o"
)
# Output: ~$12,500/month
```

---

## Cost Optimization Strategies

### Strategy 1: Model Routing

Route requests to appropriate model tiers:

```python
class ModelRouter:
    def __init__(self):
        self.classifier = load_complexity_classifier()
    
    def route(self, query: str, context: str) -> str:
        complexity = self.classifier.predict(query)
        
        if complexity < 0.3:
            return "gpt-4o-mini"  # Simple queries
        elif complexity < 0.7:
            return "gpt-4o-mini"  # Medium, try cheap first
        else:
            return "gpt-4o"  # Complex queries
    
    def route_with_fallback(self, query: str, context: str) -> str:
        # Try cheap model first
        response = self.try_model("gpt-4o-mini", query, context)
        
        if self.is_quality_sufficient(response):
            return response
        
        # Fallback to expensive model
        return self.try_model("gpt-4o", query, context)
```

**Potential savings:** 50-70% with minimal quality impact

### Strategy 2: Prompt Optimization

Reduce token count without losing quality:

```python
# Before: 2,500 tokens
system_prompt = """
You are a helpful customer support assistant for Acme Corp. 
You have access to our product documentation and should answer 
questions accurately and helpfully. Always be polite and professional.
If you don't know something, say so rather than making things up.
Format your responses clearly with bullet points when listing items.
[... more verbose instructions ...]
"""

# After: 800 tokens
system_prompt = """
You are Acme Corp's support assistant.
Rules:
- Answer from provided context only
- Admit uncertainty
- Use bullet points for lists
- Be concise
"""

# Savings: 1,700 tokens × $2.50/1M = $0.00425 per request
# At 10K requests/day: $42.50/day = $1,275/month
```

### Strategy 3: Caching

Cache responses for repeated or similar queries:

```python
class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.exact_cache = TTLCache(maxsize=10000, ttl=ttl_seconds)
        self.semantic_cache = SemanticCache(threshold=0.95)
    
    def get_or_generate(self, query: str, context: str) -> tuple[str, bool]:
        # Check exact cache
        cache_key = self.make_key(query, context)
        if cache_key in self.exact_cache:
            return self.exact_cache[cache_key], True  # Cache hit
        
        # Check semantic cache
        similar = self.semantic_cache.find_similar(query)
        if similar:
            return similar.response, True  # Semantic hit
        
        # Generate new response
        response = self.generate(query, context)
        self.exact_cache[cache_key] = response
        self.semantic_cache.add(query, response)
        
        return response, False  # Cache miss

# With 30% cache hit rate:
# Baseline: $3,000/month
# With caching: $2,100/month
# Savings: $900/month
```

### Strategy 4: Batch Processing

Process multiple requests together for efficiency:

```python
# Real-time: pay full price
for query in queries:
    response = model.generate(query)

# Batch API (OpenAI offers 50% discount):
batch_responses = model.batch_generate(queries)
# Cost: 50% of real-time pricing
```

### Strategy 5: Output Length Control

Limit response length appropriately:

```python
# Reduce unnecessary output
response = model.generate(
    prompt=prompt,
    max_tokens=300,  # Limit output
    stop=["\n\n"]    # Stop at natural break
)

# Cost impact:
# Before: avg 500 output tokens = $0.005 per request (GPT-4o)
# After: avg 250 output tokens = $0.0025 per request
# Savings: 50% on output costs
```

### Cost Optimization Summary

| Strategy | Effort | Potential Savings |
|----------|--------|-------------------|
| Model routing | Medium | 50-70% |
| **Context Caching** | Low | **60-90% (Input)** |
| Prompt optimization | Low | 20-40% |
| Response caching | Medium | 20-40% |
| Batch processing | Low | 50% (OpenAI/Anthropic) |

---

## Context Caching Economics

**The 2025 "Golden Rule" for RAG.**
If you have a fixed system prompt or a shared knowledge base (prefix) larger than 10,000 tokens, **Context Caching** is mandatory.

**Break-even Analysis:**
- **Standard Input**: $3.00 / 1M tokens (Claude Sonnet 4.5)
- **Cached Input**: $0.30 / 1M tokens (90% discount)
- **Cache Write Fee**: $3.75 / 1M tokens (one-time)

`Break-even = (Write Fee) / (Standard Rate - Cached Rate) ≈ 1.4 requests`

If your long prefix is used by **more than 2 users**, caching it is strictly cheaper than sending it raw every time.

---

## Self-Hosting & GPU Cloud Arbitrage

**The Reserved vs. Serverless Tradeoff:**

| Model Size | Serverless (RunPod/Together) | Reserved (Lambda/AWS) |
|------------|-----------------------------|-----------------------|
| **Burst Capacity** | Infinite (cold starts) | Fixed |
| **Utilization** | Pay only for compute time | 24/7 fixed cost |
| **TCO Break-even**| **Cost-effective < 40% util** | **Cost-effective > 40% util** |

**Principal-level Nuance:**
"GPU Cloud Arbitrage" involves moving production workloads between providers based on **Spot Instance availability**. In 2025, tools like **Skypilot** automate this, saving up to 60% on self-hosting costs by following "low-demand" regions globally.

### When Self-Hosting Makes Sense

```
Break-even analysis:

API cost at scale:
- 1M requests/month
- 2,500 tokens average
- GPT-4o: ~$25,000/month

Self-hosted equivalent (Llama 70B):
- 4x H100 80GB: ~$12/hour × 730 = $8,760/month
- Engineering time: $5,000/month (0.5 FTE)
- Ops overhead: $2,000/month
- Total: ~$15,760/month

Savings: $9,240/month = 37%
```

### Self-Hosting Cost Components

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| GPU compute | $5K-20K | Depends on model size |
| Storage | $200-500 | Model weights, logs |
| Networking | $100-500 | Egress, load balancing |
| Engineering | $5K-15K | Partial FTE for ops |
| Monitoring | $100-500 | Observability tools |

### GPU Requirements by Model Size

| Model Size | GPU Config | Estimated Cost/Month |
|------------|------------|---------------------|
| 7B (INT4) | 1x A10G | $500-800 |
| 7B (FP16) | 1x A100 40GB | $1,500-2,500 |
| 70B (INT4) | 2x A100 80GB | $5,000-8,000 |
| 70B (FP16) | 4x A100 80GB | $10,000-15,000 |
| 405B (INT4) | 8x H100 | $20,000-30,000 |

### Decision Framework

```
Choose API when:
- Volume < 100K requests/month
- No ML ops expertise
- Need highest quality (frontier models)
- Fast iteration needed

Choose self-hosting when:
- Volume > 500K requests/month
- Have ML infrastructure team
- Data privacy requirements
- Predictable, stable workload
- Custom fine-tuning needed
```

---

## Total Cost of Ownership

### TCO Components

```python
def calculate_tco(scenario: dict) -> dict:
    # Direct costs
    api_or_compute = scenario["monthly_api_cost"]
    
    # Engineering costs
    development = scenario["dev_hours"] * scenario["engineer_rate"]
    maintenance = scenario["maintenance_hours"] * scenario["engineer_rate"]
    
    # Infrastructure
    vector_db = scenario["vector_db_cost"]
    monitoring = scenario["monitoring_cost"]
    
    # Indirect costs
    downtime_risk = scenario["expected_downtime_hours"] * scenario["revenue_per_hour"]
    
    monthly_tco = (
        api_or_compute +
        development / 12 +  # Amortized over year
        maintenance +
        vector_db +
        monitoring +
        downtime_risk
    )
    
    return {
        "monthly_tco": monthly_tco,
        "yearly_tco": monthly_tco * 12,
        "breakdown": {
            "llm": api_or_compute,
            "engineering": development / 12 + maintenance,
            "infrastructure": vector_db + monitoring,
            "risk": downtime_risk
        }
    }
```

### Example TCO Comparison

**Scenario: Customer Support Bot (50K requests/month)**

| Cost Component | API-Based | Self-Hosted |
|----------------|-----------|-------------|
| LLM costs | $5,000 | $3,000 |
| Vector DB | $70 | $200 |
| Engineering (monthly) | $500 | $3,000 |
| Monitoring | $100 | $200 |
| **Monthly Total** | **$5,670** | **$6,400** |

*At this scale, API is cheaper due to engineering overhead.*

**Scenario: Large-Scale RAG (2M requests/month)**

| Cost Component | API-Based | Self-Hosted |
|----------------|-----------|-------------|
| LLM costs | $50,000 | $15,000 |
| Vector DB | $500 | $1,000 |
| Engineering (monthly) | $1,000 | $8,000 |
| Monitoring | $200 | $500 |
| **Monthly Total** | **$51,700** | **$24,500** |

*At this scale, self-hosting is significantly cheaper.*

---

## Interview Questions

### Q: How would you optimize costs for a high-volume RAG application?

**Strong answer:**
I would approach cost optimization in layers:

**1. Architecture optimization:**
- Model routing: Use cheap model for simple queries
- Caching: 30-40% of queries may be cacheable
- Prompt compression: Minimize system prompt tokens

**2. Model selection:**
```
Simple queries (60%): GPT-4o-mini at $0.001/request
Complex queries (40%): GPT-4o at $0.01/request
Weighted avg: $0.0046/request (vs $0.01 all GPT-4o)
Savings: 54%
```

**3. Infrastructure:**
- Batch embedding updates (50% cheaper)
- Right-size vector DB
- Use spot instances where possible

**4. Monitoring:**
- Track cost per query type
- Alert on anomalies
- Regular cost reviews

### Q: When would you recommend self-hosting vs using APIs?

**Strong answer:**
Decision depends on multiple factors:

**Volume threshold:**
- Below 100K/month: Almost always API
- 100K-500K: Evaluate case by case
- Above 500K: Often self-hosting wins

**Team capabilities:**
- No ML ops: API regardless of scale
- Strong infra team: Consider self-hosting earlier

**Quality requirements:**
- Need absolute best: APIs (frontier models)
- Good enough works: Self-hosted open models

**Other factors:**
- Data privacy: May force self-hosting
- Latency control: Self-hosting gives more control
- Fine-tuning needs: Self-hosting enables more customization

**My recommendation process:**
1. Start with APIs for fastest iteration
2. Build abstraction layer for model switching
3. Evaluate self-hosting when spend exceeds $10K/month
4. Pilot with shadow deployment before committing

---

## References

- OpenAI Pricing: https://openai.com/pricing
- Anthropic Pricing: https://www.anthropic.com/pricing
- Google AI Pricing: https://ai.google.dev/pricing
- Lambda Labs GPU Pricing: https://lambdalabs.com/service/gpu-cloud
- RunPod Pricing: https://www.runpod.io/pricing

---

*Previous: [Capability Assessment](02-capability-assessment.md) | Next: [Model Selection Guide](04-model-selection-guide.md)*

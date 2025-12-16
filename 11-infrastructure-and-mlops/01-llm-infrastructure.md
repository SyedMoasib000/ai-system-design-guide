# LLM Infrastructure

Building production LLM systems requires understanding deployment options, scaling patterns, and operational concerns. This chapter covers the infrastructure layer.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Serving Architecture](#serving-architecture)
- [Scaling Patterns](#scaling-patterns)
- [Cost Management](#cost-management)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Disaster Recovery](#disaster-recovery)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Deployment Options

### API vs Self-Hosted

| Factor | API Providers | Self-Hosted |
|--------|---------------|-------------|
| Setup time | Minutes | Days to weeks |
| Operational burden | None | Significant |
| Cost at low volume | Lower | Higher (fixed costs) |
| Cost at high volume | Higher | Lower (scale economics) |
| Latency control | Limited | Full control |
| Data privacy | Data leaves your infra | Data stays local |
| Model selection | Provider's models | Any open model |
| Customization | Fine-tuning via API | Full control |

### When to Use API Providers

```python
# Decision framework
def should_use_api(requirements: dict) -> bool:
    # Strong signals for API
    if requirements["time_to_market"] == "urgent":
        return True
    if requirements["query_volume"] < 100_000_per_month:
        return True
    if requirements["team_ml_expertise"] == "low":
        return True
    
    # Strong signals for self-hosted
    if requirements["data_residency"] == "strict":
        return False
    if requirements["latency_p99_ms"] < 100:
        return False
    if requirements["query_volume"] > 10_000_000_per_month:
        return False
    
    # Default to API for simplicity
    return True
```

### Self-Hosting Options

| Option | Complexity | Performance | Use Case |
|--------|------------|-------------|----------|
| vLLM | Medium | Excellent | Production serving |
| TGI (HuggingFace) | Medium | Very good | HuggingFace ecosystem |
| TensorRT-LLM | High | Best (NVIDIA) | Maximum performance |
| Ollama | Low | Good | Development, small scale |
| llama.cpp | Low | Good | CPU inference, edge |

---

## Serving Architecture

### Single Model Serving

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Gateway   │────▶│  LLM Server │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Cache    │
                    └─────────────┘
```

### Multi-Model Serving

```
                    ┌─────────────────────────────── │
                    │         Load Balancer          │
                    └───────────────┬────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  GPT-4 Pool   │       │  Claude Pool  │       │ Llama 70B Pool│
    │  (API calls)  │       │  (API calls)  │       │ (self-hosted) │
    └───────────────┘       └───────────────┘       └───────────────┘
```

### Model Router Pattern

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": GPT4oMini(),
            "complex": Claude35Sonnet(),
            "code": Claude35Sonnet(),
            "long_context": Gemini15Pro(),
            "vision": GPT4o()
        }
        self.classifier = QueryClassifier()
    
    async def route(self, request: Request) -> Response:
        # Classify request type
        request_type = self.classifier.classify(request)
        
        # Route to appropriate model
        model = self.models[request_type]
        
        # Execute with fallback
        try:
            return await model.generate(request)
        except RateLimitError:
            return await self.fallback(request, request_type)
    
    async def fallback(self, request: Request, original_type: str) -> Response:
        # Define fallback order
        fallbacks = {
            "simple": ["complex", "long_context"],
            "complex": ["simple"],
            "code": ["complex"]
        }
        
        for fallback_type in fallbacks.get(original_type, []):
            try:
                return await self.models[fallback_type].generate(request)
            except Exception:
                continue
        
        raise ServiceUnavailableError("All models unavailable")
```

---

## Scaling Patterns

### Horizontal Scaling

```python
# Kubernetes HPA config for LLM service
hpa_config = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
"""
```

### GPU Scaling for Self-Hosted

| Scale | GPUs | Suggested Setup |
|-------|------|-----------------|
| Dev/Test | 1 | Single A10G or L4 |
| Small prod | 2-4 | 2x A100 with tensor parallel |
| Medium prod | 4-8 | 4x H100 with tensor parallel |
| Large prod | 8+ | Multi-node with pipeline parallel |

### Queue-Based Architecture

For high-throughput async workloads:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Producers  │────▶│    Queue    │────▶│  Consumers  │
└─────────────┘     │  (Redis/    │     │  (LLM       │
                    │   SQS)      │     │   Workers)  │
                    └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Results    │
                                        │  Store      │
                                        └─────────────┘
```

```python
class AsyncLLMProcessor:
    def __init__(self):
        self.queue = RedisQueue("llm_requests")
        self.results = RedisResults("llm_results")
    
    async def submit(self, request: Request) -> str:
        request_id = generate_id()
        await self.queue.enqueue({
            "id": request_id,
            "request": request.to_dict()
        })
        return request_id
    
    async def get_result(self, request_id: str, timeout: int = 300) -> Response:
        return await self.results.wait_for(request_id, timeout)
    
    # Worker process
    async def worker_loop(self):
        while True:
            job = await self.queue.dequeue()
            try:
                result = await self.llm.generate(job["request"])
                await self.results.store(job["id"], result)
            except Exception as e:
                await self.results.store_error(job["id"], str(e))
```

---

## Cost Management

### Cost Tracking

```python
class CostTracker:
    # Pricing as of December 2025 (verify current rates)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 0.25, "output": 1.25},
    }
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def track(self, request_id: str, model: str, tokens: dict):
        cost = self.calculate_cost(
            model,
            tokens["input"],
            tokens["output"]
        )
        
        self.metrics.record(
            "llm_cost",
            cost,
            tags={"model": model, "request_id": request_id}
        )
        
        return cost
```

### Cost Optimization Strategies

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| Model routing | 50-80% | Route simple queries to cheap models |
| Caching | 30-70% | Cache frequent queries |
| Prompt optimization | 10-30% | Shorter prompts, structured output |
| Batch API | 50% | Use batch endpoints for async work |
| Self-hosting | Variable | At scale, can be cheaper |

### Budget Alerts

```python
class BudgetManager:
    def __init__(self, daily_budget: float, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
    
    async def check_and_alert(self):
        today_cost = await self.get_today_cost()
        utilization = today_cost / self.daily_budget
        
        if utilization >= 1.0:
            await self.alert("CRITICAL: Daily budget exceeded", today_cost)
            # Consider enabling cost controls
            await self.enable_rate_limiting()
        elif utilization >= self.alert_threshold:
            await self.alert("WARNING: Approaching daily budget", today_cost)
    
    async def enable_rate_limiting(self):
        # Reduce throughput to stay within budget
        self.rate_limiter.set_rate(
            requests_per_minute=self.calculate_safe_rate()
        )
```

---

## Monitoring and Alerting

### Key Metrics

```python
LLM_METRICS = {
    # Latency
    "ttft_seconds": "Time to first token",
    "total_latency_seconds": "Total request time",
    
    # Throughput
    "requests_per_second": "Request rate",
    "tokens_per_second": "Token generation rate",
    
    # Resources
    "gpu_utilization": "GPU compute usage",
    "gpu_memory_utilization": "GPU memory usage",
    "kv_cache_utilization": "KV cache usage",
    
    # Quality (sampled)
    "quality_score": "LLM-as-judge score",
    "faithfulness_score": "RAG faithfulness",
    
    # Errors
    "error_rate": "Failed requests percentage",
    "rate_limit_hits": "Rate limit rejections",
    
    # Cost
    "cost_per_request": "Average cost per request",
    "daily_cost": "Total daily spend"
}
```

### Alert Configuration

```yaml
alerts:
  - name: high_error_rate
    condition: error_rate > 0.05
    for: 5m
    severity: critical
    
  - name: high_latency
    condition: p99_latency > 10s
    for: 5m
    severity: warning
    
  - name: cost_spike
    condition: hourly_cost > 2 * avg_hourly_cost
    for: 1h
    severity: warning
    
  - name: quality_degradation
    condition: avg_quality_score < 3.5
    for: 30m
    severity: warning
    
  - name: gpu_memory_pressure
    condition: gpu_memory_utilization > 0.95
    for: 5m
    severity: warning
```

---

## Disaster Recovery

### Multi-Provider Failover

```python
class MultiProviderClient:
    def __init__(self):
        self.providers = [
            OpenAIClient(),
            AnthropicClient(),
            GoogleClient()
        ]
        self.primary = 0
    
    async def generate(self, request: Request) -> Response:
        # Try primary provider first
        try:
            return await self.providers[self.primary].generate(request)
        except (RateLimitError, ServiceError) as e:
            return await self.failover(request, e)
    
    async def failover(self, request: Request, original_error: Exception) -> Response:
        for i, provider in enumerate(self.providers):
            if i == self.primary:
                continue
            try:
                response = await provider.generate(request)
                # Log failover for monitoring
                self.log_failover(self.primary, i, original_error)
                return response
            except Exception:
                continue
        
        raise AllProvidersUnavailable("All LLM providers failed")
```

### Graceful Degradation

```python
class GracefulDegradation:
    def __init__(self):
        self.cache = ResponseCache()
        self.fallback_responses = FallbackResponses()
    
    async def handle_outage(self, request: Request) -> Response:
        # Level 1: Try cache
        cached = await self.cache.get_similar(request.query)
        if cached and cached.similarity > 0.9:
            return Response(
                content=cached.response,
                metadata={"source": "cache", "degraded": True}
            )
        
        # Level 2: Try fallback responses
        fallback = self.fallback_responses.get(request.intent)
        if fallback:
            return Response(
                content=fallback,
                metadata={"source": "fallback", "degraded": True}
            )
        
        # Level 3: Graceful error
        return Response(
            content="I am currently experiencing issues. Please try again later or contact support.",
            metadata={"source": "error", "degraded": True}
        )
```

---

## Interview Questions

### Q: How would you design infrastructure for 1M LLM queries per day?

**Strong answer:**

"At 1M queries per day, that is about 12 queries per second on average, with peaks potentially 3-5x higher. Here is my approach:

**Architecture:**
- Load balancer distributing across multiple API endpoints
- Model router for cost optimization (route simple queries to cheaper models)
- Redis cache for frequent queries
- Queue-based processing for async workloads

**Cost optimization is critical at this scale:**
- Route 60-70% of simple queries to GPT-4o-mini or Claude Haiku
- Implement semantic caching (30%+ cache hit rate target)
- Use batch API for non-urgent requests (50% discount)
- At this volume, self-hosting becomes cost-competitive

**Reliability:**
- Multi-provider setup with automatic failover
- Rate limiting per user to prevent abuse
- Queue-based architecture for handling spikes
- Graceful degradation when providers are unavailable

**Monitoring:**
- Real-time cost tracking with budget alerts
- Latency percentiles (p50, p95, p99)
- Quality metrics sampled continuously
- Error rate and rate-limit hit tracking

At 1M queries with average 2K tokens, using GPT-4o would cost about $25K/day. With routing and caching, I can reduce this to $5-8K/day."

### Q: When would you self-host vs use API providers?

**Strong answer:**

"My decision framework considers several factors:

**Use API providers when:**
- Volume is under 1M queries/month (cost crossover point)
- Time-to-market is critical
- Team lacks GPU infrastructure expertise
- You want the latest models immediately
- Workload is variable and hard to predict

**Self-host when:**
- Data cannot leave your infrastructure (compliance, security)
- Volume exceeds 10M queries/month (significant savings)
- You need latency under 100ms P99
- You need custom model weights or fine-tuning
- You want full control over model behavior

**Hybrid approach often works best:**
- Self-host for high-volume predictable workloads
- API for spikes and specialized models
- API as fallback for self-hosted failures

The hidden costs of self-hosting: GPU procurement/rental, engineering time for ops, model updates, monitoring infrastructure. Factor in at least 1-2 dedicated engineers for infrastructure."

---

## References

- vLLM: https://docs.vllm.ai/
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Text Generation Inference: https://huggingface.co/docs/text-generation-inference
- OpenAI Pricing: https://openai.com/pricing
- Anthropic Pricing: https://www.anthropic.com/pricing

---

*Next: [CI/CD for LLM Applications](02-cicd.md)*

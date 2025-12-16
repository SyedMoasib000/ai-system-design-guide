# LLM Serving Architecture

This chapter covers the architecture and design patterns for serving LLMs in production.

## Table of Contents

- [Serving Options](#serving-options)
- [API Gateway Design](#api-gateway-design)
- [Request Routing](#request-routing)
- [Batching Strategies](#batching-strategies)
- [Caching](#caching)
- [Load Balancing](#load-balancing)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Serving Options

### API Providers vs Self-Hosted

| Factor | API Providers | Self-Hosted |
|--------|---------------|-------------|
| **Setup time** | Minutes | Weeks |
| **Operational burden** | None | High |
| **Cost at low volume** | Low | High (fixed) |
| **Cost at high volume** | High | Lower |
| **Latency** | Variable | Controllable |
| **Data privacy** | Shared infra | Full control |
| **Model selection** | Provider's models | Any model |

### Cost Crossover Analysis

```python
def calculate_crossover(
    api_cost_per_1m_tokens: float,
    self_hosted_fixed_monthly: float,
    self_hosted_variable_per_1m: float
) -> int:
    """
    Calculate monthly tokens where self-hosted becomes cheaper.
    """
    # API cost = tokens * api_rate
    # Self-hosted = fixed + tokens * variable_rate
    # Crossover: tokens * api = fixed + tokens * variable
    # tokens = fixed / (api - variable)
    
    if api_cost_per_1m_tokens <= self_hosted_variable_per_1m:
        return float('inf')  # API always cheaper
    
    crossover = self_hosted_fixed_monthly / (
        api_cost_per_1m_tokens - self_hosted_variable_per_1m
    )
    return int(crossover * 1_000_000)

# Example: GPT-4o vs self-hosted Llama 70B
crossover = calculate_crossover(
    api_cost_per_1m_tokens=10.0,  # GPT-4o output
    self_hosted_fixed_monthly=5000,  # 8x A100 cluster
    self_hosted_variable_per_1m=0.5  # Compute cost
)
# Crossover at ~526M tokens/month
```

---

## API Gateway Design

### Gateway Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     INGRESS LAYER                         │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │   │
│  │  │  Auth  │  │  Rate  │  │ Input  │  │Request │         │   │
│  │  │        │  │ Limit  │  │ Valid  │  │  Log   │         │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     ROUTING LAYER                         │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │   Model    │  │  Provider  │  │   Load     │         │   │
│  │  │  Selector  │  │  Selector  │  │  Balancer  │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     EGRESS LAYER                          │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │   │
│  │  │ Output │  │ Output │  │ Cost   │  │Response│         │   │
│  │  │ Valid  │  │ Filter │  │ Track  │  │  Log   │         │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class LLMGateway:
    def __init__(self):
        self.auth = AuthService()
        self.rate_limiter = RateLimiter()
        self.router = ModelRouter()
        self.providers = ProviderPool()
        self.output_filter = OutputFilter()
        self.cost_tracker = CostTracker()
    
    async def handle_request(self, request: LLMRequest) -> LLMResponse:
        # Auth
        user = await self.auth.validate(request.api_key)
        if not user:
            raise AuthenticationError()
        
        # Rate limit
        if not await self.rate_limiter.check(user.id, request.model):
            raise RateLimitError()
        
        # Input validation
        self.validate_input(request)
        
        # Route to model/provider
        model, provider = await self.router.route(request, user)
        
        # Execute
        response = await self.providers.execute(provider, model, request)
        
        # Output filtering
        filtered_response = await self.output_filter.filter(response)
        
        # Cost tracking
        await self.cost_tracker.record(user.id, request, response)
        
        return filtered_response
```

---

## Request Routing

### Model Selection Router

```python
class ModelRouter:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.model_config = {
            "simple": "gpt-4o-mini",
            "complex": "gpt-4o",
            "code": "claude-3.5-sonnet",
            "long_context": "gemini-1.5-pro"
        }
    
    async def route(self, request: LLMRequest, user: User) -> tuple[str, str]:
        # User override
        if request.model:
            return request.model, self.get_provider(request.model)
        
        # Automatic routing
        query_type = await self.classifier.classify(request.prompt)
        
        # Context length check
        if count_tokens(request.prompt) > 30000:
            query_type = "long_context"
        
        model = self.model_config[query_type]
        provider = self.get_provider(model)
        
        return model, provider
```

### Cost-Aware Routing

```python
class CostAwareRouter:
    def __init__(self, budget_per_request: float = 0.10):
        self.budget = budget_per_request
        self.pricing = ModelPricing()
    
    async def route(self, request: LLMRequest) -> str:
        estimated_tokens = self.estimate_tokens(request)
        
        # Try models from cheapest to most expensive
        models_by_cost = sorted(
            self.available_models,
            key=lambda m: self.pricing.estimate_cost(m, estimated_tokens)
        )
        
        for model in models_by_cost:
            cost = self.pricing.estimate_cost(model, estimated_tokens)
            quality = await self.estimate_quality(model, request)
            
            if cost <= self.budget and quality >= 0.8:
                return model
        
        # Fall back to cheapest if nothing fits budget
        return models_by_cost[0]
```

---

## Batching Strategies

### Dynamic Batching

```python
class DynamicBatcher:
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: int = 50
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending = []
        self.lock = asyncio.Lock()
    
    async def add_request(self, request: LLMRequest) -> LLMResponse:
        future = asyncio.Future()
        
        async with self.lock:
            self.pending.append((request, future))
            
            if len(self.pending) >= self.max_batch_size:
                await self.flush()
        
        # Wait for batch timeout if not flushed
        try:
            return await asyncio.wait_for(future, timeout=self.max_wait_ms / 1000)
        except asyncio.TimeoutError:
            async with self.lock:
                if self.pending:
                    await self.flush()
            return await future
    
    async def flush(self):
        batch = self.pending
        self.pending = []
        
        # Process batch
        responses = await self.process_batch([r for r, _ in batch])
        
        # Resolve futures
        for (_, future), response in zip(batch, responses):
            future.set_result(response)
```

### Continuous Batching

```python
class ContinuousBatcher:
    """
    For self-hosted models: add new requests to batch
    while others are still generating.
    """
    
    def __init__(self, model, max_concurrent: int = 64):
        self.model = model
        self.max_concurrent = max_concurrent
        self.active_sequences = []
    
    async def add_request(self, request: LLMRequest):
        if len(self.active_sequences) >= self.max_concurrent:
            await self.wait_for_slot()
        
        sequence = Sequence(request)
        self.active_sequences.append(sequence)
        return sequence.future
    
    async def run(self):
        while True:
            if not self.active_sequences:
                await asyncio.sleep(0.001)
                continue
            
            # Get next token for all active sequences
            prompts = [s.current_state for s in self.active_sequences]
            next_tokens = await self.model.generate_tokens(prompts)
            
            # Update sequences
            completed = []
            for seq, token in zip(self.active_sequences, next_tokens):
                seq.add_token(token)
                if seq.is_complete:
                    completed.append(seq)
                    seq.future.set_result(seq.output)
            
            # Remove completed
            self.active_sequences = [
                s for s in self.active_sequences if s not in completed
            ]
```

---

## Caching

### Semantic Cache

```python
class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600
    ):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.vector_store = VectorStore()
    
    async def get(self, query: str) -> str | None:
        query_embedding = await embed(query)
        
        results = await self.vector_store.search(
            query_embedding,
            top_k=1,
            filter={"expires_at": {"$gt": datetime.now()}}
        )
        
        if results and results[0].score > self.threshold:
            return results[0].payload["response"]
        
        return None
    
    async def set(self, query: str, response: str):
        query_embedding = await embed(query)
        
        await self.vector_store.upsert(
            id=hash(query),
            vector=query_embedding,
            payload={
                "query": query,
                "response": response,
                "expires_at": datetime.now() + timedelta(seconds=self.ttl)
            }
        )
```

### KV Cache Sharing

```python
class KVCacheManager:
    """
    Share KV cache across requests with common prefixes.
    """
    
    def __init__(self):
        self.cache = {}
    
    def get_cached_prefix(self, prompt: str) -> tuple[str, any]:
        # Find longest cached prefix
        best_match = None
        best_length = 0
        
        for prefix, kv_cache in self.cache.items():
            if prompt.startswith(prefix) and len(prefix) > best_length:
                best_match = (prefix, kv_cache)
                best_length = len(prefix)
        
        return best_match
    
    def cache_prefix(self, prefix: str, kv_cache: any):
        self.cache[prefix] = kv_cache
        
        # Evict old entries if too many
        if len(self.cache) > 1000:
            self.evict_lru()
```

---

## Load Balancing

### Weighted Load Balancer

```python
class WeightedLoadBalancer:
    def __init__(self, endpoints: list[dict]):
        self.endpoints = endpoints
        self.weights = [e["weight"] for e in endpoints]
        self.health = {e["url"]: True for e in endpoints}
    
    def select_endpoint(self) -> str:
        healthy = [
            (e, w) for e, w in zip(self.endpoints, self.weights)
            if self.health[e["url"]]
        ]
        
        if not healthy:
            raise NoHealthyEndpointsError()
        
        endpoints, weights = zip(*healthy)
        return random.choices(endpoints, weights=weights)[0]["url"]
    
    async def health_check(self):
        for endpoint in self.endpoints:
            try:
                await self.ping(endpoint["url"])
                self.health[endpoint["url"]] = True
            except:
                self.health[endpoint["url"]] = False
```

### Least-Connections

```python
class LeastConnectionsBalancer:
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.connections = {e: 0 for e in endpoints}
        self.lock = asyncio.Lock()
    
    async def select_endpoint(self) -> str:
        async with self.lock:
            endpoint = min(self.connections, key=self.connections.get)
            self.connections[endpoint] += 1
            return endpoint
    
    async def release_endpoint(self, endpoint: str):
        async with self.lock:
            self.connections[endpoint] -= 1
```

---

## Interview Questions

### Q: How would you design an LLM serving layer for a multi-tenant SaaS application?

**Strong answer:**

"The serving layer needs to handle multi-tenancy, cost allocation, and fair resource sharing.

**API Gateway:**
- Authentication: API keys mapped to tenants
- Rate limiting: per-tenant limits (requests/min, tokens/day)
- Request validation: input size limits, content filters

**Routing:**
- Model selection based on tenant tier (premium gets GPT-4o, free gets mini)
- Cost-aware routing within tenant budget
- Provider failover for reliability

**Isolation:**
- Tenant ID in all request context
- Separate queues per tenant to prevent noisy neighbor
- Cost tracking per tenant

**Caching:**
- Semantic cache with tenant-scoped keys
- Common queries across tenants can share cache (if not sensitive)

**Monitoring:**
- Per-tenant metrics: usage, latency, error rate
- Cross-tenant: total capacity, provider health
- Alerting on per-tenant anomalies

The key is that no single tenant should be able to degrade service for others."

### Q: When would you choose self-hosted models over API providers?

**Strong answer:**

"Self-hosting makes sense when:

**Data sensitivity:** Regulated data (healthcare, finance) that cannot leave infrastructure. Even with BAAs, some organizations prefer full control.

**Scale economics:** At very high volume (10M+ requests/month), self-hosted becomes cheaper. The crossover depends on model size and hardware efficiency.

**Latency requirements:** API providers add network latency. For < 100ms P99 requirements, self-hosted with local inference is necessary.

**Customization:** Need to run fine-tuned models or specialized architectures that providers do not offer.

**I would not self-host when:**
- Volume is low (< 1M requests/month)
- Team lacks GPU/ML infrastructure expertise
- Workload is spiky (hard to capacity plan)
- Need frequent model updates (providers update automatically)

Hybrid often works: self-host for high-volume predictable workloads, API for spikes and specialized models."

---

## References

- vLLM: https://docs.vllm.ai/
- TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/

---

*Next: [CI/CD for LLM Applications](02-cicd.md)*

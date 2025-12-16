# LLM Observability

Observability for LLM systems requires adapting the three pillars of logs, metrics, and traces for the unique characteristics of AI applications.

## Table of Contents

- [Why LLM Observability is Different](#why-llm-observability-is-different)
- [The Three Pillars](#the-three-pillars)
- [Key Metrics](#key-metrics)
- [Tracing LLM Pipelines](#tracing-llm-pipelines)
- [Quality Monitoring](#quality-monitoring)
- [Cost Tracking](#cost-tracking)
- [Alerting Strategy](#alerting-strategy)
- [Observability Tools](#observability-tools)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why LLM Observability is Different

Traditional observability focuses on:
- Request/response patterns
- Latency and throughput
- Error rates
- Resource utilization

LLM systems add:
- **Quality is a first-class metric**: A fast, available system producing bad outputs is failing
- **Non-determinism**: Same input can produce different outputs
- **Token economics**: Cost scales with usage in complex ways
- **Multi-component pipelines**: RAG has retrieval, reranking, generation steps
- **Subjective correctness**: Often no ground truth to compare against

---

## The Three Pillars

### Logging

```python
class LLMLogger:
    def log_request(
        self,
        request_id: str,
        model: str,
        messages: list[dict],
        parameters: dict
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "type": "llm_request",
            "model": model,
            "parameters": parameters,
            "input_tokens": self.count_tokens(messages),
            # Hash for privacy, full content in secure store
            "content_hash": self.hash_content(messages)
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_response(
        self,
        request_id: str,
        response: str,
        latency_ms: float,
        tokens: dict
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "type": "llm_response",
            "latency_ms": latency_ms,
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
            "ttft_ms": tokens.get("ttft_ms"),
            "content_hash": self.hash_content(response)
        }
        self.logger.info(json.dumps(log_entry))
```

**What to log:**
- Request ID for correlation
- Model and parameters
- Token counts
- Latency (TTFT and total)
- Content (hashed if privacy-sensitive)

### Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"]
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM request latency",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

llm_ttft_seconds = Histogram(
    "llm_ttft_seconds",
    "Time to first token",
    ["model"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# Token metrics
tokens_used_total = Counter(
    "tokens_used_total",
    "Total tokens consumed",
    ["model", "direction"]  # direction: input/output
)

# Cost metrics
llm_cost_dollars = Counter(
    "llm_cost_dollars",
    "LLM cost in dollars",
    ["model"]
)

# Quality metrics (sampled)
quality_score = Gauge(
    "llm_quality_score",
    "Sampled quality score",
    ["model", "task_type"]
)
```

### Traces

End-to-end tracing for RAG pipelines:

```python
from opentelemetry import trace

tracer = trace.get_tracer("rag_pipeline")

async def rag_query(query: str) -> str:
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query", query)
        
        # Embedding step
        with tracer.start_as_current_span("embed_query") as embed_span:
            query_embedding = await embed(query)
            embed_span.set_attribute("embedding_dim", len(query_embedding))
        
        # Retrieval step
        with tracer.start_as_current_span("vector_search") as search_span:
            results = await vector_db.search(query_embedding, top_k=10)
            search_span.set_attribute("results_count", len(results))
            search_span.set_attribute("top_score", results[0].score if results else 0)
        
        # Reranking step
        with tracer.start_as_current_span("rerank") as rerank_span:
            reranked = await reranker.rerank(query, results)
            rerank_span.set_attribute("reranked_count", len(reranked))
        
        # Generation step
        with tracer.start_as_current_span("generate") as gen_span:
            response = await llm.generate(query, context=reranked[:5])
            gen_span.set_attribute("model", llm.model)
            gen_span.set_attribute("output_tokens", count_tokens(response))
        
        return response
```

---

## Key Metrics

### Operational Metrics

| Metric | Description | Typical Alert Threshold |
|--------|-------------|------------------------|
| Request rate | Requests per second | Anomaly detection |
| Error rate | Failed requests / total | > 5% |
| Latency p50 | Median response time | > 2s |
| Latency p95 | 95th percentile | > 5s |
| Latency p99 | 99th percentile | > 10s |
| TTFT | Time to first token | > 1s |
| Token throughput | Tokens per second | < baseline |

### Quality Metrics

| Metric | Description | Collection Method |
|--------|-------------|-------------------|
| Quality score | LLM-as-judge rating | Sampled (1-5%) |
| Faithfulness | RAG answer grounded in context | Sampled |
| Relevance | Answer addresses the question | Sampled |
| User satisfaction | Thumbs up/down, ratings | User feedback |
| Task completion | Did user achieve goal? | Implicit signals |

### Cost Metrics

| Metric | Description | Granularity |
|--------|-------------|-------------|
| Cost per request | Average cost | Per model |
| Daily cost | Total daily spend | Overall + per model |
| Cost per user action | Cost to complete user goal | Per task type |
| Token efficiency | Value delivered per token | Per use case |

---

## Quality Monitoring

### Sampling Strategy

```python
class QualitySampler:
    def __init__(self, sample_rate: float = 0.05):
        self.sample_rate = sample_rate
        self.judge = LLMJudge()
    
    async def maybe_evaluate(
        self,
        request_id: str,
        query: str,
        context: list[str],
        response: str
    ):
        # Sample randomly
        if random.random() > self.sample_rate:
            return
        
        # Evaluate quality
        scores = await self.judge.evaluate(
            query=query,
            context=context,
            response=response,
            criteria=["relevance", "faithfulness", "helpfulness"]
        )
        
        # Record metrics
        for criterion, score in scores.items():
            quality_score.labels(
                model=self.model,
                criterion=criterion
            ).set(score)
        
        # Store for analysis
        await self.store_evaluation(request_id, scores)
```

### Drift Detection

```python
class QualityDriftDetector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.baseline_scores = []
        self.current_scores = []
    
    def add_score(self, score: float):
        self.current_scores.append(score)
        
        if len(self.current_scores) >= self.window_size:
            self.check_drift()
            self.current_scores = []
    
    def check_drift(self):
        if not self.baseline_scores:
            self.baseline_scores = self.current_scores.copy()
            return
        
        # Statistical test for drift
        baseline_mean = np.mean(self.baseline_scores)
        current_mean = np.mean(self.current_scores)
        
        # Simple threshold-based detection
        drift_threshold = 0.1  # 10% degradation
        if (baseline_mean - current_mean) / baseline_mean > drift_threshold:
            self.alert_drift(baseline_mean, current_mean)
    
    def alert_drift(self, baseline: float, current: float):
        alert = {
            "type": "quality_drift",
            "baseline_score": baseline,
            "current_score": current,
            "degradation_pct": (baseline - current) / baseline * 100
        }
        self.send_alert(alert)
```

---

## Cost Tracking

### Real-Time Cost Calculation

```python
class CostTracker:
    # Pricing per 1M tokens (verify current rates)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 0.25, "output": 1.25},
    }
    
    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: str
    ) -> float:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Record metrics
        llm_cost_dollars.labels(model=model).inc(total_cost)
        tokens_used_total.labels(model=model, direction="input").inc(input_tokens)
        tokens_used_total.labels(model=model, direction="output").inc(output_tokens)
        
        # Log for analysis
        self.log_cost(request_id, model, input_tokens, output_tokens, total_cost)
        
        return total_cost
```

### Cost Attribution

```python
class CostAttributor:
    def attribute_cost(
        self,
        request_id: str,
        user_id: str,
        team: str,
        use_case: str,
        cost: float
    ):
        # Store for billing and analysis
        attribution = {
            "request_id": request_id,
            "user_id": user_id,
            "team": team,
            "use_case": use_case,
            "cost": cost,
            "timestamp": datetime.utcnow()
        }
        
        self.store(attribution)
        
        # Update running totals
        self.update_user_total(user_id, cost)
        self.update_team_total(team, cost)
        
        # Check budgets
        if self.exceeds_budget(team):
            self.alert_budget_exceeded(team)
```

---

## Alerting Strategy

### Alert Configuration

```yaml
alerts:
  # Availability
  - name: high_error_rate
    condition: error_rate > 0.05
    for: 5m
    severity: critical
    runbook: "Check provider status, verify API keys, review recent changes"
    
  # Latency
  - name: high_latency_p95
    condition: latency_p95 > 10s
    for: 5m
    severity: warning
    runbook: "Check model, reduce context size, verify provider status"
    
  # Cost
  - name: cost_spike
    condition: hourly_cost > 2 * rolling_avg_hourly_cost
    for: 1h
    severity: warning
    runbook: "Check for traffic spike, review recent deployments, verify caching"
    
  # Quality
  - name: quality_degradation
    condition: avg_quality_score < 3.5 over 1h
    for: 30m
    severity: warning
    runbook: "Review recent changes, check model performance, sample responses"
    
  # Resource
  - name: rate_limit_approaching
    condition: rate_limit_usage > 0.8
    for: 15m
    severity: warning
    runbook: "Consider model routing, implement backpressure"
```

### Alert Prioritization

| Severity | Response Time | Examples |
|----------|---------------|----------|
| Critical | < 15 min | Service down, > 50% error rate |
| High | < 1 hour | > 10% error rate, P99 > 30s |
| Warning | < 4 hours | Quality degradation, cost spike |
| Info | Next business day | Trend changes, capacity planning |

---

## Observability Tools

### LLM-Specific Tools

| Tool | Focus | Best For |
|------|-------|----------|
| LangSmith | LangChain tracing | LangChain-based apps |
| Langfuse | Open source tracing | Self-hosted, privacy |
| Weights & Biases | Experiment tracking | ML teams |
| Arize Phoenix | LLM monitoring | Production monitoring |
| Helicone | API proxy logging | Simple integration |

### Integration Example: Langfuse

```python
from langfuse import Langfuse

langfuse = Langfuse()

async def traced_rag_query(query: str) -> str:
    # Start trace
    trace = langfuse.trace(name="rag_query", input=query)
    
    # Embedding span
    embed_span = trace.span(name="embed")
    embedding = await embed(query)
    embed_span.end()
    
    # Retrieval span
    retrieve_span = trace.span(name="retrieve")
    results = await vector_db.search(embedding)
    retrieve_span.end(output={"count": len(results)})
    
    # Generation span
    gen_span = trace.generation(
        name="generate",
        model="gpt-4o",
        input={"query": query, "context": results}
    )
    response = await llm.generate(query, context=results)
    gen_span.end(output=response)
    
    # End trace
    trace.update(output=response)
    
    return response
```

---

## Interview Questions

### Q: What metrics would you track for a production LLM system?

**Strong answer:**

"I organize metrics into three categories:

**Operational metrics:** These are table stakes for any service.
- Request rate and error rate
- Latency percentiles: p50, p95, p99
- Time to first token (TTFT) for streaming
- Availability

**Quality metrics:** This is what makes LLM observability unique.
- Sampled quality scores using LLM-as-judge (1-5% sample rate)
- For RAG: faithfulness and relevance scores
- User feedback: thumbs up/down, explicit ratings
- Task completion rate where measurable

**Cost metrics:**
- Cost per request by model
- Daily/weekly cost trends
- Cost per successful user action
- Token efficiency

I set alerts for operational issues (error rate > 5%, P95 > SLA) and quality drift (average score drops 10% from baseline). Cost alerts for spikes help catch runaway usage.

The key insight is that a fast, available LLM system producing bad outputs is still failing. Quality must be a first-class metric."

### Q: How do you detect quality degradation in production?

**Strong answer:**

"I use several approaches:

**Continuous sampling:** I evaluate 1-5% of requests using LLM-as-judge. This gives me a quality signal without evaluating everything.

**Drift detection:** I maintain a baseline quality distribution and use statistical tests to detect when current scores drift significantly. A 10% degradation triggers a warning.

**User feedback:** Thumbs up/down, explicit ratings if available. This is ground truth for user satisfaction.

**Implicit signals:** Task completion, retry rate, escalation rate, session length. If users are struggling more, quality may have dropped.

**What to do when I detect degradation:**
1. Check for recent deployments or prompt changes
2. Sample specific responses to diagnose the issue
3. Check if it is model-specific (provider issue) or universal
4. Roll back if necessary, then investigate

I also maintain a golden test set of queries with expected behaviors that I run on every deployment to catch regressions before production."

---

## References

- OpenTelemetry: https://opentelemetry.io/
- Langfuse: https://langfuse.com/docs
- LangSmith: https://docs.smith.langchain.com/

---

*Next: [CI/CD for LLM Applications](../11-infrastructure-and-mlops/02-cicd.md)*

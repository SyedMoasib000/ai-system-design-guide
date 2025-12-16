# Reliability Patterns

Production LLM systems need robust reliability patterns beyond basic retry logic. This chapter covers advanced patterns for building resilient AI applications.

## Table of Contents

- [Reliability Challenges](#reliability-challenges)
- [Retry Patterns](#retry-patterns)
- [Circuit Breaker](#circuit-breaker)
- [Bulkhead Pattern](#bulkhead-pattern)
- [Timeout Strategies](#timeout-strategies)
- [Graceful Degradation](#graceful-degradation)
- [Multi-Provider Failover](#multi-provider-failover)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Reliability Challenges

### LLM-Specific Failure Modes

| Failure Mode | Cause | Impact |
|--------------|-------|--------|
| Rate limiting | Quota exceeded | Request rejection |
| Timeouts | Long generation, network issues | Slow/failed responses |
| Provider outage | Infrastructure issues | Complete failure |
| Quality degradation | Model updates, load | Worse outputs |
| Context overflow | Input too large | Request failure |
| Malformed output | Generation errors | Parsing failures |

### Reliability Targets

| Tier | Availability | Latency p99 | Examples |
|------|--------------|-------------|----------|
| Critical | 99.99% | < 3s | Payment processing |
| Standard | 99.9% | < 10s | Customer support |
| Best effort | 99% | < 30s | Background tasks |

---

## Retry Patterns

### Exponential Backoff with Jitter

```python
import random
import asyncio
from typing import TypeVar, Callable

T = TypeVar("T")

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: float = 0.5
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        # Add jitter to prevent thundering herd
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


async def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig,
    retryable_exceptions: tuple = (Exception,)
) -> T:
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_retries:
                break
            
            delay = config.get_delay(attempt)
            await asyncio.sleep(delay)
    
    raise last_exception
```

### Retryable vs Non-Retryable Errors

```python
class LLMRetryPolicy:
    RETRYABLE = [
        RateLimitError,
        TimeoutError,
        ServiceUnavailableError,
        ConnectionError
    ]
    
    NOT_RETRYABLE = [
        AuthenticationError,
        InvalidRequestError,
        ContentPolicyViolation,
        ContextLengthExceeded
    ]
    
    @classmethod
    def should_retry(cls, error: Exception) -> bool:
        for retryable_type in cls.RETRYABLE:
            if isinstance(error, retryable_type):
                return True
        return False
    
    @classmethod
    def get_retry_after(cls, error: Exception) -> float | None:
        # Some rate limit errors include retry-after header
        if hasattr(error, "retry_after"):
            return error.retry_after
        return None
```

---

## Circuit Breaker

### Implementation

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=30)
    half_open_max_calls: int = 3
    success_threshold: int = 2  # Successes needed to close

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._recovery_timeout_elapsed():
                self._transition_to_half_open()
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_open(self):
        self.state = CircuitState.OPEN
        self.success_count = 0
    
    def _transition_to_half_open(self):
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _recovery_timeout_elapsed(self) -> bool:
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time >= self.config.recovery_timeout
```

### Usage with LLM Client

```python
class ResilientLLMClient:
    def __init__(self):
        self.circuit_breakers = {
            "openai": CircuitBreaker("openai", CircuitBreakerConfig()),
            "anthropic": CircuitBreaker("anthropic", CircuitBreakerConfig()),
        }
    
    async def generate(self, prompt: str, provider: str = "openai") -> str:
        cb = self.circuit_breakers[provider]
        
        if not cb.can_execute():
            raise CircuitOpenError(f"Circuit breaker open for {provider}")
        
        try:
            result = await self._call_provider(provider, prompt)
            cb.record_success()
            return result
        except RetryableError as e:
            cb.record_failure()
            raise
```

---

## Bulkhead Pattern

### Isolating Resources

```python
import asyncio
from contextlib import asynccontextmanager

class Bulkhead:
    """
    Isolate resources to prevent cascade failures.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int,
        max_queued: int = 100
    ):
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_semaphore = asyncio.Semaphore(max_queued)
    
    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0):
        # Check queue capacity
        if not self.queue_semaphore.locked():
            await self.queue_semaphore.acquire()
        else:
            raise BulkheadFullError(f"Bulkhead {self.name} queue full")
        
        try:
            # Wait for execution slot
            acquired = await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=timeout
            )
            self.queue_semaphore.release()
            
            try:
                yield
            finally:
                self.semaphore.release()
        except asyncio.TimeoutError:
            self.queue_semaphore.release()
            raise BulkheadTimeoutError(f"Bulkhead {self.name} timeout")


class BulkheadedLLMClient:
    def __init__(self):
        # Separate bulkheads for different workloads
        self.bulkheads = {
            "realtime": Bulkhead("realtime", max_concurrent=50),
            "batch": Bulkhead("batch", max_concurrent=200),
            "critical": Bulkhead("critical", max_concurrent=10)
        }
    
    async def generate(
        self,
        prompt: str,
        priority: str = "realtime"
    ) -> str:
        bulkhead = self.bulkheads[priority]
        
        async with bulkhead.acquire():
            return await self._call_llm(prompt)
```

---

## Timeout Strategies

### Layered Timeouts

```python
class TimeoutConfig:
    def __init__(
        self,
        connection_timeout: float = 5.0,
        read_timeout: float = 30.0,
        total_timeout: float = 60.0
    ):
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
        self.total_timeout = total_timeout


class TimeoutManager:
    def __init__(self, config: TimeoutConfig):
        self.config = config
    
    async def execute_with_timeout(self, func, *args, **kwargs):
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.total_timeout
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Request timed out after {self.config.total_timeout}s"
            )
```

### Adaptive Timeouts

```python
class AdaptiveTimeout:
    """
    Adjust timeouts based on observed latency.
    """
    
    def __init__(
        self,
        initial_timeout: float = 30.0,
        min_timeout: float = 10.0,
        max_timeout: float = 120.0,
        percentile: float = 0.99
    ):
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.latencies: list[float] = []
        self.current_timeout = initial_timeout
    
    def record_latency(self, latency: float):
        self.latencies.append(latency)
        
        # Keep last 1000 observations
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
        
        # Update timeout to percentile + buffer
        if len(self.latencies) >= 10:
            sorted_latencies = sorted(self.latencies)
            idx = int(len(sorted_latencies) * self.percentile)
            p99_latency = sorted_latencies[idx]
            
            # Add 20% buffer
            new_timeout = p99_latency * 1.2
            self.current_timeout = max(
                self.min_timeout,
                min(self.max_timeout, new_timeout)
            )
    
    def get_timeout(self) -> float:
        return self.current_timeout
```

---

## Graceful Degradation

### Degradation Levels

```python
class DegradationLevel(Enum):
    FULL = "full"           # All features
    REDUCED = "reduced"     # Fewer features
    MINIMAL = "minimal"     # Core only
    CACHED = "cached"       # Cached responses only
    OFFLINE = "offline"     # Error message

class GracefulDegrader:
    def __init__(self):
        self.current_level = DegradationLevel.FULL
        self.health_checker = HealthChecker()
    
    async def get_response(self, query: str) -> str:
        level = await self.health_checker.get_degradation_level()
        
        if level == DegradationLevel.FULL:
            return await self.full_pipeline(query)
        
        elif level == DegradationLevel.REDUCED:
            # Skip expensive operations
            return await self.reduced_pipeline(query)
        
        elif level == DegradationLevel.MINIMAL:
            # Simpler model, no retrieval
            return await self.minimal_pipeline(query)
        
        elif level == DegradationLevel.CACHED:
            # Only return cached responses
            cached = await self.cache.get_similar(query)
            if cached:
                return cached
            return "I'm experiencing issues. Please try again later."
        
        else:
            return "Service temporarily unavailable."
    
    async def full_pipeline(self, query: str) -> str:
        # RAG + frontier model + ensemble verification
        context = await self.retrieve(query)
        response = await self.generate(query, context, model="gpt-4o")
        verified = await self.verify(response)
        return verified
    
    async def reduced_pipeline(self, query: str) -> str:
        # RAG + smaller model, no verification
        context = await self.retrieve(query)
        return await self.generate(query, context, model="gpt-4o-mini")
    
    async def minimal_pipeline(self, query: str) -> str:
        # Direct generation with smallest model
        return await self.generate(query, None, model="gpt-4o-mini")
```

---

## Multi-Provider Failover

### Provider Manager

```python
class ProviderManager:
    def __init__(self):
        self.providers = {
            "primary": OpenAIProvider(),
            "secondary": AnthropicProvider(),
            "tertiary": GoogleProvider()
        }
        self.health = {name: True for name in self.providers}
        self.priority_order = ["primary", "secondary", "tertiary"]
    
    async def generate(self, request: dict) -> str:
        for provider_name in self.priority_order:
            if not self.health[provider_name]:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                result = await provider.generate(request)
                return result
            except RetryableError as e:
                # Mark unhealthy but continue to next provider
                self.health[provider_name] = False
                asyncio.create_task(
                    self._health_check_later(provider_name)
                )
                continue
        
        raise AllProvidersUnavailableError()
    
    async def _health_check_later(self, provider_name: str):
        await asyncio.sleep(30)  # Wait before retrying
        try:
            await self.providers[provider_name].health_check()
            self.health[provider_name] = True
        except:
            # Schedule another check
            asyncio.create_task(self._health_check_later(provider_name))
```

### Request Hedging

```python
class HedgedRequest:
    """
    Send parallel requests to multiple providers, use first response.
    """
    
    def __init__(self, providers: list, hedge_delay: float = 2.0):
        self.providers = providers
        self.hedge_delay = hedge_delay
    
    async def generate(self, request: dict) -> str:
        # Start primary request
        tasks = [asyncio.create_task(self.providers[0].generate(request))]
        
        try:
            # Wait for primary with hedge delay
            result = await asyncio.wait_for(tasks[0], timeout=self.hedge_delay)
            return result
        except asyncio.TimeoutError:
            # Primary slow, start hedged requests
            for provider in self.providers[1:]:
                tasks.append(asyncio.create_task(provider.generate(request)))
            
            # Return first successful result
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending
            for task in pending:
                task.cancel()
            
            # Get result from completed task
            for task in done:
                if task.exception() is None:
                    return task.result()
            
            # All failed
            raise AllProvidersFailedError()
```

---

## Interview Questions

### Q: How do you design for high availability in LLM systems?

**Strong answer:**

"I use multiple layers of reliability:

**Retry with backoff:** Exponential backoff with jitter for transient failures. Important to distinguish retryable (rate limits, timeouts) from non-retryable (auth, bad request) errors.

**Circuit breaker:** If a provider fails repeatedly, stop trying for a cooldown period. This prevents wasting latency on a dead provider and gives it time to recover.

**Multi-provider failover:** Never depend on a single provider. I configure primary/secondary/tertiary with automatic failover. Each provider has its own circuit breaker.

**Graceful degradation:** Define what happens when no providers are available. Better to return a degraded response (simpler model, cached result) than to fail completely.

**Bulkheading:** Isolate different workloads. A batch processing surge should not take down real-time queries.

The key insight is assuming failure. LLM APIs are less reliable than traditional APIs. Design as if the provider will go down, because it will."

### Q: What is the difference between circuit breaker and retry?

**Strong answer:**

"They solve different problems:

**Retry** handles transient failures. If a single request fails, try again. It assumes failures are independent and the next attempt may succeed.

**Circuit breaker** handles systemic failures. If many requests are failing, stop trying entirely. It assumes the downstream system is unhealthy and repeated attempts waste resources and slow recovery.

**How they work together:**
1. Request fails → retry with backoff (attempt 1, 2, 3)
2. If all retries fail → circuit breaker records failure
3. After N failures → circuit opens, rejects requests immediately
4. After timeout → circuit half-opens, allows limited test requests
5. If tests succeed → circuit closes, normal operation resumes

Without circuit breaker: during an outage, every request waits through all retries before failing. Latency spikes, resources exhausted.

With circuit breaker: after detecting the outage, requests fail fast. System remains responsive, can fail over to alternatives."

---

## References

- Microsoft Resilience Patterns: https://learn.microsoft.com/en-us/azure/architecture/patterns/
- Netflix Hystrix: https://github.com/Netflix/Hystrix

---

*Next: [Safety and Guardrails](02-guardrails.md)*

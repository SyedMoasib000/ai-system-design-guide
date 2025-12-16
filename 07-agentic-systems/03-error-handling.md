# Error Handling and Recovery in Agentic Systems

Agentic systems fail in complex ways. This chapter covers error handling patterns, recovery strategies, and building resilient agents.

## Table of Contents

- [Agent Failure Modes](#agent-failure-modes)
- [Error Detection](#error-detection)
- [Recovery Strategies](#recovery-strategies)
- [Guardrails for Agents](#guardrails-for-agents)
- [Human-in-the-Loop Patterns](#human-in-the-loop-patterns)
- [Testing Agent Reliability](#testing-agent-reliability)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Agent Failure Modes

### Taxonomy of Agent Failures

| Failure Type | Description | Frequency | Impact |
|--------------|-------------|-----------|--------|
| Tool selection error | Picks wrong tool for task | Common | Medium |
| Argument error | Wrong parameters to tool | Common | Low-Medium |
| Reasoning loop | Repeats same action | Uncommon | High |
| Runaway execution | Excessive iterations/cost | Uncommon | High |
| Hallucinated tool | Calls non-existent tool | Rare | Medium |
| Goal drift | Strays from original task | Common | Medium |
| Partial completion | Stops before finishing | Common | Medium |

### Root Causes

```
Agent Failures
├── Model Failures
│   ├── Hallucination (invents facts/tools)
│   ├── Instruction following (ignores constraints)
│   └── Reasoning errors (flawed logic)
│
├── Tool Failures
│   ├── Tool errors (exceptions, timeouts)
│   ├── Rate limits (external APIs)
│   └── Unexpected responses (schema changes)
│
├── State Failures
│   ├── Context overflow (exceeds token limit)
│   ├── State corruption (incorrect memory)
│   └── Checkpoint failures (cannot resume)
│
└── Design Failures
    ├── Ambiguous tool descriptions
    ├── Missing error guidance
    └── Insufficient stopping criteria
```

---

## Error Detection

### Detection Patterns

```python
class AgentErrorDetector:
    def __init__(self):
        self.max_iterations = 20
        self.max_cost = 1.0  # dollars
        self.action_history = []
    
    def detect_errors(self, action: dict, observation: str) -> list[str]:
        errors = []
        
        # Detect loops
        if self.is_loop(action):
            errors.append("LOOP_DETECTED")
        
        # Detect runaway
        if self.iteration_count > self.max_iterations:
            errors.append("MAX_ITERATIONS_EXCEEDED")
        
        if self.total_cost > self.max_cost:
            errors.append("COST_LIMIT_EXCEEDED")
        
        # Detect tool errors
        if "error" in observation.lower() or "failed" in observation.lower():
            errors.append("TOOL_ERROR")
        
        # Detect hallucinated tools
        if action["tool"] not in self.available_tools:
            errors.append("UNKNOWN_TOOL")
        
        # Detect goal drift
        if self.is_off_topic(action):
            errors.append("GOAL_DRIFT")
        
        return errors
    
    def is_loop(self, action: dict) -> bool:
        # Check if last N actions are identical
        self.action_history.append(self.normalize_action(action))
        if len(self.action_history) >= 3:
            recent = self.action_history[-3:]
            if len(set(recent)) == 1:
                return True
        return False
    
    def is_off_topic(self, action: dict) -> bool:
        # Use embedding similarity to detect drift from original goal
        action_embedding = embed(str(action))
        goal_embedding = embed(self.original_goal)
        similarity = cosine_similarity(action_embedding, goal_embedding)
        return similarity < 0.3
```

### Observation Validation

```python
class ObservationValidator:
    def validate(self, tool_name: str, expected_schema: dict, observation: str) -> dict:
        result = {"valid": True, "errors": [], "parsed": None}
        
        # Check for error indicators
        if self.contains_error(observation):
            result["valid"] = False
            result["errors"].append("Tool returned error")
            return result
        
        # Parse if JSON expected
        if expected_schema.get("type") == "json":
            try:
                parsed = json.loads(observation)
                result["parsed"] = parsed
                
                # Validate against schema
                schema_errors = self.validate_schema(parsed, expected_schema)
                if schema_errors:
                    result["valid"] = False
                    result["errors"].extend(schema_errors)
            except json.JSONDecodeError:
                result["valid"] = False
                result["errors"].append("Invalid JSON response")
        
        # Check for empty response
        if not observation or observation.strip() == "":
            result["valid"] = False
            result["errors"].append("Empty response")
        
        return result
```

---

## Recovery Strategies

### Retry with Backoff

```python
class RetryHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def execute_with_retry(self, func, *args, **kwargs):
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError:
                delay = self.base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            except ToolError as e:
                last_error = e
                # Some tool errors are not retryable
                if e.is_permanent:
                    raise
        
        raise MaxRetriesExceeded(last_error)
```

### Error-Informed Replanning

```python
class ReplanningAgent:
    async def handle_error(self, error: str, context: dict) -> str:
        replan_prompt = f"""
Previous action failed with error: {error}

Current context:
- Original goal: {context['goal']}
- Completed steps: {context['completed']}
- Failed action: {context['failed_action']}

Analyze what went wrong and suggest an alternative approach.
If the goal is unachievable, explain why.
If you can work around the error, provide revised steps.
"""
        
        new_plan = await self.llm.generate(replan_prompt)
        return new_plan
    
    async def run(self, goal: str) -> str:
        plan = await self.create_plan(goal)
        
        for step in plan.steps:
            try:
                result = await self.execute_step(step)
            except ToolError as e:
                # Replan around the error
                new_plan = await self.handle_error(
                    str(e),
                    {
                        "goal": goal,
                        "completed": self.completed_steps,
                        "failed_action": step
                    }
                )
                return await self.run_plan(new_plan)
        
        return await self.synthesize_result()
```

### Fallback Chain

```python
class FallbackChain:
    def __init__(self, strategies: list):
        self.strategies = strategies
    
    async def execute(self, task: dict) -> dict:
        errors = []
        
        for strategy in self.strategies:
            try:
                result = await strategy.execute(task)
                if result["success"]:
                    return result
            except Exception as e:
                errors.append({
                    "strategy": strategy.name,
                    "error": str(e)
                })
                continue
        
        return {
            "success": False,
            "errors": errors,
            "fallback_response": self.get_fallback_response(task)
        }

# Define fallback chain
fallback = FallbackChain([
    PrimaryAgent(),           # Full agent with all tools
    SimplifiedAgent(),        # Agent with safe tools only
    DirectRAGResponse(),      # No agent, just RAG
    HumanEscalation()         # Forward to human
])
```

---

## Guardrails for Agents

### Action Validation

```python
class ActionGuardrail:
    def __init__(self, allowed_tools: list, constraints: dict):
        self.allowed_tools = allowed_tools
        self.constraints = constraints
    
    def validate(self, action: dict) -> tuple[bool, str]:
        # Check tool is allowed
        if action["tool"] not in self.allowed_tools:
            return False, f"Tool {action['tool']} not allowed"
        
        # Check tool-specific constraints
        tool_constraints = self.constraints.get(action["tool"], {})
        
        # Check argument limits
        if "max_results" in tool_constraints:
            if action["args"].get("limit", 0) > tool_constraints["max_results"]:
                return False, "Exceeds maximum results limit"
        
        # Check dangerous patterns
        if self.contains_dangerous_pattern(action):
            return False, "Action contains dangerous pattern"
        
        return True, ""
    
    def contains_dangerous_pattern(self, action: dict) -> bool:
        dangerous_patterns = [
            r"rm\s+-rf",
            r"DROP\s+TABLE",
            r"DELETE\s+FROM.*WHERE\s+1=1",
        ]
        
        action_str = json.dumps(action)
        for pattern in dangerous_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return True
        return False
```

### Cost and Rate Limiting

```python
class AgentLimiter:
    def __init__(
        self,
        max_iterations: int = 20,
        max_cost: float = 1.0,
        max_time: int = 300
    ):
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        self.max_time = max_time
        self.start_time = None
        self.iterations = 0
        self.cost = 0.0
    
    def check_limits(self) -> tuple[bool, str]:
        if self.iterations >= self.max_iterations:
            return False, "Maximum iterations exceeded"
        
        if self.cost >= self.max_cost:
            return False, "Cost limit exceeded"
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_time:
            return False, "Time limit exceeded"
        
        return True, ""
    
    def record_step(self, tokens_used: int, model: str):
        self.iterations += 1
        self.cost += self.calculate_cost(tokens_used, model)
```

---

## Human-in-the-Loop Patterns

### Confidence-Based Escalation

```python
class HumanEscalationHandler:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    async def maybe_escalate(
        self,
        response: str,
        confidence: float,
        context: dict
    ) -> dict:
        if confidence >= self.confidence_threshold:
            return {"action": "proceed", "response": response}
        
        # Create escalation request
        escalation = await self.create_escalation(response, confidence, context)
        
        # Notify human
        await self.notify_human(escalation)
        
        # Return waiting response
        return {
            "action": "escalated",
            "message": "I want to make sure I give you accurate information. I have escalated your question to a human expert.",
            "escalation_id": escalation["id"]
        }
    
    async def create_escalation(self, response: str, confidence: float, context: dict) -> dict:
        return {
            "id": generate_id(),
            "original_query": context["query"],
            "ai_response": response,
            "confidence": confidence,
            "context": context,
            "status": "pending"
        }
```

### Approval Workflow

```python
class ApprovalWorkflow:
    async def request_approval(
        self,
        action: dict,
        reason: str
    ) -> dict:
        approval_request = {
            "id": generate_id(),
            "action": action,
            "reason": reason,
            "requested_at": datetime.now(),
            "status": "pending"
        }
        
        await self.save_request(approval_request)
        await self.notify_approvers(approval_request)
        
        # Wait for approval with timeout
        try:
            approval = await asyncio.wait_for(
                self.wait_for_approval(approval_request["id"]),
                timeout=3600  # 1 hour timeout
            )
            return approval
        except asyncio.TimeoutError:
            return {"approved": False, "reason": "Timeout"}
    
    def requires_approval(self, action: dict) -> bool:
        # Define actions that need human approval
        high_risk_actions = ["delete", "modify", "send_email", "purchase"]
        return action["tool"] in high_risk_actions
```

---

## Testing Agent Reliability

### Adversarial Testing

```python
class AgentAdversarialTester:
    def __init__(self, agent):
        self.agent = agent
        self.test_cases = self.load_test_cases()
    
    async def run_tests(self) -> dict:
        results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for test in self.test_cases:
            try:
                response = await self.agent.run(test["input"])
                
                if test["type"] == "should_complete":
                    if response["status"] == "complete":
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "test": test["name"],
                            "error": "Did not complete"
                        })
                
                elif test["type"] == "should_refuse":
                    if response["status"] == "refused":
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "test": test["name"],
                            "error": "Did not refuse dangerous request"
                        })
                
                elif test["type"] == "should_not_loop":
                    if response["iterations"] < 10:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "test": test["name"],
                            "error": f"Looped {response['iterations']} times"
                        })
                        
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "test": test["name"],
                    "error": str(e)
                })
        
        return results
```

---

## Interview Questions

### Q: How do you handle agent loops?

**Strong answer:**

"Loops are when an agent repeats the same action expecting different results. I handle them at multiple levels:

**Detection:**
I track the last N actions and detect when consecutive actions are identical or semantically equivalent. I use a hash of normalized actions for exact matches and embeddings for semantic similarity.

**Prevention:**
Clear stopping criteria in the system prompt. If a tool returns an error, I include explicit guidance: 'If this tool fails, try a different approach.'

**Recovery:**
When a loop is detected, I inject context into the next prompt: 'Your last 3 actions were identical. This approach is not working. Consider a different strategy.' This often breaks the loop.

**Hard limits:**
Maximum iterations (typically 15-20) as a backstop. If reached, I gracefully terminate with a partial result rather than erroring.

The key is distinguishing intentional repetition (legitimate) from stuck loops (problem). Context matters."

### Q: When should an agent escalate to a human?

**Strong answer:**

"I use confidence-based escalation with multiple triggers:

**Low confidence:** If the agent's confidence score falls below a threshold (typically 0.6-0.7), I escalate. This catches cases where the agent is uncertain.

**High-stakes actions:** Certain actions always require approval regardless of confidence: financial transactions, data deletion, external communications.

**Error accumulation:** If multiple tools fail or the agent has to replan more than twice, something is wrong. Better to get human input.

**Out-of-scope detection:** If the request does not match any known intent patterns or the agent cannot find relevant tools, it should admit this rather than hallucinate.

The escalation message matters: 'I want to make sure I give you accurate information. Let me connect you with a specialist.' This maintains trust rather than just failing.

I also track escalation rate as a metric. If it is too high, the agent needs improvement. If too low, maybe the threshold is wrong."

---

## References

- LangGraph Error Handling: https://langchain-ai.github.io/langgraph/
- Anthropic Agent Safety: https://docs.anthropic.com/claude/docs/tool-use-best-practices

---

*Next: [Memory and State](../08-memory-and-state/01-memory-management.md)*

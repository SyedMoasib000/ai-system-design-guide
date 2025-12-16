# Tool Design

Tools are the interface between agents and the external world. Well-designed tools make agents more capable and reliable. This chapter covers tool design principles, patterns, and best practices.

## Table of Contents

- [Tool Design Principles](#tool-design-principles)
- [Tool Definition Patterns](#tool-definition-patterns)
- [Common Tool Categories](#common-tool-categories)
- [Error Handling](#error-handling)
- [Tool Composition](#tool-composition)
- [Security Considerations](#security-considerations)
- [Testing Tools](#testing-tools)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Tool Design Principles

### Principle 1: Clear and Specific Descriptions

The LLM selects tools based on descriptions. Clarity is essential.

```python
# Bad: Vague description
@tool
def search(query: str) -> str:
    """Search for things."""
    pass

# Good: Specific description
@tool
def search_product_catalog(query: str) -> str:
    """Search the product catalog for items matching the query.
    
    Use this tool when the user asks about:
    - Product availability
    - Product specifications
    - Product pricing
    
    Args:
        query: Natural language search query (e.g., "red running shoes size 10")
    
    Returns:
        JSON list of matching products with name, price, and availability
    """
    pass
```

### Principle 2: Atomic Operations

Each tool should do one thing well.

```python
# Bad: Tool does too much
@tool
def manage_order(action: str, order_id: str, data: dict) -> str:
    """Create, update, or cancel an order."""
    if action == "create":
        # ...
    elif action == "update":
        # ...
    elif action == "cancel":
        # ...

# Good: Separate tools for each operation
@tool
def create_order(items: list[dict], customer_id: str) -> str:
    """Create a new order for a customer."""
    pass

@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel an existing order."""
    pass

@tool
def update_order_address(order_id: str, new_address: dict) -> str:
    """Update the shipping address for an order."""
    pass
```

### Principle 3: Informative Return Values

Return enough context for the agent to understand what happened.

```python
# Bad: Minimal return
@tool
def send_email(to: str, subject: str, body: str) -> str:
    email_api.send(to, subject, body)
    return "sent"

# Good: Informative return
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    try:
        result = email_api.send(to, subject, body)
        return json.dumps({
            "status": "sent",
            "message_id": result.id,
            "recipient": to,
            "timestamp": result.sent_at.isoformat()
        })
    except EmailError as e:
        return json.dumps({
            "status": "failed",
            "error": str(e),
            "recipient": to
        })
```

### Principle 4: Predictable Behavior

Tools should behave consistently without surprises.

```python
# Bad: Different behavior based on hidden state
@tool
def get_user_data(user_id: str) -> str:
    if cache.has(user_id):
        return cache.get(user_id)  # Different format than DB
    return db.get(user_id)

# Good: Consistent behavior
@tool
def get_user_data(user_id: str) -> str:
    """Get user profile data.
    
    Always returns JSON with: name, email, created_at, subscription_tier
    Returns error if user not found.
    """
    user = db.get(user_id)
    if not user:
        return json.dumps({"error": "User not found", "user_id": user_id})
    return json.dumps(user.to_dict())
```

---

## Tool Definition Patterns

### Pattern 1: Function-Based Tools

Simple Python functions with decorators:

```python
from langchain.tools import tool

@tool
def calculate_shipping(
    weight_kg: float,
    destination_country: str,
    shipping_method: str = "standard"
) -> str:
    """Calculate shipping cost for a package.
    
    Args:
        weight_kg: Package weight in kilograms
        destination_country: ISO country code (e.g., "US", "GB")
        shipping_method: "standard", "express", or "overnight"
    
    Returns:
        JSON with cost, estimated_days, and carrier
    """
    rates = shipping_api.get_rates(weight_kg, destination_country, shipping_method)
    return json.dumps({
        "cost_usd": rates.cost,
        "estimated_days": rates.days,
        "carrier": rates.carrier
    })
```

### Pattern 2: OpenAI Function Calling Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Usage
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)
```

### Pattern 3: Class-Based Tools

For tools with state or complex logic:

```python
class DatabaseQueryTool:
    def __init__(self, connection_string: str, allowed_tables: list[str]):
        self.db = Database(connection_string)
        self.allowed_tables = allowed_tables
    
    @property
    def definition(self) -> dict:
        return {
            "name": "query_database",
            "description": f"Query the database. Allowed tables: {self.allowed_tables}",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query"
                    }
                },
                "required": ["sql"]
            }
        }
    
    def run(self, sql: str) -> str:
        # Validate query
        if not self.is_safe_query(sql):
            return json.dumps({"error": "Query not allowed"})
        
        # Execute
        results = self.db.execute(sql)
        return json.dumps({"rows": results[:100], "total_count": len(results)})
    
    def is_safe_query(self, sql: str) -> bool:
        # Check for SELECT only, allowed tables, etc.
        parsed = sqlparse.parse(sql)[0]
        return (
            parsed.get_type() == "SELECT" and
            all(t in self.allowed_tables for t in extract_tables(sql))
        )
```

---

## Common Tool Categories

### Information Retrieval Tools

```python
@tool
def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """Search internal knowledge base for relevant documents."""
    results = vector_db.search(embed(query), top_k=max_results)
    return json.dumps([{
        "title": r.title,
        "snippet": r.text[:500],
        "source": r.source,
        "relevance": r.score
    } for r in results])

@tool
def get_webpage_content(url: str) -> str:
    """Fetch and extract text content from a webpage."""
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return text[:10000]  # Limit length
```

### Data Manipulation Tools

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Supports: +, -, *, /, ^, sqrt, sin, cos, tan, log, exp
    Example: "sqrt(16) + 2^3" returns "12.0"
    """
    try:
        result = safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies using current exchange rates."""
    rate = exchange_api.get_rate(from_currency, to_currency)
    converted = amount * rate
    return json.dumps({
        "original": {"amount": amount, "currency": from_currency},
        "converted": {"amount": round(converted, 2), "currency": to_currency},
        "rate": rate,
        "timestamp": datetime.now().isoformat()
    })
```

### Action Tools

```python
@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str] = None
) -> str:
    """Create a calendar event.
    
    Args:
        title: Event title
        start_time: ISO format datetime (e.g., "2025-12-20T14:00:00")
        end_time: ISO format datetime
        attendees: Optional list of email addresses
    """
    event = calendar_api.create_event(
        title=title,
        start=parse_datetime(start_time),
        end=parse_datetime(end_time),
        attendees=attendees or []
    )
    return json.dumps({
        "status": "created",
        "event_id": event.id,
        "link": event.html_link
    })

@tool
def send_slack_message(channel: str, message: str) -> str:
    """Send a message to a Slack channel."""
    result = slack.post_message(channel=channel, text=message)
    return json.dumps({
        "status": "sent",
        "channel": channel,
        "timestamp": result.ts
    })
```

### Confirmation Tools

For high-stakes actions, require confirmation:

```python
@tool
def request_human_approval(action: str, details: str) -> str:
    """Request human approval before proceeding with a sensitive action.
    
    Use this when:
    - Deleting data
    - Sending external communications
    - Making purchases
    - Any irreversible action
    """
    approval_id = create_approval_request(action, details)
    return json.dumps({
        "status": "pending_approval",
        "approval_id": approval_id,
        "message": "Waiting for human approval. Do not proceed until approved."
    })
```

---

## Error Handling

### Graceful Error Responses

```python
@tool
def api_call_with_error_handling(endpoint: str, params: dict) -> str:
    """Call an external API with proper error handling."""
    try:
        response = requests.post(endpoint, json=params, timeout=30)
        response.raise_for_status()
        return json.dumps({
            "status": "success",
            "data": response.json()
        })
    except requests.Timeout:
        return json.dumps({
            "status": "error",
            "error_type": "timeout",
            "message": "Request timed out after 30 seconds",
            "suggestion": "Try again or use a simpler query"
        })
    except requests.HTTPError as e:
        return json.dumps({
            "status": "error",
            "error_type": "http_error",
            "status_code": e.response.status_code,
            "message": str(e),
            "suggestion": "Check parameters and try again"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error_type": "unknown",
            "message": str(e)
        })
```

### Retry Logic

```python
@tool
def resilient_api_call(query: str) -> str:
    """Call API with automatic retry on transient failures."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = api.call(query)
            return json.dumps({"status": "success", "data": result})
        except TransientError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return json.dumps({
                "status": "error",
                "message": f"Failed after {max_retries} attempts: {e}"
            })
        except PermanentError as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "suggestion": "This error cannot be retried"
            })
```

---

## Tool Composition

### Wrapper Tools

Create high-level tools from low-level ones:

```python
@tool
def research_topic(topic: str) -> str:
    """Comprehensive research on a topic using multiple sources."""
    results = {
        "web": search_web(topic),
        "academic": search_arxiv(topic),
        "news": search_news(topic)
    }
    
    # Synthesize
    summary = llm.generate(
        f"Summarize these research results about {topic}:\n{json.dumps(results)}"
    )
    
    return json.dumps({
        "summary": summary,
        "sources": results
    })
```

### Tool Pipelines

```python
class ToolPipeline:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
    
    def run(self, initial_input: str) -> str:
        current = initial_input
        results = []
        
        for tool in self.tools:
            current = tool.run(current)
            results.append({
                "tool": tool.name,
                "output": current
            })
        
        return json.dumps({
            "final_result": current,
            "pipeline": results
        })

# Usage
data_pipeline = ToolPipeline([
    fetch_data_tool,
    clean_data_tool,
    analyze_data_tool,
    format_report_tool
])
```

---

## Security Considerations

### Input Validation

```python
@tool
def query_user_data(user_id: str, fields: list[str]) -> str:
    """Query user data with field restrictions."""
    # Validate user_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        return json.dumps({"error": "Invalid user ID format"})
    
    # Whitelist allowed fields
    allowed_fields = {"name", "email", "created_at", "subscription"}
    requested = set(fields)
    
    if not requested.issubset(allowed_fields):
        invalid = requested - allowed_fields
        return json.dumps({
            "error": f"Fields not allowed: {invalid}",
            "allowed_fields": list(allowed_fields)
        })
    
    user = db.get_user(user_id, fields=list(requested))
    return json.dumps(user)
```

### Rate Limiting

```python
class RateLimitedTool:
    def __init__(self, tool: Tool, max_calls: int, period_seconds: int):
        self.tool = tool
        self.limiter = RateLimiter(max_calls, period_seconds)
    
    @property
    def name(self):
        return self.tool.name
    
    @property
    def description(self):
        return self.tool.description
    
    def run(self, **kwargs) -> str:
        if not self.limiter.allow():
            return json.dumps({
                "error": "rate_limited",
                "message": f"Too many calls. Max {self.limiter.max_calls} per {self.limiter.period}s",
                "retry_after": self.limiter.retry_after()
            })
        return self.tool.run(**kwargs)
```

### Sandboxing Dangerous Operations

```python
@tool
def execute_code(code: str, language: str = "python") -> str:
    """Execute code in a sandboxed environment.
    
    Limitations:
    - No network access
    - No file system access
    - 30 second timeout
    - 256MB memory limit
    """
    if language not in ["python", "javascript"]:
        return json.dumps({"error": "Unsupported language"})
    
    try:
        result = sandbox.execute(
            code=code,
            language=language,
            timeout=30,
            memory_mb=256,
            network=False,
            filesystem=False
        )
        return json.dumps({
            "status": "success",
            "output": result.stdout,
            "errors": result.stderr
        })
    except SandboxTimeout:
        return json.dumps({"error": "Execution timed out (30s limit)"})
    except SandboxMemoryError:
        return json.dumps({"error": "Memory limit exceeded (256MB)"})
```

---

## Testing Tools

### Unit Testing

```python
class TestSearchTool:
    def test_successful_search(self):
        result = search_product_catalog("red shoes")
        data = json.loads(result)
        
        assert "products" in data
        assert len(data["products"]) > 0
        assert all("name" in p for p in data["products"])
    
    def test_empty_search(self):
        result = search_product_catalog("xyznonexistent123")
        data = json.loads(result)
        
        assert data["products"] == []
        assert "message" in data
    
    def test_error_handling(self):
        with mock.patch("api.search", side_effect=APIError("Service down")):
            result = search_product_catalog("shoes")
            data = json.loads(result)
            
            assert data["status"] == "error"
            assert "Service down" in data["message"]
```

### Integration Testing with Agents

```python
def test_agent_uses_tool_correctly():
    agent = Agent(tools=[calculate, convert_currency])
    
    # Test that agent selects correct tool
    response = agent.run("What is 15% of 250?")
    
    # Verify tool was called
    assert agent.last_tool_call.name == "calculate"
    assert "37.5" in response
    
def test_agent_handles_tool_error():
    agent = Agent(tools=[failing_tool])
    
    response = agent.run("Use the failing tool")
    
    # Agent should handle error gracefully
    assert "error" in response.lower() or "unable" in response.lower()
```

---

## Interview Questions

### Q: How do you design tools for reliable agent use?

**Strong answer:**
I follow these principles:

**1. Clear descriptions:**
- Specific about when to use the tool
- Include examples in docstring
- List what the tool cannot do

**2. Atomic operations:**
- One tool, one purpose
- Easier for LLM to select correctly
- Simpler error handling

**3. Informative returns:**
- Always return structured data (JSON)
- Include status, data, and errors
- Provide suggestions on errors

**4. Robust error handling:**
- Never throw unhandled exceptions
- Return errors as data
- Include retry guidance

**5. Validation:**
- Validate all inputs
- Whitelist allowed values
- Sanitize before use

### Q: How do you handle security for agent tools?

**Strong answer:**
Multiple layers of protection:

**1. Input validation:**
- Strict type checking
- Format validation (regex, parsing)
- Whitelist allowed values

**2. Authorization:**
- Check user permissions before tool execution
- Scope tools to user context
- Audit all tool calls

**3. Rate limiting:**
- Per-user limits
- Per-tool limits
- Graceful degradation

**4. Sandboxing:**
- Code execution in isolated containers
- Network restrictions
- Resource limits (CPU, memory, time)

**5. Human-in-the-loop:**
- Require approval for high-risk actions
- Flag unusual patterns
- Audit trail for compliance

---

## References

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use
- LangChain Tools: https://python.langchain.com/docs/modules/tools/

---

*Previous: [Agent Architectures](01-agent-architectures.md) | Next: [Error Handling and Recovery](03-error-handling.md)*

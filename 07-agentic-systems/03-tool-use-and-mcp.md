# Tool Use and MCP (Dec 2025)

Tools are the "hands" of an agent. In late 2025, the industry has standardized on the **Model Context Protocol (MCP)**, which replaces fragmented custom tool definitions with a unified, local-first communication layer.

## Table of Contents

- [The Tool-Use Mechanism](#mechanism)
- [Model Context Protocol (MCP)](#mcp)
- [Defining High-Precision Tools](#precision)
- [MCP vs. OpenAI Function Calling](#mcp-vs-openai)
- [Streaming Tool Calls](#streaming)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Tool-Use Mechanism

Tool use occurs in a 3-step cycle:
1. **Schema Presentation**: The model is given a JSON schema of the tools.
2. **Intent & Extraction**: The model outputs a "Call" (e.g., `{"tool": "get_weather", "args": {"city": "Tokyo"}}`).
3. **Execution & Contextualization**: The system runs the function and feeds the result back into the prompt.

**2025 Nuance**: We no longer "hardcode" tool definitions into the system prompt. We use **Dynamic Manifests** that fetch only necessary tools based on the user's intent.

---

## Model Context Protocol (MCP)

Developed by Anthropic and adopted as an industry standard in 2025, MCP allows models to interact with data and tools regardless of where they live.

- **MCP Client**: The AI application (e.g., your agent code).
- **MCP Server**: A standalone process that exposes Tools (Functions), Resources (Data), and Prompts (Templates).
- **Communication**: Uses JSON-RPC over stdio or HTTP.

### Why MCP?
- **Security**: Tools run in their own process, not in the model logic.
- **Portability**: Write a "Postgres Tool" once, use it in Claude, GPT, or Llama.
- **Discoverability**: Standardized `list_tools` and `get_resource` commands.

---

## Defining High-Precision Tools

A 2025 "Production-Quality" tool must include:

1. **Strict Type Validation**: Use Pydantic or Zod to enforce schemas before the model even sees the call.
2. **Detailed Docstrings**: Describe *when NOT* to use the tool.
3. **Confidence Thresholds**: Require the model to output a `confidence` score for the tool call.

```python
# MCP Server Example (Conceptual)
@server.tool()
class ExecuteSQL(PydanticModel):
    """Executes a Read-Only SQL query. DO NOT use for DROP/DELETE."""
    query: str = Field(..., description="The SELECT query to run.")

    async def run(self):
        # Implementation here...
        pass
```

---

## MCP vs. OpenAI Function Calling

| Feature | OpenAI Native | MCP |
|---------|---------------|-----|
| **Coupling** | High (OpenAI specific) | Low (Agnostic) |
| **Transport** | JSON in API body | JSON-RPC (Local/Remote) |
| **Data Access**| No native data "Resource" | Native `Resources` support |
| **Best For** | Prototyping | Enterprise Orchestration |

---

## Streaming Tool Calls

Late 2025 models support **Partial Tool Speculation**.
Instead of waiting for the full JSON to generate, the system starts "Prefetching" tool results as soon as the tool name and critical IDs are visible in the stream. This reduces perceived latency by **400-800ms**.

---

## Interview Questions

### Q: How does MCP solve the "Too Many Tools" problem (Schema Overload)?

**Strong answer:**
In 2023, giving a model 50 tools would degrade performance because the prompt became too long. MCP solves this through **Dynamic Resource Discovery**. Instead of loading 50 tool schemas into the prompt, the agent sends a `list_resources` call to the MCP server. It then only "attaches" the specific tools relevant to the current `Resource` context. This keeps the prompt lean and the context window focused on reasoning rather than parsing unused schemas.

### Q: Why is it important to separate "Tool Logic" from the "Agent App" using MCP servers?

**Strong answer:**
Separation of concerns. If the tool logic (e.g., a Python scraper) lives in a separate MCP server, I can scale the scraping infrastructure independently of the LLM orchestrator. More importantly, it provides a **Security Sandbox**. If a model tries to perform an injection through a tool argument, it only affects the MCP server process, which can be containerized with zero network access to the core Agent state.

---

## References
- Anthropic. "The Model Context Protocol Specification" (2025)
- JSON-RPC 2.0 Specification.
- Pydantic v3.0 Documentation.

---

*Next: [Multi-Agent Orchestration](04-multi-agent-orchestration.md)*

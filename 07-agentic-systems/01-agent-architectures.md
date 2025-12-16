# Agent Architectures

Agents are LLM-powered systems that can take actions, use tools, and work toward goals autonomously. This chapter covers the core patterns, architectures, and design decisions for building production agents.

## Table of Contents

- [What Are Agents](#what-are-agents)
- [Agent Components](#agent-components)
- [Core Agent Patterns](#core-agent-patterns)
- [ReAct Pattern](#react-pattern)
- [Planning Patterns](#planning-patterns)
- [Multi-Agent Systems](#multi-agent-systems)
- [Production Considerations](#production-considerations)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## What Are Agents

An agent is an LLM that can:
1. **Reason** about what to do
2. **Act** by calling tools or APIs
3. **Observe** the results
4. **Iterate** until the task is complete

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent Loop                              │
│                                                                 │
│   User Goal ──► Think ──► Act ──► Observe ──► Think ──► ...   │
│                   │         │         │                         │
│                   ▼         ▼         ▼                         │
│               Reasoning   Tool     Results                      │
│                          Calls                                  │
│                                                                 │
│   ... ──► Done? ──► Yes ──► Return Result                      │
│             │                                                   │
│             No ──► Continue Loop                                │
└─────────────────────────────────────────────────────────────────┘
```

### Agents vs Chains

| Aspect | Chain | Agent |
|--------|-------|-------|
| Control flow | Fixed, predetermined | Dynamic, LLM-decided |
| Tool usage | Fixed sequence | Chosen per step |
| Iteration | Usually single pass | Loops until done |
| Complexity | Lower | Higher |
| Reliability | More predictable | Less predictable |
| Use case | Well-defined workflows | Open-ended tasks |

---

## Agent Components

### Component 1: LLM (The Brain)

The LLM decides what actions to take:

```python
class Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM(model)
        self.tools = []
        self.memory = []
    
    def think(self, goal: str, observations: list) -> Action:
        prompt = self.build_prompt(goal, observations)
        response = self.llm.generate(prompt)
        return self.parse_action(response)
```

**Model requirements for agents:**
- Strong instruction following
- Reliable function calling
- Good reasoning capabilities
- Consistent output format

### Component 2: Tools

Tools are functions the agent can call:

```python
from typing import Callable

@tool
def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
    
    Returns:
        Search results as text
    """
    return web_search_api(query)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
    
    Returns:
        Confirmation message
    """
    return email_api.send(to, subject, body)
```

**Tool design principles:**
- Clear, specific descriptions
- Well-defined parameters
- Informative return values
- Error handling built in

### Component 3: Memory

Memory allows agents to maintain context:

```python
class AgentMemory:
    def __init__(self):
        self.short_term = []  # Current conversation
        self.long_term = VectorStore()  # Historical context
        self.working = {}  # Current task state
    
    def add_observation(self, observation: str):
        self.short_term.append({
            "type": "observation",
            "content": observation,
            "timestamp": now()
        })
    
    def add_action(self, action: str, result: str):
        self.short_term.append({
            "type": "action",
            "action": action,
            "result": result,
            "timestamp": now()
        })
    
    def get_context(self, max_tokens: int = 4000) -> str:
        # Return recent history that fits in context
        return self.format_history(self.short_term[-20:])
```

### Component 4: Orchestrator

The orchestrator manages the agent loop:

```python
class AgentOrchestrator:
    def __init__(self, agent: Agent, max_iterations: int = 10):
        self.agent = agent
        self.max_iterations = max_iterations
    
    def run(self, goal: str) -> str:
        observations = []
        
        for i in range(self.max_iterations):
            # Think: Decide next action
            action = self.agent.think(goal, observations)
            
            # Check if done
            if action.type == "finish":
                return action.result
            
            # Act: Execute tool
            try:
                result = self.execute_tool(action)
                observations.append(f"Action: {action}\nResult: {result}")
            except Exception as e:
                observations.append(f"Action: {action}\nError: {e}")
        
        return "Max iterations reached without completion"
```

---

## Core Agent Patterns

### Pattern 1: Simple Tool Agent

Single LLM call with tool selection:

```python
def simple_agent(query: str, tools: list) -> str:
    response = llm.generate(
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"
    )
    
    if response.tool_calls:
        # Execute tool and return result
        tool_call = response.tool_calls[0]
        result = execute_tool(tool_call)
        return result
    else:
        return response.content
```

**Use when:**
- Simple, single-step tasks
- Tool selection is straightforward
- No iteration needed

### Pattern 2: Iterative Agent

Multiple rounds of think-act-observe:

```python
def iterative_agent(goal: str, tools: list, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": goal}
    ]
    
    for step in range(max_steps):
        response = llm.generate(messages=messages, tools=tools)
        
        if not response.tool_calls:
            # Agent decided to respond
            return response.content
        
        # Execute tools and add results
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "Could not complete task in allowed steps"
```

**Use when:**
- Multi-step tasks
- Need to gather information iteratively
- Results of one tool inform next action

### Pattern 3: Hierarchical Agent

Manager agent delegates to specialized agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Manager Agent                              │
│                                                                 │
│   Analyzes task ──► Delegates to specialists ──► Synthesizes   │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Research   │    │   Writer    │    │   Code      │
│   Agent     │    │   Agent     │    │   Agent     │
└─────────────┘    └─────────────┘    └─────────────┘
```

```python
class ManagerAgent:
    def __init__(self):
        self.specialists = {
            "research": ResearchAgent(),
            "writing": WritingAgent(),
            "code": CodeAgent()
        }
    
    def run(self, task: str) -> str:
        # Analyze and decompose task
        plan = self.plan(task)
        
        results = []
        for step in plan.steps:
            specialist = self.specialists[step.agent_type]
            result = specialist.run(step.subtask)
            results.append(result)
        
        # Synthesize final result
        return self.synthesize(results)
```

---

## ReAct Pattern

ReAct (Reasoning + Acting) is the most common agent pattern.

### ReAct Prompt Structure

```
You are an agent that can use tools to accomplish tasks.

Available tools:
{tool_descriptions}

For each step, output your reasoning and action in this format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]

After receiving an observation, continue with another Thought/Action 
or conclude with:

Thought: [Your reasoning]
Final Answer: [Your final response]
```

### ReAct Implementation

```python
class ReActAgent:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.tool_descriptions = self.format_tools(tools)
    
    def run(self, query: str) -> str:
        prompt = self.build_prompt(query)
        scratchpad = ""
        
        for _ in range(10):  # Max iterations
            response = self.llm.generate(prompt + scratchpad)
            
            # Parse response
            if "Final Answer:" in response:
                return self.extract_final_answer(response)
            
            # Extract action
            action = self.parse_action(response)
            
            # Execute tool
            tool = self.tools.get(action.name)
            if not tool:
                observation = f"Error: Unknown tool {action.name}"
            else:
                try:
                    observation = tool.run(action.input)
                except Exception as e:
                    observation = f"Error: {e}"
            
            # Add to scratchpad
            scratchpad += f"\n{response}\nObservation: {observation}\n"
        
        return "Could not complete task"
    
    def parse_action(self, text: str) -> Action:
        action_match = re.search(r"Action: (.+)", text)
        input_match = re.search(r"Action Input: (.+)", text)
        
        return Action(
            name=action_match.group(1).strip(),
            input=input_match.group(1).strip()
        )
```

### ReAct Example Trace

```
User: What is the weather in Tokyo and should I bring an umbrella?

Thought: I need to find the current weather in Tokyo to answer this question.
Action: get_weather
Action Input: {"city": "Tokyo"}

Observation: {"temperature": 22, "condition": "rainy", "humidity": 85}

Thought: The weather in Tokyo is rainy, so the user should bring an umbrella.
Final Answer: The weather in Tokyo is 22°C and rainy with 85% humidity. 
Yes, you should definitely bring an umbrella.
```

---

## Planning Patterns

### Plan-and-Execute

Create a plan upfront, then execute:

```python
class PlanAndExecuteAgent:
    def run(self, task: str) -> str:
        # Step 1: Create plan
        plan = self.planner.create_plan(task)
        
        # Step 2: Execute each step
        results = []
        for step in plan.steps:
            result = self.executor.execute(step, results)
            results.append(result)
            
            # Optional: Replan if needed
            if result.needs_replanning:
                plan = self.planner.replan(task, results)
        
        # Step 3: Synthesize
        return self.synthesize(results)
    
    def create_plan(self, task: str) -> Plan:
        prompt = f"""Create a step-by-step plan to accomplish: {task}

Output a numbered list of steps. Each step should be:
- Specific and actionable
- Dependent only on previous steps
- Clearly contributing to the goal

Plan:"""
        
        response = self.llm.generate(prompt)
        return self.parse_plan(response)
```

### Advantages of Planning

| Aspect | ReAct (No Planning) | Plan-and-Execute |
|--------|---------------------|------------------|
| Adaptability | High | Medium |
| Efficiency | Lower (trial and error) | Higher (directed) |
| Predictability | Lower | Higher |
| Complex tasks | May struggle | Better suited |
| Error recovery | Per-step | Can replan |

### LLM Compiler Pattern

Parallel execution of independent steps:

```python
class LLMCompiler:
    def run(self, task: str) -> str:
        # Generate DAG of tasks
        dag = self.plan_as_dag(task)
        
        # Execute in parallel where possible
        results = {}
        for level in dag.levels():
            # All tasks in a level can run in parallel
            level_results = asyncio.gather(*[
                self.execute(task, results)
                for task in level.tasks
            ])
            results.update(level_results)
        
        return self.synthesize(results)

# Example DAG:
#     ┌─► Task A ─┐
#     │          │
# Start         ├─► Task D ─► End
#     │          │
#     └─► Task B ─┘
#     │
#     └─► Task C ─────────────┘
```

---

## Multi-Agent Systems

### When to Use Multiple Agents

| Scenario | Single Agent | Multi-Agent |
|----------|--------------|-------------|
| Simple tasks | ✅ | Overkill |
| Complex, multi-domain | Struggles | ✅ |
| Need specialization | Limited | ✅ |
| Parallel work | No | ✅ |
| Clear handoffs | N/A | ✅ |

### Multi-Agent Patterns

**1. Supervisor Pattern:**
```
Supervisor ──► Agent A ──► Supervisor
          ──► Agent B ──► Supervisor
          ──► Agent C ──► Supervisor
```

**2. Pipeline Pattern:**
```
Agent A ──► Agent B ──► Agent C ──► Result
```

**3. Debate Pattern:**
```
Agent A ◄──► Agent B
    │           │
    └─── Judge ─┘
```

### Implementation Example

```python
class MultiAgentSystem:
    def __init__(self):
        self.researcher = Agent(
            name="Researcher",
            tools=[search_web, search_arxiv],
            system_prompt="You are a research specialist..."
        )
        self.analyst = Agent(
            name="Analyst", 
            tools=[calculate, visualize],
            system_prompt="You are a data analyst..."
        )
        self.writer = Agent(
            name="Writer",
            tools=[format_document],
            system_prompt="You are a technical writer..."
        )
        self.supervisor = SupervisorAgent()
    
    def run(self, task: str) -> str:
        # Supervisor decides workflow
        workflow = self.supervisor.plan(task)
        
        context = {}
        for step in workflow:
            agent = getattr(self, step.agent)
            result = agent.run(step.task, context)
            context[step.agent] = result
        
        return context.get("writer", context)
```

---

## Production Considerations

### Reliability Patterns

```python
class ProductionAgent:
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
    
    def run_with_safety(self, task: str) -> AgentResult:
        for attempt in range(self.max_retries):
            try:
                result = timeout(self.timeout)(self.run)(task)
                
                # Validate result
                if self.validate_result(result):
                    return AgentResult(success=True, result=result)
                    
            except TimeoutError:
                logger.warning(f"Attempt {attempt + 1} timed out")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
        
        return AgentResult(success=False, error="Max retries exceeded")
```

### Cost Control

```python
class CostAwareAgent:
    def __init__(self, budget_per_task: float = 1.0):
        self.budget = budget_per_task
        self.spent = 0
    
    def run(self, task: str) -> str:
        while self.spent < self.budget:
            # Check budget before each LLM call
            estimated_cost = self.estimate_call_cost()
            if self.spent + estimated_cost > self.budget:
                return self.fallback_response()
            
            # Make LLM call
            response, cost = self.llm_call_with_cost_tracking()
            self.spent += cost
            
            if self.is_complete(response):
                return response
        
        return "Budget exceeded"
```

### Observability

```python
class ObservableAgent:
    def run(self, task: str) -> str:
        trace_id = generate_trace_id()
        
        with tracer.start_span("agent_run", trace_id=trace_id) as span:
            span.set_attribute("task", task)
            
            for step_num in range(self.max_steps):
                with tracer.start_span(f"step_{step_num}") as step_span:
                    # Think
                    thought = self.think(task)
                    step_span.set_attribute("thought", thought)
                    
                    # Act
                    action = self.parse_action(thought)
                    step_span.set_attribute("action", action.name)
                    
                    # Observe
                    result = self.execute(action)
                    step_span.set_attribute("result_length", len(result))
                    
                    if self.is_done(result):
                        return result
```

---

## Interview Questions

### Q: Explain the ReAct pattern and its limitations.

**Strong answer:**
ReAct combines Reasoning and Acting in an interleaved loop:

**Pattern:**
1. Thought: Model reasons about what to do
2. Action: Model selects and calls a tool
3. Observation: Tool result is added to context
4. Repeat until done

**Advantages:**
- Simple and interpretable
- Works with any LLM that can follow format
- Flexible tool selection

**Limitations:**
- Can get stuck in loops
- No upfront planning (trial and error)
- Context grows with each step (cost, context limits)
- May not be efficient for complex tasks
- Sensitive to prompt engineering

**When I would use alternatives:**
- Plan-and-execute for complex, multi-step tasks
- Router pattern for simple tool selection
- Multi-agent for specialized subtasks

### Q: How do you handle agent reliability in production?

**Strong answer:**
Multiple layers of reliability:

**1. Iteration limits:**
```python
max_iterations = 10
max_time = 60 seconds
```

**2. Output validation:**
- Parse and validate each action
- Check tool calls are well-formed
- Verify final answer format

**3. Error handling:**
- Catch tool execution errors
- Add error to context, let agent retry
- Fallback after N consecutive errors

**4. Cost controls:**
- Budget per task
- Track token usage
- Abort if budget exceeded

**5. Human escalation:**
- Confidence thresholds
- Automatic escalation for high-risk actions
- Human-in-the-loop for critical decisions

**6. Observability:**
- Log every step
- Trace end-to-end
- Alert on anomalies

### Q: When would you use multi-agent vs single agent?

**Strong answer:**
**Single agent when:**
- Task is well-defined
- Single domain of expertise needed
- Simple tool set
- Want simpler debugging

**Multi-agent when:**
- Task spans multiple domains
- Different expertise needed (research, code, writing)
- Can parallelize work
- Want modularity and reusability
- Complex tasks that benefit from specialization

**Tradeoffs:**
- Multi-agent adds complexity
- Harder to debug
- More expensive (multiple LLM calls)
- But: better separation of concerns, can optimize each agent

---

## References

- Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
- Wang et al. "Plan-and-Solve Prompting" (2023)
- AutoGPT: https://github.com/Significant-Gravitas/AutoGPT
- LangGraph: https://python.langchain.com/docs/langgraph
- CrewAI: https://github.com/joaomdmoura/crewAI

---

*Next: [Tool Design](02-tool-design.md)*

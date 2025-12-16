# LangChain Ecosystem

LangChain is the most widely adopted framework for building LLM applications. This chapter covers its core abstractions, patterns, and when to use or avoid it.

## Table of Contents

- [LangChain Overview](#langchain-overview)
- [Core Abstractions](#core-abstractions)
- [LangChain Expression Language](#langchain-expression-language)
- [LangGraph for Agents](#langgraph-for-agents)
- [LangSmith for Observability](#langsmith-for-observability)
- [When to Use LangChain](#when-to-use-langchain)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## LangChain Overview

### The Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                     LANGCHAIN ECOSYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   LangChain      │  │   LangGraph      │  │   LangSmith    │ │
│  │   (Core)         │  │   (Agents)       │  │   (Observability)│
│  │                  │  │                  │  │                │ │
│  │ • Prompts        │  │ • State graphs   │  │ • Tracing      │ │
│  │ • Chains         │  │ • Cycles         │  │ • Evaluation   │ │
│  │ • Retrievers     │  │ • Checkpointing  │  │ • Datasets     │ │
│  │ • Tools          │  │ • Human-in-loop  │  │ • Monitoring   │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   LangChain Hub                           │   │
│  │            Prompt templates, chains, agents               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Use When |
|-----------|---------|----------|
| **langchain-core** | Base abstractions | Always (minimal dependency) |
| **langchain** | Chains, agents | Building pipelines |
| **langchain-community** | Third-party integrations | Using specific providers |
| **langgraph** | Stateful agents | Complex agent workflows |
| **langsmith** | Observability | Production monitoring |

---

## Core Abstractions

### Chat Models

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Unified interface across providers
openai_model = ChatOpenAI(model="gpt-4o", temperature=0)
anthropic_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Same API for both
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain RAG in one sentence.")
]

response = openai_model.invoke(messages)
# Or: response = anthropic_model.invoke(messages)
```

### Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Simple template
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{query}")
])

# With message history
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Invoke
formatted = simple_prompt.invoke({
    "role": "technical",
    "query": "What is a vector database?"
})
```

### Output Parsers

```python
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# Structured output with Pydantic
class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Source references")

parser = JsonOutputParser(pydantic_object=Answer)

# Include format instructions in prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions about documents. {format_instructions}"),
    ("human", "{question}")
]).partial(format_instructions=parser.get_format_instructions())
```

### Retrievers

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever

# Vector store retriever
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Custom retriever
class HybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    keyword_retriever: BaseRetriever
    
    def _get_relevant_documents(self, query: str) -> list:
        vector_docs = self.vector_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)
        return self.merge_results(vector_docs, keyword_docs)
```

---

## LangChain Expression Language

### LCEL Basics

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Chain with pipe operator
chain = prompt | model | parser

# Equivalent to:
# result = parser.invoke(model.invoke(prompt.invoke(input)))


# RAG chain with LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("What is the capital of France?")
```

### Branching and Routing

```python
from langchain_core.runnables import RunnableBranch

# Conditional routing
route = RunnableBranch(
    (lambda x: "code" in x["question"].lower(), code_chain),
    (lambda x: "math" in x["question"].lower(), math_chain),
    default_chain
)

# With classifier
from langchain_core.runnables import RunnableParallel

classifier_chain = prompt_classifier | model | StrOutputParser()

def route_by_classification(input):
    classification = classifier_chain.invoke(input)
    if classification == "technical":
        return technical_chain
    return general_chain

routed_chain = RunnableLambda(route_by_classification)
```

### Parallel Execution

```python
# Run multiple chains in parallel
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

result = parallel_chain.invoke({"text": document})
# result = {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

### Streaming

```python
# Stream responses
async for chunk in chain.astream({"question": "Explain transformers"}):
    print(chunk, end="", flush=True)

# Stream events (for debugging)
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## LangGraph for Agents

### Why LangGraph

| LangChain Agents | LangGraph |
|------------------|-----------|
| Linear execution | Graph-based execution |
| Limited cycles | Supports cycles |
| Implicit state | Explicit state |
| Hard to debug | Visualizable |

### Basic Graph Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

# Define nodes
def agent_node(state: AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    return {"messages": [response], "next_step": "decide"}

def tool_node(state: AgentState) -> AgentState:
    # Execute tool based on last message
    tool_call = state["messages"][-1].tool_calls[0]
    result = execute_tool(tool_call)
    return {"messages": [result], "next_step": "agent"}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda state: "tools" if has_tool_calls(state) else END,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")

# Compile
app = graph.compile()
```

### Checkpointing and Memory

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add persistence
memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

# Invoke with thread ID for conversation continuity
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [HumanMessage("Hello")]}, config)

# Later, continue same conversation
result = app.invoke({"messages": [HumanMessage("What did I say?")]}, config)
```

### Human-in-the-Loop

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Define interrupt points
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("human_review", human_review_node)

# Interrupt before tool execution
app = graph.compile(interrupt_before=["tools"])

# Run until interrupt
result = app.invoke(input, config)

# Review and approve
if user_approves():
    result = app.invoke(None, config)  # Continue
else:
    result = app.invoke({"messages": [SystemMessage("User rejected")]}, config)
```

---

## LangSmith for Observability

### Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain calls are automatically traced
response = chain.invoke(input)

# Manual tracing with context
from langsmith import traceable

@traceable(name="my_custom_function")
def process_query(query: str) -> str:
    # Your logic here
    return result
```

### Evaluation

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create dataset
dataset = client.create_dataset("qa-test")
client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "Retrieval Augmented Generation..."}],
    dataset_id=dataset.id
)

# Define evaluators
def correctness(run, example):
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    # Return score 0-1
    return {"score": similarity(prediction, reference)}

# Run evaluation
results = evaluate(
    chain.invoke,
    data=dataset,
    evaluators=[correctness],
    experiment_prefix="v1"
)
```

---

## When to Use LangChain

### Use LangChain When

| Scenario | Why |
|----------|-----|
| Rapid prototyping | Quick iteration with many integrations |
| Multi-provider support | Unified API across OpenAI, Anthropic, etc. |
| Standard patterns | RAG, agents, chains are built-in |
| Team familiarity | Common abstractions, easier onboarding |
| Observability needed | LangSmith integration |

### Avoid LangChain When

| Scenario | Why | Alternative |
|----------|-----|-------------|
| Simple use case | Overhead not worth it | Direct API calls |
| Maximum performance | Abstraction overhead | Custom implementation |
| Full control needed | Framework constraints | Build from primitives |
| Minimal dependencies | Large dependency tree | Lightweight alternatives |

### Minimal LangChain

```python
# Use only langchain-core for minimal footprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Simple chain without heavy dependencies
chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)
```

---

## Interview Questions

### Q: When would you use LangChain vs build from scratch?

**Strong answer:**

"I use LangChain when:
- Prototyping quickly and need many integrations
- Team already knows the framework
- Need observability (LangSmith)
- Using standard patterns (RAG, agents)

I build from scratch when:
- Performance is critical (avoid abstraction overhead)
- Use case is very simple (direct API calls suffice)
- Need full control over behavior
- Minimizing dependencies is important

In practice, I often start with LangChain for prototyping, then evaluate whether the abstractions help or hinder as the project matures. For production, I might keep LangChain for non-critical paths and use direct calls for hot paths."

### Q: How does LangGraph differ from LangChain agents?

**Strong answer:**

"LangGraph provides graph-based agent orchestration with explicit state management.

**Key differences:**

1. **Cycles:** LangGraph supports cycles (agent loops), LangChain agents are more linear.

2. **State:** LangGraph has explicit typed state passed between nodes. LangChain agents have implicit state.

3. **Checkpointing:** LangGraph has built-in persistence for interrupts and recovery.

4. **Visualization:** LangGraph graphs can be visualized to understand flow.

5. **Human-in-the-loop:** LangGraph has native support for interrupts and approval flows.

I use LangGraph when building complex agents that need loops, memory, or human oversight. For simple tool-calling agents, LangChain's basic agent executor is sufficient."

---

## References

- LangChain Docs: https://python.langchain.com/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangSmith: https://docs.smith.langchain.com/

---

*Next: [LlamaIndex](03-llamaindex.md)*

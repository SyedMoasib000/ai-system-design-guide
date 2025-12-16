# AI Frameworks Overview

A practical guide to the major frameworks for building LLM applications. This chapter covers when to use each framework and their tradeoffs.

## Table of Contents

- [Framework Landscape](#framework-landscape)
- [LangChain](#langchain)
- [LlamaIndex](#llamaindex)
- [Semantic Kernel](#semantic-kernel)
- [Haystack](#haystack)
- [DSPy](#dspy)
- [Build vs Buy Decision](#build-vs-buy-decision)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Framework Landscape

### Framework Categories

| Category | Frameworks | Use Case |
|----------|------------|----------|
| General purpose | LangChain, Semantic Kernel | Broad LLM applications |
| RAG-focused | LlamaIndex, Haystack | Document retrieval and QA |
| Agent-focused | LangGraph, CrewAI, AutoGen | Agentic workflows |
| Optimization | DSPy | Prompt optimization |
| Serving | vLLM, TGI, Ollama | Model inference |

### Selection Guide

```
Start Here
    │
    ├── Need RAG primarily? ──────────► LlamaIndex
    │
    ├── Need agents/workflows? ───────► LangGraph
    │
    ├── Need Microsoft ecosystem? ────► Semantic Kernel
    │
    ├── Need maximum control? ────────► Build custom
    │
    └── Need flexibility? ────────────► LangChain
```

---

## LangChain

### Overview

Most popular general-purpose LLM framework. Provides abstractions for prompts, chains, agents, and integrations.

### Core Concepts

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model
llm = ChatOpenAI(model="gpt-4o")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Chain (LCEL - LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# Execute
response = chain.invoke({"input": "Hello!"})
```

### When to Use LangChain

**Good for:**
- Rapid prototyping
- Integration with many providers
- Standard RAG patterns
- Teams new to LLM development

**Challenges:**
- Abstraction overhead
- Debugging can be complex
- Frequent breaking changes historically
- Performance overhead for simple use cases

### LangGraph for Agents

```python
from langgraph.graph import StateGraph, END

# Define state
class AgentState(TypedDict):
    messages: list
    next_action: str

# Define nodes
def agent_node(state):
    # LLM decides next action
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def tool_node(state):
    # Execute tool
    result = execute_tool(state)
    return {"messages": state["messages"] + [result]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges("tools", should_continue)

app = workflow.compile()
```

---

## LlamaIndex

### Overview

Focused on connecting LLMs with data. Excellent for RAG applications.

### Core Concepts

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

### Advanced RAG with LlamaIndex

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Custom chunking
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# Build index
index = VectorStoreIndex(nodes)

# Custom retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# With postprocessing (reranking)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# Query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor]
)
```

### When to Use LlamaIndex

**Good for:**
- RAG applications
- Document ingestion and indexing
- Structured data querying
- Knowledge graph integration

**Challenges:**
- Less flexible for non-RAG use cases
- Abstractions can hide complexity
- Learning curve for advanced patterns

---

## Semantic Kernel

### Overview

Microsoft's framework for AI orchestration. Strong C# support and Azure integration.

### Core Concepts

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Create kernel
kernel = sk.Kernel()

# Add AI service
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://your-endpoint.openai.azure.com",
        api_key="your-key"
    )
)

# Define a function
@kernel.function
def summarize(text: str) -> str:
    """Summarize the given text."""
    return f"Summarize this: {text}"

# Execute
result = await kernel.invoke(summarize, text="Long document...")
```

### When to Use Semantic Kernel

**Good for:**
- Microsoft/Azure ecosystem
- C# or .NET applications
- Enterprise environments
- Teams familiar with Microsoft patterns

**Challenges:**
- Smaller community than LangChain
- Fewer integrations
- Python support less mature than C#

---

## Haystack

### Overview

Production-ready framework focused on search and QA pipelines.

### Core Concepts

```python
from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Create document store
document_store = InMemoryDocumentStore()

# Build indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
indexing_pipeline.add_component("writer", DocumentWriter(document_store))
indexing_pipeline.connect("embedder", "writer")

# Run pipeline
indexing_pipeline.run({"embedder": {"documents": documents}})
```

### When to Use Haystack

**Good for:**
- Production search systems
- Pipeline-oriented architecture
- Teams wanting explicit control
- Hybrid search implementations

---

## DSPy

### Overview

Programmatic approach to optimizing LLM pipelines. Treats prompts as learnable parameters.

### Core Concepts

```python
import dspy

# Configure LM
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)

# Define a module
class RAG(dspy.Module):
    def __init__(self, num_docs=3):
        self.retrieve = dspy.Retrieve(k=num_docs)
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Use the module
rag = RAG()
response = rag("What is the capital of France?")

# Optimize with examples
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=your_metric)
optimized_rag = optimizer.compile(rag, trainset=training_examples)
```

### When to Use DSPy

**Good for:**
- Prompt optimization at scale
- Research and experimentation
- Teams with ML expertise
- When prompt engineering hits limits

**Challenges:**
- Steeper learning curve
- Less intuitive than template-based approaches
- Newer, smaller community

---

## Build vs Buy Decision

### When to Build Custom

| Signal | Build Custom |
|--------|--------------|
| Simple use case | Single LLM call, basic RAG |
| Performance critical | Remove framework overhead |
| Deep customization | Unusual patterns |
| Team expertise | Strong engineering team |
| Long-term ownership | Full control needed |

### When to Use a Framework

| Signal | Use Framework |
|--------|---------------|
| Rapid prototyping | Get to MVP fast |
| Standard patterns | Common RAG/agent patterns |
| Many integrations | Multiple providers/tools |
| Team learning | New to LLM development |
| Community support | Benefit from ecosystem |

### Thin Wrapper Approach

Many production systems use a thin wrapper:

```python
class LLMClient:
    """Thin wrapper over provider APIs."""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.client = self._init_client(provider)
    
    async def generate(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs
    ) -> dict:
        response = await self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            **kwargs
        )
        return self._parse_tool_response(response)
```

**Benefits:**
- Full control
- No framework overhead
- Easy to understand and debug
- Switch providers easily

---

## Interview Questions

### Q: When would you use LangChain vs build custom?

**Strong answer:**

"My decision depends on the use case complexity and timeline.

**Use LangChain when:**
- Rapid prototyping is the priority
- The use case is standard (basic RAG, simple chains)
- Team is new to LLM development
- Need many integrations quickly

**Build custom when:**
- Use case is simple (direct API calls)
- Performance is critical
- Need deep customization
- Have strong engineering team
- Long-term maintenance matters

**My practical approach:**

For production systems, I often start with a thin wrapper over provider APIs. This gives me full control and easy debugging. I only adopt a framework if it provides significant value beyond what I can build in a few days.

The frameworks add abstraction overhead. For a simple RAG pipeline, the direct API approach might be 50 lines of code. With LangChain, you get more features but also more complexity to debug.

That said, LangGraph is valuable for complex agents because state management and checkpointing are genuinely hard problems. I would not rebuild that from scratch."

### Q: How do you compare LlamaIndex and LangChain for RAG?

**Strong answer:**

"Both can build RAG systems, but they have different philosophies.

**LlamaIndex** is data-centric. It shines at ingestion, indexing, and retrieval. The abstractions are built around documents and indexes. If my primary use case is connecting LLMs to data, LlamaIndex is often simpler.

**LangChain** is more general-purpose. It provides building blocks for many patterns: chains, agents, tools, memory. RAG is one of many things it does.

**My selection criteria:**

Choose LlamaIndex when:
- Pure RAG/QA use case
- Complex document processing needed
- Want opinionated defaults for retrieval

Choose LangChain when:
- RAG is part of a larger system
- Need agents and tools
- Want flexibility to compose patterns

In practice, I have used both together. LlamaIndex for document processing and indexing, LangChain for the agent orchestration that uses the LlamaIndex retriever."

---

## References

- LangChain: https://python.langchain.com/
- LlamaIndex: https://docs.llamaindex.ai/
- Semantic Kernel: https://learn.microsoft.com/semantic-kernel/
- Haystack: https://haystack.deepset.ai/
- DSPy: https://dspy-docs.vercel.app/

---

*Next: [Agent Frameworks Comparison](02-agent-frameworks.md)*

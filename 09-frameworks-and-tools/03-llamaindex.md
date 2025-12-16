# LlamaIndex

LlamaIndex is a data framework for LLM applications, specializing in connecting LLMs to data sources. It excels at document processing and RAG implementations.

## Table of Contents

- [LlamaIndex Overview](#llamaindex-overview)
- [Core Concepts](#core-concepts)
- [Document Processing](#document-processing)
- [Indexing Strategies](#indexing-strategies)
- [Query Engines](#query-engines)
- [Agents in LlamaIndex](#agents-in-llamaindex)
- [LangChain vs LlamaIndex](#langchain-vs-llamaindex)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## LlamaIndex Overview

### Design Philosophy

| LangChain | LlamaIndex |
|-----------|------------|
| General-purpose LLM framework | Data-focused LLM framework |
| Chain composition | Index and query abstraction |
| Broad integrations | Deep document processing |
| Agent-first | Data-first |

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLAMAINDEX ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Documents  │────▶│    Nodes     │────▶│    Index     │    │
│  │   (Source)   │     │   (Chunks)   │     │  (VectorDB)  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                     │           │
│                                                     ▼           │
│                              ┌──────────────────────────────┐   │
│                              │       Query Engine           │   │
│                              │  ┌────────┐  ┌────────────┐  │   │
│                              │  │Retriever│  │ Synthesizer│  │   │
│                              │  └────────┘  └────────────┘  │   │
│                              └──────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Documents and Nodes

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# Create document
doc = Document(
    text="LlamaIndex is a data framework for LLM applications...",
    metadata={"source": "docs.txt", "author": "team"}
)

# Parse into nodes (chunks)
parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents([doc])
# Each node has: text, metadata, relationships (prev/next)
```

### Index Types

| Index Type | Use Case | Tradeoffs |
|------------|----------|-----------|
| **VectorStoreIndex** | Semantic search | Standard RAG |
| **SummaryIndex** | Summarization | Processes all nodes |
| **KeywordTableIndex** | Keyword search | Fast, less semantic |
| **TreeIndex** | Hierarchical | Good for long docs |
| **KnowledgeGraphIndex** | Relationships | Complex setup |

```python
from llama_index.core import VectorStoreIndex, SummaryIndex

# Vector index for semantic search
vector_index = VectorStoreIndex.from_documents(documents)

# Summary index for summarization
summary_index = SummaryIndex.from_documents(documents)
```

### Settings (Global Configuration)

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure globally
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

---

## Document Processing

### Loading Documents

```python
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.database import DatabaseReader

# PDF
pdf_reader = PDFReader()
pdf_docs = pdf_reader.load_data("document.pdf")

# Web pages
web_reader = SimpleWebPageReader()
web_docs = web_reader.load_data(["https://example.com"])

# Database
db_reader = DatabaseReader(
    sql_database=database,
    table_name="articles"
)
db_docs = db_reader.load_data(query="SELECT content FROM articles")
```

### Advanced Parsing

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)

# Semantic chunking (based on embedding similarity)
semantic_parser = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=95
)

# Hierarchical (parent-child relationships)
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Large, medium, small
)

nodes = hierarchical_parser.get_nodes_from_documents(documents)
```

### Metadata Extraction

```python
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline

# Build extraction pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512),
        TitleExtractor(),
        SummaryExtractor(),
        KeywordExtractor(keywords=5)
    ]
)

nodes = pipeline.run(documents=documents)
# Each node now has: title, summary, keywords in metadata
```

---

## Indexing Strategies

### Vector Store Integration

```python
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import pinecone

# Connect to Pinecone
pinecone.init(api_key="...", environment="...")
pinecone_index = pinecone.Index("my-index")

# Create LlamaIndex vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

### Hybrid Search

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Vector retriever
vector_retriever = index.as_retriever(similarity_top_k=5)

# BM25 retriever
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5
)

# Combine with fusion
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,  # Generate 1 query variation
    mode="reciprocal_rerank"
)
```

---

## Query Engines

### Basic Query Engine

```python
# Create query engine from index
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"  # or "tree_summarize", "refine"
)

response = query_engine.query("What is LlamaIndex?")
print(response.response)
print(response.source_nodes)  # Retrieved chunks
```

### Response Modes

| Mode | Description | Use When |
|------|-------------|----------|
| `compact` | Single LLM call | Default, balanced |
| `refine` | Iterative refinement | Long context |
| `tree_summarize` | Hierarchical | Summarization |
| `simple_summarize` | Direct concat | Simple cases |
| `no_text` | Retrieval only | Just need sources |

### Custom Query Engine

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer

class RAGQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    synthesizer: BaseSynthesizer
    
    def custom_query(self, query_str: str) -> str:
        # Retrieve
        nodes = self.retriever.retrieve(query_str)
        
        # Rerank (custom logic)
        reranked = self.rerank(nodes, query_str)
        
        # Synthesize
        response = self.synthesizer.synthesize(query_str, reranked)
        
        return response
```

### Sub-Question Query Engine

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Create tools from different indices
tools = [
    QueryEngineTool.from_defaults(
        query_engine=finance_index.as_query_engine(),
        name="finance",
        description="Financial documents"
    ),
    QueryEngineTool.from_defaults(
        query_engine=legal_index.as_query_engine(),
        name="legal",
        description="Legal documents"
    )
]

# Engine that decomposes complex queries
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools
)

# Complex query gets broken into sub-questions
response = query_engine.query(
    "Compare the financial implications of the merger with its legal risks"
)
```

---

## Agents in LlamaIndex

### Basic Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Define tools
def search_docs(query: str) -> str:
    """Search internal documents."""
    return query_engine.query(query).response

def calculate(expression: str) -> str:
    """Evaluate mathematical expression."""
    return str(eval(expression))

tools = [
    FunctionTool.from_defaults(fn=search_docs),
    FunctionTool.from_defaults(fn=calculate)
]

# Create agent
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True
)

response = agent.chat("What was our Q3 revenue and what's 20% of it?")
```

### Query Engine as Tool

```python
from llama_index.core.tools import QueryEngineTool

# Wrap query engines as tools
sales_tool = QueryEngineTool.from_defaults(
    query_engine=sales_index.as_query_engine(),
    name="sales_data",
    description="Query sales figures and forecasts"
)

hr_tool = QueryEngineTool.from_defaults(
    query_engine=hr_index.as_query_engine(),
    name="hr_policies",
    description="Query HR policies and procedures"
)

# Agent with query engine tools
agent = ReActAgent.from_tools([sales_tool, hr_tool], llm=llm)
```

---

## LangChain vs LlamaIndex

### Feature Comparison

| Feature | LangChain | LlamaIndex |
|---------|-----------|------------|
| Document loaders | Many | Many |
| Chunking strategies | Basic | Advanced (semantic, hierarchical) |
| Index types | Vector only | Multiple (tree, graph, keyword) |
| Query optimization | Manual | Built-in (sub-questions, routing) |
| Agent framework | Extensive | Good |
| Chain composition | LCEL | Pipelines |
| Community size | Larger | Smaller |

### When to Choose

**Choose LlamaIndex when:**
- Document processing is core to your application
- Need advanced chunking (semantic, hierarchical)
- Complex query patterns (sub-questions, routing)
- Building primarily RAG applications

**Choose LangChain when:**
- Need general LLM orchestration
- Building diverse agent types
- Want extensive integrations
- Team already knows it

**Use Both:**
```python
# LlamaIndex for indexing, LangChain for orchestration
from llama_index.core import VectorStoreIndex
from langchain.chains import LLMChain

# Index with LlamaIndex
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# Use in LangChain
from llama_index.core.langchain_helpers.agents import IndexToolConfig
tool = IndexToolConfig.from_query_engine(query_engine)
```

---

## Interview Questions

### Q: How does LlamaIndex differ from LangChain?

**Strong answer:**

"LlamaIndex is data-centric while LangChain is orchestration-centric.

**LlamaIndex strengths:**
- Advanced document processing (semantic chunking, hierarchical parsing)
- Multiple index types (vector, tree, graph, keyword)
- Built-in query optimization (sub-questions, routing)
- Deep focus on RAG patterns

**LangChain strengths:**
- General-purpose LLM orchestration
- Extensive agent framework
- LCEL for chain composition
- Larger ecosystem

I use LlamaIndex when document understanding is central to the application. I use LangChain for general orchestration. They can work together: LlamaIndex for indexing and retrieval, LangChain for overall workflow."

### Q: Explain hierarchical indexing in LlamaIndex.

**Strong answer:**

"Hierarchical indexing creates parent-child relationships between chunks at different granularities.

**How it works:**
1. Parse document into large chunks (e.g., 2048 tokens)
2. Split those into medium chunks (512 tokens)
3. Split further into small chunks (128 tokens)
4. Index small chunks for retrieval
5. On retrieval, can access parent chunks for more context

**Benefits:**
- Retrieve precise matches (small chunks)
- Get surrounding context (parent chunks)
- Better for long documents

**Implementation:**
```python
parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
```

This is valuable when you need both precision (exact answer location) and context (surrounding paragraphs for understanding)."

---

## References

- LlamaIndex Docs: https://docs.llamaindex.ai/
- LlamaHub (Data Loaders): https://llamahub.ai/

---

*Next: [DSPy](04-dspy.md)*

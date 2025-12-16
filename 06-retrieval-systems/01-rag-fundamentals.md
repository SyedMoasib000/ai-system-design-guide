# RAG Fundamentals

Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models to produce grounded, accurate responses. This chapter covers the core concepts, architecture patterns, and design decisions for building production RAG systems.

## Table of Contents

- [What Is RAG](#what-is-rag)
- [Why RAG](#why-rag)
- [RAG Architecture](#rag-architecture)
- [RAG vs Fine-Tuning](#rag-vs-fine-tuning)
- [The RAG Pipeline](#the-rag-pipeline)
- [Quality Dimensions](#quality-dimensions)
- [Common Failure Modes](#common-failure-modes)
- [Naive RAG vs Advanced RAG](#naive-rag-vs-advanced-rag)
- [Production Considerations](#production-considerations)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## What Is RAG

RAG augments an LLM with external knowledge by retrieving relevant information and including it in the generation context.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG System                                 │
│                                                                     │
│   Query ──────► Retriever ──────► Relevant Documents                │
│                                          │                          │
│                                          ▼                          │
│   Query ──────────────────────► LLM + Context ──────► Response      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Core idea:** Instead of relying solely on knowledge encoded in model weights during training, retrieve up-to-date, specific information at query time.

---

## Why RAG

### Problems RAG Solves

| Problem | How RAG Helps |
|---------|---------------|
| Knowledge cutoff | Retrieve current information |
| Hallucination | Ground responses in retrieved facts |
| Domain specificity | Access proprietary/specialized data |
| Transparency | Cite sources for claims |
| Privacy | Keep data in your control |

### When to Use RAG

**Good fit for RAG:**
- Question answering over documents
- Customer support with knowledge base
- Enterprise search with synthesis
- Research assistants
- Any task requiring factual accuracy over specific corpus

**Not ideal for RAG:**
- Creative writing
- Tasks requiring reasoning without external knowledge
- Very latency-sensitive applications (retrieval adds time)
- When data is already in model's training set

---

## RAG Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE                                │
│                                                                     │
│   Documents ──► Chunking ──► Embedding ──► Vector Database          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE                                 │
│                                                                     │
│   Query ──► Embed ──► Retrieve ──► Rerank ──► Generate ──► Response │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Purpose | Options |
|-----------|---------|---------|
| Chunking | Split documents into retrievable units | Fixed size, semantic, recursive |
| Embedding | Convert text to vectors | OpenAI, Cohere, BGE, E5 |
| Vector DB | Store and search embeddings | Pinecone, Qdrant, Weaviate, Chroma |
| Retriever | Find relevant chunks | Dense, sparse, hybrid |
| Reranker | Re-score retrieved chunks | Cross-encoder, Cohere, LLM |
| Generator | Produce final response | GPT-4, Claude, Llama |

---

## RAG vs Fine-Tuning

This is a common interview question. Here is a decision framework:

### Comparison Matrix

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| **Data freshness** | Updates immediately | Requires retraining |
| **Data volume** | Any amount | Needs 1K-100K examples |
| **Latency** | Adds retrieval time (100-500ms) | No added latency |
| **Cost** | Per-query retrieval + tokens | Training cost + per-token |
| **Privacy** | Data stays in your control | Training data sent to provider |
| **Accuracy** | High for factual tasks | High for style/behavior |
| **Hallucination** | Reduced (grounded) | Still possible |
| **Explainability** | Can cite sources | Black box |
| **Maintenance** | Update documents | Retrain periodically |

### Decision Framework

**Choose RAG when:**
- Data updates frequently
- Need to cite sources
- Working with large document corpus
- Privacy requires data control
- Factual accuracy is critical

**Choose Fine-Tuning when:**
- Need specific style or behavior
- Task is well-defined with clear examples
- Latency is critical
- Data is static
- Changing how model responds, not what it knows

**Consider Both when:**
- Need both knowledge and behavior changes
- High-stakes applications requiring maximum accuracy
- Fine-tune on RAG-style prompts to improve retrieval usage

---

## The RAG Pipeline

### Step 1: Document Ingestion

```python
def ingest_document(document: Document) -> list[Chunk]:
    # 1. Extract text from various formats
    text = extract_text(document)  # PDF, DOCX, HTML, etc.
    
    # 2. Clean and preprocess
    text = clean_text(text)  # Remove noise, normalize
    
    # 3. Extract metadata
    metadata = extract_metadata(document)
    
    # 4. Chunk the document
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # 5. Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata = {
            **metadata,
            "chunk_index": chunk.index,
            "source": document.source
        }
    
    return chunks
```

### Step 2: Embedding and Indexing

```python
def index_chunks(chunks: list[Chunk], vector_db):
    # Batch embed for efficiency
    embeddings = embedding_model.embed_batch(
        [chunk.text for chunk in chunks]
    )
    
    # Upsert to vector database
    vector_db.upsert([
        {
            "id": chunk.id,
            "embedding": embedding,
            "metadata": chunk.metadata,
            "text": chunk.text
        }
        for chunk, embedding in zip(chunks, embeddings)
    ])
```

### Step 3: Query Processing

```python
def process_query(query: str, user_context: dict) -> str:
    # 1. Embed the query
    query_embedding = embedding_model.embed(query)
    
    # 2. Retrieve candidates
    candidates = vector_db.search(
        query_embedding,
        top_k=20,
        filter=build_filter(user_context)  # Permissions, etc.
    )
    
    # 3. Rerank
    reranked = reranker.rerank(query, candidates, top_k=5)
    
    # 4. Build context
    context = format_context(reranked)
    
    # 5. Generate response
    response = llm.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        context=context,
        query=query
    )
    
    return response
```

### Step 4: Response Generation

```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context does not contain the answer, say "I could not find information about that"
3. Cite sources using [1], [2] format
4. Be concise and accurate
"""

def generate_response(query: str, context: str) -> str:
    prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    
    return llm.generate(
        system=RAG_SYSTEM_PROMPT,
        user=prompt,
        temperature=0.1
    )
```

---

## Quality Dimensions

RAG quality has multiple independent dimensions:

### Retrieval Quality

| Metric | Measures | Good Value |
|--------|----------|------------|
| Precision@K | Fraction of retrieved docs that are relevant | > 0.7 |
| Recall@K | Fraction of relevant docs that are retrieved | > 0.8 |
| MRR | Position of first relevant doc | > 0.8 |
| NDCG | Ranking quality weighted by position | > 0.7 |

### Generation Quality

| Metric | Measures | How to Evaluate |
|--------|----------|-----------------|
| Faithfulness | Is response grounded in context? | LLM judge or NLI |
| Relevance | Does response answer the question? | LLM judge |
| Completeness | Are all parts of question addressed? | LLM judge |
| Conciseness | No unnecessary information? | LLM judge or length |

### End-to-End Quality

| Metric | Measures | How to Evaluate |
|--------|----------|-----------------|
| Answer correctness | Is the final answer right? | Ground truth comparison |
| User satisfaction | Did user find it helpful? | Thumbs up/down, CSAT |
| Task completion | Did user complete their goal? | Session analysis |

---

## Common Failure Modes

### Failure Mode 1: Poor Retrieval

**Symptoms:** 
- Relevant context not retrieved
- Answer says "I don't have information"
- Wrong answer based on irrelevant context

**Causes:**
- Poor chunking (breaks context)
- Weak embedding model
- Query-document semantic gap

**Solutions:**
- Use hybrid search (dense + sparse)
- Improve chunking strategy
- Add query expansion
- Use HyDE (hypothetical document embedding)

### Failure Mode 2: Lost in the Middle

**Symptoms:**
- Model ignores relevant context in middle positions
- Only uses first or last chunks

**Causes:**
- Attention bias toward start/end
- Too much context overwhelms model

**Solutions:**
- Limit to 3-5 high-quality chunks
- Rerank to put best chunks first
- Use recursive summarization for long context

### Failure Mode 3: Hallucination Despite Context

**Symptoms:**
- Model asserts facts not in context
- Confident wrong answers

**Causes:**
- Prompt not enforcing context-only
- Model's parametric knowledge overriding
- Ambiguous or partial context

**Solutions:**
- Stronger system prompt instructions
- Lower temperature
- Add "if not in context, say so" instruction
- Post-generation faithfulness check

### Failure Mode 4: Chunking Artifacts

**Symptoms:**
- Answers miss context split across chunks
- Incoherent retrieved passages

**Causes:**
- Breaking tables, lists, code blocks
- No overlap between chunks
- Wrong chunk size for content type

**Solutions:**
- Content-aware chunking
- Parent-child chunking pattern
- Increase overlap
- Document-specific chunking rules

---

## Naive RAG vs Advanced RAG

### Naive RAG

Single retrieval step, direct generation:

```
Query → Embed → Retrieve → Generate
```

**Limitations:**
- No query understanding
- Fixed retrieval strategy
- No quality checks
- Single retrieval pass

### Advanced RAG Techniques

| Technique | What It Does | When to Use |
|-----------|--------------|-------------|
| Query rewriting | Reformulate for better retrieval | Conversational context |
| Query expansion | Add related terms | Technical domains |
| HyDE | Generate hypothetical doc, then retrieve | Semantic gap |
| Reranking | Re-score with cross-encoder | Always (cheap improvement) |
| Iterative retrieval | Multiple retrieval rounds | Complex questions |
| Self-RAG | Model decides when to retrieve | Mixed query types |
| Corrective RAG | Verify and re-retrieve if needed | High-stakes applications |

### Agentic RAG

Agent decides retrieval strategy dynamically:

```python
def agentic_rag(query: str) -> str:
    # Agent loop
    while not done:
        # Decide action
        action = agent.decide(query, current_context)
        
        if action.type == "retrieve":
            chunks = retrieve(action.query)
            current_context.add(chunks)
        
        elif action.type == "verify":
            verified = verify_relevance(current_context)
            if not verified:
                action = agent.decide_retry(query)
        
        elif action.type == "answer":
            return generate(query, current_context)
```

---

## Production Considerations

### Latency Budget

```
Typical RAG latency breakdown:

Query embedding:           50-100ms
Vector search:            50-100ms
Reranking:               100-200ms
LLM generation:          500-2000ms
Network overhead:         50-100ms
─────────────────────────────────────
Total:                   750-2500ms
```

### Optimization Strategies

| Strategy | Impact | Tradeoff |
|----------|--------|----------|
| Caching | 80%+ latency reduction for repeat queries | Storage, invalidation |
| Smaller embedding model | Faster embedding | Slight quality loss |
| Fewer chunks retrieved | Faster, cheaper | May miss relevant info |
| Streaming generation | Better perceived latency | Slightly more complex |
| Async reranking | Reduce latency | Implementation complexity |

### Caching Strategies

```python
class RAGCache:
    def __init__(self):
        self.exact_cache = {}  # Query -> response
        self.embedding_cache = {}  # Text hash -> embedding
        self.semantic_cache = None  # For similar queries
    
    def get_or_compute(self, query: str) -> str:
        # 1. Exact match
        if query in self.exact_cache:
            return self.exact_cache[query]
        
        # 2. Semantic match (optional)
        if self.semantic_cache:
            similar = self.semantic_cache.find_similar(query, threshold=0.95)
            if similar:
                return similar.response
        
        # 3. Compute
        response = self.rag_pipeline(query)
        self.exact_cache[query] = response
        return response
```

### Evaluation Pipeline

```python
def evaluate_rag(test_set: list[dict]) -> dict:
    metrics = {
        "retrieval_precision": [],
        "retrieval_recall": [],
        "faithfulness": [],
        "answer_relevance": [],
        "answer_correctness": []
    }
    
    for sample in test_set:
        query = sample["query"]
        ground_truth_docs = sample["relevant_docs"]
        ground_truth_answer = sample["answer"]
        
        # Run RAG
        retrieved = retrieve(query)
        response = generate(query, retrieved)
        
        # Evaluate retrieval
        metrics["retrieval_precision"].append(
            precision(retrieved, ground_truth_docs)
        )
        metrics["retrieval_recall"].append(
            recall(retrieved, ground_truth_docs)
        )
        
        # Evaluate generation (using RAGAS or LLM judge)
        metrics["faithfulness"].append(
            evaluate_faithfulness(response, retrieved)
        )
        metrics["answer_relevance"].append(
            evaluate_relevance(response, query)
        )
        metrics["answer_correctness"].append(
            evaluate_correctness(response, ground_truth_answer)
        )
    
    return {k: mean(v) for k, v in metrics.items()}
```

---

## Interview Questions

### Q: Walk me through how you would design a RAG system for enterprise search.

**Strong answer:**
I would structure this in two phases: offline indexing and online querying.

**Offline pipeline:**
1. Connect to document sources (SharePoint, Confluence, etc.)
2. Extract and clean text with metadata
3. Chunk using semantic or recursive strategy (500 tokens, 50 overlap)
4. Embed with production model (text-embedding-3-large or Cohere)
5. Index in vector database with metadata (source, permissions, date)
6. Set up incremental sync for updates

**Online pipeline:**
1. Embed user query
2. Apply permission filters at retrieval
3. Hybrid search (dense + BM25)
4. Rerank top 20 to get top 5
5. Generate response with citations
6. Stream response back

**Key considerations:**
- Permission filtering MUST happen at retrieval, not post-generation
- Reranking provides significant quality improvement
- Caching for repeated queries
- Evaluation pipeline with ground truth test set

### Q: How do you handle hallucination in RAG systems?

**Strong answer:**
Hallucination mitigation happens at multiple layers:

**Retrieval quality:**
- If we retrieve irrelevant context, model may fall back to parametric knowledge
- Use reranking to ensure retrieved chunks are relevant
- Implement retrieval quality monitoring

**Prompt engineering:**
- Strong instruction: "Only answer based on provided context"
- Include "If not in context, say you don't know"
- Lower temperature (0.1-0.3)

**Output validation:**
- Faithfulness scoring (NLI or LLM judge)
- Citation extraction and verification
- Block responses that fail validation

**Abstention strategy:**
- Allow model to say "I don't have information about that"
- Confidence thresholding
- Human escalation for low-confidence responses

### Q: What is the difference between naive RAG and advanced RAG?

**Strong answer:**
**Naive RAG:** Single retrieve-then-generate:
- Embed query
- Retrieve top-K chunks
- Generate response

**Advanced RAG** adds multiple enhancements:

| Enhancement | Purpose |
|-------------|---------|
| Query rewriting | Handle conversational context |
| Hybrid search | Combine semantic + keyword matching |
| Reranking | Better ranking than embedding similarity alone |
| Iterative retrieval | Multi-hop reasoning |
| Self-correction | Verify and re-try if retrieval is poor |

**When to use each:**
- Naive RAG: Prototyping, simple use cases, tight latency budget
- Advanced RAG: Production systems, complex queries, high accuracy requirements

The additional latency (100-300ms) is usually worth the quality improvement.

---

## References

- Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023)
- RAGAS: https://docs.ragas.io/
- LangChain RAG Tutorial: https://python.langchain.com/docs/use_cases/question_answering/
- LlamaIndex: https://docs.llamaindex.ai/

---

*Next: [Chunking Strategies](02-chunking-strategies.md)*

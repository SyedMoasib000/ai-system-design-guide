# AI System Design Interview Question Bank

This chapter provides a comprehensive collection of interview questions organized by topic. Each question includes the depth of answer expected and key points that strong candidates cover.

## Table of Contents

- [RAG Architecture Questions](#rag-architecture-questions)
- [Agentic Systems Questions](#agentic-systems-questions)
- [Model Selection Questions](#model-selection-questions)
- [Optimization Questions](#optimization-questions)
- [Evaluation Questions](#evaluation-questions)
- [Production and MLOps Questions](#production-and-mlops-questions)
- [System Design Scenarios](#system-design-scenarios)

---

## RAG Architecture Questions

### Q1: Walk me through the architecture of a production RAG system

**What interviewers look for:**
- Understanding of the full pipeline: ingestion, indexing, retrieval, generation
- Awareness of chunking strategies and their tradeoffs
- Knowledge of embedding models and vector databases
- Understanding of reranking and its importance

**Strong answer covers:**
1. Document ingestion pipeline with preprocessing
2. Chunking strategy selection based on document types
3. Embedding model choice with cost/quality tradeoffs
4. Vector database selection criteria
5. Retrieval with hybrid search (dense + sparse)
6. Reranking layer before generation
7. Generation with proper context formatting
8. Observability and evaluation hooks

**Sample Answer:**

"A production RAG system has two main pipelines: ingestion and query.

**Ingestion pipeline:** Documents come in through various sources. First, I parse them using a document processor that handles PDFs, HTML, and Office formats. Then I chunk them, and my strategy depends on document type. For technical docs, I use recursive chunking with 512-token chunks and 50-token overlap. For legal documents, I preserve paragraph boundaries. Each chunk gets embedded using a model like text-embedding-3-large or an open source alternative like BGE if we need to self-host.

These embeddings go into a vector database. I typically use Qdrant or Pinecone depending on scale and ops requirements. Alongside vector storage, I index the raw text in Elasticsearch for keyword search.

**Query pipeline:** When a query comes in, I run hybrid search: semantic search against the vector DB and BM25 against Elasticsearch. I combine results using Reciprocal Rank Fusion. This gives me the best of both worlds since semantic handles paraphrases while keyword handles exact terms and acronyms.

I then rerank the top 50 results using a cross-encoder like Cohere Rerank or bge-reranker. This step typically improves precision by 10-15%. The top 5-10 reranked chunks become my context.

For generation, I format the context clearly with source labels, add the user query, and call the LLM with a system prompt that instructs citing sources. I use Claude or GPT-4o depending on requirements.

Finally, I have observability hooks at each stage: retrieval latency, reranker latency, LLM latency, plus quality metrics like faithfulness sampled on a percentage of requests."

**Follow-up to expect:** How would you handle documents with tables and images?

---

### Q2: When would you choose RAG over fine-tuning, and vice versa?

**What interviewers look for:**
- Clear decision framework
- Understanding of both approaches
- Cost and maintenance considerations

**Strong answer framework:**

| Factor | Favor RAG | Favor Fine-tuning |
|--------|-----------|-------------------|
| Data freshness | Frequently updated data | Static knowledge |
| Data volume | Any size works | Need 1K-100K quality examples |
| Latency tolerance | Can accept 200-500ms retrieval | Need fastest possible response |
| Use case | Factual accuracy on specific docs | Style, tone, or behavior change |
| Privacy | Data stays in your control | Training data goes to provider |
| Maintenance | Update documents any time | Retrain on data changes |

**Sample Answer:**

"The choice between RAG and fine-tuning depends on what you are trying to achieve.

**Choose RAG when:**
- Your knowledge base changes frequently. With RAG, I just update documents and they are immediately available. Fine-tuning requires retraining.
- You need citations and traceability. RAG naturally provides source attribution since I know which chunks informed the answer.
- You want to avoid hallucination on specific facts. Grounding the model in retrieved context keeps it honest.
- Data privacy is critical. Documents stay in your infrastructure rather than going to a training pipeline.

**Choose fine-tuning when:**
- You need to change the model's behavior, style, or format consistently. For example, making it always respond in a specific JSON schema or adopting a particular tone.
- Latency is extremely tight and you cannot afford retrieval overhead.
- You have stable, high-quality training examples that represent the task well.
- You want to teach the model domain-specific terminology or reasoning patterns.

**In practice, I often combine both:** I might fine-tune a model to follow our output format and tool-calling conventions, then use RAG to ground its answers in our documentation. This gives me behavioral consistency from fine-tuning and factual accuracy from RAG.

For example, at scale I might fine-tune a smaller model to handle 70% of queries efficiently, and route complex queries to a frontier model with RAG."

**Key insight to mention:** These are not mutually exclusive. Many production systems combine RAG with a fine-tuned model for best results.

---

### Q3: How do you handle the "lost in the middle" problem?

**What interviewers look for:**
- Awareness of context window attention patterns
- Practical mitigation strategies

**Strong answer covers:**
1. The problem: Models pay more attention to the beginning and end of context, less to the middle
2. Research basis: Liu et al. 2023 "Lost in the Middle" paper
3. Mitigations:
   - Limit retrieved chunks to 3-5 most relevant
   - Place critical information at start and end of context
   - Use reranking to ensure quality before stuffing context
   - Consider recursive summarization for long contexts
   - Use models with better long-context handling (Gemini 1.5, Claude 3.5)

**Sample Answer:**

"The 'lost in the middle' problem comes from research by Liu et al. in 2023. They found that LLMs pay disproportionate attention to information at the beginning and end of their context window, with reduced attention to content in the middle.

This means if I stuff 20 retrieved chunks into my context, the model might effectively ignore chunks 8-15, even if they contain the most relevant information.

**My mitigations:**

First, I limit context size. More is not always better. I typically use 5-10 high-quality chunks rather than 20 mediocre ones. Quality over quantity.

Second, I rerank aggressively before context stuffing. A cross-encoder ensures my top chunks are truly the most relevant, not just what the embedding model thought was similar.

Third, I order strategically. I put the most important chunk first, second-most-important last, and less critical ones in the middle. Some teams even duplicate critical information at both ends.

Fourth, for very long contexts, I use hierarchical approaches. I might summarize groups of related chunks and include both summaries and key verbatim sections.

Finally, model selection matters. Claude 3.5 and Gemini 1.5 Pro have shown better long-context performance than earlier models. If I must use very long contexts, I choose models specifically tested for this."

---

### Q4: Explain chunking strategies and when to use each

**What interviewers look for:**
- Knowledge of multiple strategies
- Understanding of tradeoffs
- Practical experience choosing

**Strong answer:**

| Strategy | How It Works | Best For | Tradeoff |
|----------|--------------|----------|----------|
| Fixed size | Split by token/character count | General purpose, simple docs | May break mid-sentence |
| Sentence | Split on sentence boundaries | Q&A, conversational | Variable chunk sizes |
| Semantic | Cluster by meaning similarity | Coherent topics across paragraphs | Compute cost for clustering |
| Recursive | Try large, fall back to smaller | Structured documents | Implementation complexity |
| Parent-child | Small for retrieval, return large | Need precision + context | Storage overhead |
| Document | Entire doc as one chunk | Short docs, summaries | Context length limits |

**Key insight:** Use semantic or parent-child chunking when retrieval precision matters. Use fixed size with overlap for speed and simplicity.

---

### Q5: How would you evaluate a RAG system?

**What interviewers look for:**
- Knowledge of RAG-specific metrics
- Understanding of offline vs online evaluation
- Practical evaluation pipeline design

**Strong answer covers:**

**Retrieval metrics:**
- Precision@K: What fraction of retrieved docs are relevant?
- Recall@K: What fraction of relevant docs were retrieved?
- MRR (Mean Reciprocal Rank): How high is the first relevant result?
- NDCG: Ranking quality considering position

**Generation metrics (RAGAS framework):**
- Faithfulness: Is the answer grounded in retrieved context?
- Answer relevance: Does the answer address the question?
- Context relevance: Is retrieved context actually useful?
- Context recall: Did we retrieve all needed information?

**End-to-end metrics:**
- Answer correctness vs ground truth
- User satisfaction (thumbs up/down, CSAT)
- Task completion rate

**Evaluation pipeline:**
1. Curated test set with ground truth
2. Automated evaluation with LLM-as-judge
3. Human evaluation for subset
4. A/B testing in production

**Sample Answer:**

"I evaluate RAG systems at three levels: retrieval, generation, and end-to-end.

**For retrieval evaluation**, I measure whether we are getting the right documents. Precision@K tells me what fraction of retrieved documents are actually relevant. Recall@K tells me if we missed important documents. MRR shows how high the first relevant result appears. I typically target Precision@5 above 0.8 and Recall@10 above 0.9.

**For generation evaluation**, I use the RAGAS framework. Faithfulness is critical because it measures whether the answer is grounded in the context, detecting hallucination. Answer relevance checks if we actually addressed the question. Context relevance tells me if my retrieval is fetching useful information or noise.

**For end-to-end evaluation**, I compare against ground truth when available, using exact match or semantic similarity. In production, I track user signals like thumbs up/down ratings, regeneration rate, and task completion.

**My evaluation pipeline works like this:**

Offline, I maintain a curated test set of 200+ question-answer pairs with labeled relevant documents. On every change, I run automated evaluation using RAGAS metrics and LLM-as-judge for subjective quality.

I set quality gates: faithfulness must exceed 0.85, answer relevance above 0.80. If a change degrades these, it does not ship.

In production, I sample 5% of queries for automated evaluation and track metrics over time. I also run A/B tests for significant changes, measuring user satisfaction and task completion.

Finally, I do periodic human evaluation of random samples to calibrate my automated metrics against human judgment."

---

### Q6: Describe hybrid search and when you would use it

**What interviewers look for:**
- Understanding of dense vs sparse retrieval
- Knowledge of combination methods
- Awareness of failure modes

**Strong answer:**

**Dense retrieval (embeddings):**
- Good at: Semantic similarity, paraphrases, conceptual matching
- Bad at: Exact keyword matching, rare terms, proper nouns

**Sparse retrieval (BM25, TF-IDF):**
- Good at: Exact matches, keywords, rare terms
- Bad at: Semantic similarity, synonyms

**Hybrid approach:**
1. Run both dense and sparse retrieval
2. Combine results using Reciprocal Rank Fusion (RRF) or weighted scoring
3. Rerank combined results

**When to use hybrid:**
- Domain with specific terminology (legal, medical, technical)
- Mix of keyword and conceptual queries
- When dense retrieval alone shows poor recall on exact matches

**RRF formula:** `score = sum(1 / (k + rank_i))` where k is typically 60

---

### Q7: How do you handle multi-tenant RAG systems?

**What interviewers look for:**
- Security awareness
- Understanding of isolation strategies
- Knowledge of common pitfalls

**Strong answer covers:**

**Critical principle:** Filter BEFORE retrieval, never after

```python
# WRONG: Data leaks before filtering
results = vector_db.search(query, top_k=100)
filtered = [r for r in results if r.tenant_id == tenant]

# RIGHT: Filter at database query level
results = vector_db.search(
    query, 
    top_k=10,
    filter={"tenant_id": {"$eq": tenant_id}}
)
```

**Isolation patterns by security level:**

| Pattern | Isolation | Cost | Use Case |
|---------|-----------|------|----------|
| Metadata filtering | Namespace | Low | Most SaaS apps |
| Separate collections | Collection | Medium | Sensitive data |
| Separate databases | Full | High | Regulated industries |

**Additional controls:**
- Tenant ID required in all vector metadata
- Context never contains cross-tenant data
- Cache keys scoped by tenant
- Audit logging with tenant context

**Sample Answer:**

"Multi-tenant RAG is critical for any SaaS application where different customers should only see their own data. The cardinal rule is: filter before retrieval, never after.

Here is the wrong approach:
```python
# WRONG - data leaks before filtering
results = vector_db.search(query, top_k=100)
filtered = [r for r in results if r.tenant_id == current_tenant]
```

This is dangerous because sensitive documents from other tenants are retrieved and loaded into memory. Even if you filter afterward, there are risks of logging, timing attacks, or bugs exposing that data.

The correct approach filters at the database query level:
```python
# RIGHT - filter in the database query
results = vector_db.search(
    query,
    top_k=10,
    filter={'tenant_id': {'$eq': tenant_id}}
)
```

**I implement multi-tenancy at three levels:**

**Level 1 - Metadata filtering**: Every vector includes tenant_id in metadata. All queries filter by tenant. This is the minimum for most SaaS apps.

**Level 2 - Separate collections**: Each tenant gets their own collection or namespace. Better isolation, but more operational overhead.

**Level 3 - Separate databases**: Complete isolation for regulated industries like healthcare or finance. Each tenant has their own vector DB instance.

**Other critical controls:**
- Cache keys must include tenant_id. Otherwise, one tenant might receive cached responses from another.
- Audit logging must capture tenant context for all operations.
- System prompts should never contain data from multiple tenants.
- Error messages must not leak information about other tenants' data.

I choose the isolation level based on compliance requirements and customer sensitivity."

---

### Q8: What is reranking and when would you skip it?

**What interviewers look for:**
- Understanding of two-stage retrieval
- Cost/benefit analysis
- Practical deployment experience

**Strong answer:**

**What reranking does:**
- First stage: Fast retrieval of candidates (top 50-100)
- Second stage: Expensive but accurate scoring of candidates
- Returns top K after reranking

**Reranking options:**
- Cross-encoder models (ms-marco, bge-reranker)
- Cohere Rerank API
- LLM-based reranking (expensive but flexible)

**When to skip reranking:**
- Latency budget under 200ms
- Embedding model quality is sufficient
- Cost constraints with high query volume
- Simple queries where first-stage is accurate enough

**When to use reranking:**
- Retrieval precision is critical
- Can tolerate 50-100ms additional latency
- Complex queries needing semantic understanding
- High-stakes applications (legal, medical, financial)

---

### Q9: How would you handle documents with tables, charts, and images?

**What interviewers look for:**
- Multimodal understanding
- Practical extraction strategies
- Awareness of current limitations

**Strong answer:**

**Tables:**
1. Extract table structure using document AI (Textract, Azure Doc Intelligence)
2. Options for chunking:
   - Serialize to markdown and chunk with text
   - Create separate table embeddings
   - Index table metadata with content summary
3. Consider table-specific queries that filter by table presence

**Images/Charts:**
1. Use vision-language models (GPT-4V, Claude 3.5, Gemini) for description
2. Index generated descriptions as text
3. Store image references for multimodal generation
4. For charts: consider extracting underlying data if available

**Key limitation to mention:** Many embedding models are text-only. If you embed image descriptions, retrieval quality depends on description quality.

---

### Q10: Explain vector database indexing algorithms

**What interviewers look for:**
- Understanding of ANN algorithms
- Tradeoffs between accuracy and speed
- Practical tuning experience

**Strong answer:**

**HNSW (Hierarchical Navigable Small World):**
- Graph-based approach with multiple layers
- High recall (95-99%) with low latency
- Memory intensive
- Best for: Production serving with quality requirements

**IVF (Inverted File Index):**
- Clusters vectors, searches only relevant clusters
- Trade recall for speed via nprobe parameter
- Lower memory than HNSW
- Best for: Large datasets with cost constraints

**PQ (Product Quantization):**
- Compresses vectors for memory efficiency
- Some accuracy loss
- Often combined with IVF (IVF-PQ)
- Best for: Massive scale with memory limits

**Key parameters to tune:**
- HNSW: ef_construction, ef_search, M
- IVF: nlist (clusters), nprobe (clusters to search)
- Always benchmark recall vs latency for your data

---

## Agentic Systems Questions

### Q11: What is the difference between an agent and a workflow?

**What interviewers look for:**
- Clear conceptual distinction
- Understanding of autonomy spectrum
- Practical implications for system design

**Strong answer:**

**Workflow:** Predetermined sequence of steps
- Steps are known at design time
- Control flow is explicit (if/else, loops)
- Deterministic execution path
- Easier to test, debug, and explain

**Agent:** Autonomous decision making
- Chooses actions based on observations
- Control flow determined at runtime by LLM
- Non-deterministic execution
- More flexible but harder to predict

**Autonomy spectrum:**

```
Workflows ←————————————————————————→ Agents
                                     
Single prompt → Chain → Router → ReAct → Multi-agent → Fully autonomous
```

**Key insight:** Most production systems are workflows with agentic components, not fully autonomous agents. Start with workflows, add agency where needed.

**Sample Answer:**

"The key difference is who controls the execution path.

In a **workflow**, I define the steps at design time. The code says: first do A, then do B, if condition X then do C, otherwise do D. The LLM executes within each step but does not decide the overall flow. This is deterministic and predictable.

In an **agent**, the LLM decides what to do next based on observations. I give it tools and a goal, and it chooses which tools to call in what order. The execution path is determined at runtime by the model. This is non-deterministic.

I think of it as a spectrum:

- **Single prompt**: One LLM call, no control flow
- **Chain**: Fixed sequence of LLM calls
- **Router**: LLM picks which of N paths to take
- **ReAct agent**: LLM loops with tools until done
- **Multi-agent**: Multiple LLMs coordinating

**My practical guidance**: Start with workflows. They are easier to test, debug, and explain to stakeholders. Add agentic components only where you truly need runtime flexibility.

For example, a customer support system might be a workflow where: classify intent -> retrieve context -> generate response. That is predictable. But within the retrieval step, I might use an agent that decides whether to search the knowledge base, look up order history, or both. The overall flow is controlled, but there is flexibility where needed."

---

### Q12: Explain the ReAct pattern

**What interviewers look for:**
- Understanding of the Reason + Act loop
- Knowledge of implementation details
- Awareness of failure modes

**Strong answer:**

**ReAct = Reasoning + Acting interleaved**

Loop:
1. **Thought:** LLM reasons about current state and next action
2. **Action:** LLM selects and invokes a tool
3. **Observation:** Tool returns result
4. Repeat until task complete or max iterations

**Example trace:**
```
Thought: I need to find the current stock price of NVDA
Action: stock_price(symbol="NVDA")
Observation: {"symbol": "NVDA", "price": 142.50, "currency": "USD"}
Thought: I have the price. Now I should answer the user.
Action: respond("NVIDIA stock is currently $142.50")
```

**Failure modes:**
- Tool selection errors: Wrong tool for the task
- Argument errors: Incorrect parameters
- Reasoning loops: Agent repeats same failed action
- Runaway costs: No stopping condition

**Mitigations:**
- Clear tool descriptions with examples
- Input validation on all tools
- Maximum iteration limits
- Cost tracking and alerts

**Sample Answer:**

"ReAct stands for Reasoning plus Acting. It is the most common pattern for building agents.

The agent runs in a loop with three phases:

1. **Thought**: The model reasons about the current state. What do I know? What do I still need? What should I do next?

2. **Action**: Based on that reasoning, the model selects a tool and provides arguments.

3. **Observation**: The tool executes and returns a result, which gets added to the context.

This loop continues until the model decides to give a final answer or hits a limit.

Here is a concrete example:

```
User: What is the stock price of NVIDIA and is it up or down today?

Thought: I need to get the current stock price for NVIDIA. Let me use the stock price tool.
Action: get_stock_price(symbol="NVDA")
Observation: {"symbol": "NVDA", "price": 142.50, "change": +2.3%}

Thought: I have the price and the daily change. It is up 2.3% today. I can answer now.
Final Answer: NVIDIA (NVDA) is currently trading at $142.50, up 2.3% today.
```

**The main failure modes I watch for:**

- **Loops**: Agent keeps trying the same failed action. I mitigate with max iterations and detecting repeated actions.
- **Wrong tool selection**: Agent picks an inappropriate tool. I mitigate with clear tool descriptions and examples.
- **Argument errors**: Agent passes wrong parameters. I use strict validation and return helpful error messages.
- **Runaway costs**: Agent makes many LLM calls. I track token usage and set hard limits.

ReAct is simple and works well, but for complex tasks I often prefer more structured approaches like flow engineering where I define explicit states."

---

### Q13: How do you implement tool use / function calling?

**What interviewers look for:**
- API knowledge across providers
- Tool design best practices
- Error handling understanding

**Strong answer:**

**Provider comparison (as of December 2025):**

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Parallel calls | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes |
| Tool choice control | auto/required/none | auto/any/tool | auto/any/none |
| Structured output | JSON mode | JSON mode | JSON mode |

**Tool design best practices:**
1. Clear, action-oriented names: `search_database` not `db_tool`
2. Detailed descriptions with examples in the docstring
3. Strict parameter validation with helpful error messages
4. Idempotent where possible
5. Return structured data, not prose

**Error handling:**
```python
def safe_tool_call(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return {"status": "success", "result": result}
    except ValidationError as e:
        return {"status": "error", "error_type": "validation", "message": str(e)}
    except TimeoutError:
        return {"status": "error", "error_type": "timeout", "message": "Tool timed out"}
    except Exception as e:
        return {"status": "error", "error_type": "unknown", "message": str(e)}
```

---

### Q14: How would you design a multi-agent system?

**What interviewers look for:**
- Architecture patterns
- Communication strategies
- Practical tradeoffs

**Strong answer:**

**Architecture patterns:**

| Pattern | Structure | Best For | Challenge |
|---------|-----------|----------|-----------|
| Hierarchical | Manager assigns to workers | Complex decomposable tasks | Manager becomes bottleneck |
| Peer-to-peer | Agents communicate directly | Collaborative tasks | Coordination complexity |
| Blackboard | Shared state, agents read/write | Incremental refinement | Race conditions |
| Pipeline | Sequential handoff | Staged processing | No parallelism |

**Communication approaches:**
1. **Shared state:** All agents read/write common memory
2. **Message passing:** Explicit messages between agents
3. **Orchestrator mediated:** Central coordinator routes all communication

**When to use multi-agent:**
- Task naturally decomposes into specialized subtasks
- Different tools/capabilities needed per subtask
- Parallelization provides latency benefits
- Critique/verify pattern improves quality

**When NOT to use:**
- Single agent can handle the task
- Coordination overhead exceeds benefits
- Debugging complexity is unacceptable

**Sample Answer:**

"Multi-agent systems make sense when a task naturally decomposes into specialized subtasks that benefit from different capabilities.

**Architecture patterns I consider:**

**Hierarchical (Manager-Worker)**: One manager agent decomposes the task and assigns subtasks to worker agents. The manager synthesizes results. This works well for complex tasks with clear decomposition. The risk is the manager becoming a bottleneck.

**Pipeline**: Agents hand off sequentially. Agent A does research, passes to Agent B for analysis, then Agent C for writing. Good for staged processing but no parallelism.

**Peer-to-peer**: Agents communicate directly. Good for collaborative tasks but coordination becomes complex.

**Critic/Verifier**: One agent generates, another critiques. Iterate until quality is sufficient. Powerful for improving output quality.

**Communication approaches:**

1. **Shared state**: All agents read and write to common memory. Simple but risks race conditions.
2. **Message passing**: Explicit messages between agents. More structured but more overhead.
3. **Orchestrator-mediated**: Central coordinator routes all communication. Easier to debug and monitor.

**My decision framework:**

I ask: Can a single agent with the right tools handle this? If yes, I use one agent. Simpler is better.

I use multi-agent when:
- The task spans multiple domains (research, coding, writing)
- Different tools are needed for different phases
- I want critique/verification patterns
- Parallelization provides latency benefits

For example, a content generation system might have:
- Researcher agent: Gathers information from sources
- Writer agent: Creates draft content
- Editor agent: Reviews and refines
- Fact-checker agent: Verifies claims

This separation allows specialization and parallel work where possible.

The downsides are increased complexity, harder debugging, and higher cost from multiple LLM calls. I always start simple and add agents only when they provide clear value."

---

### Q15: Explain the Model Context Protocol (MCP)

**What interviewers look for:**
- Understanding of protocol purpose
- Knowledge of architecture
- Awareness of security implications

**Strong answer:**

**What MCP solves:**
Standardizes how LLM applications connect to external tools and data sources. Think of it as a USB standard for AI tools.

**Architecture:**
- **MCP Server:** Exposes tools and resources
- **MCP Client:** LLM application that consumes tools
- **Protocol:** JSON-RPC over stdio or HTTP

**Key concepts:**
1. **Tools:** Functions the LLM can invoke
2. **Resources:** Data the LLM can read
3. **Prompts:** Reusable prompt templates
4. **Sampling:** Server can request LLM completions

**Security considerations:**
- MCP servers have host system access
- Audit what tools each server exposes
- Consider sandboxing for untrusted servers
- User consent for sensitive operations

**Current adoption (December 2025):**
- Native in Claude Desktop
- Growing ecosystem of MCP servers
- SDKs for Python and TypeScript

---

### Q16: How do you handle long-running agent tasks?

**What interviewers look for:**
- State management understanding
- Failure recovery patterns
- Practical implementation details

**Strong answer:**

**Challenges:**
- Tasks may run for minutes or hours
- Failures mid-execution lose all progress
- Cost can spiral without controls
- Users need visibility into progress

**State management patterns:**
1. **Checkpointing:** Save state after each step
2. **Event sourcing:** Log all actions, rebuild state from events
3. **Database-backed:** Persist agent state to database

**Implementation with LangGraph:**
```python
from langgraph.checkpoint import MemorySaver

# Create checkpointer
checkpointer = MemorySaver()

# Compile graph with checkpointing
app = graph.compile(checkpointer=checkpointer)

# Resume from checkpoint
config = {"configurable": {"thread_id": "task-123"}}
result = app.invoke(input, config)
```

**Reliability patterns:**
- Maximum iteration/cost limits
- Timeout per step and overall
- Dead letter queue for failed tasks
- Human escalation path

---

### Q17: What is flow engineering?

**What interviewers look for:**
- Understanding of structured agent patterns
- Knowledge of state machines for agents
- Practical design experience

**Strong answer:**

**Flow engineering** = Designing the control flow of agentic systems as explicit state machines rather than leaving all decisions to the LLM.

**Key principles:**
1. Define clear states and transitions
2. LLM decides WITHIN states, not state transitions
3. Explicit conditions for moving between states
4. Deterministic overall flow, flexible within steps

**Example: Customer support agent**

```
┌─────────────┐
│   Intake    │ ← Initial classification
└─────┬───────┘
      ↓
┌─────────────┐
│  Research   │ ← RAG retrieval
└─────┬───────┘
      ↓
┌─────────────┐     ┌─────────────┐
│  Can Answer │──No→│  Escalate   │
└─────┬───────┘     └─────────────┘
      ↓ Yes
┌─────────────┐
│  Respond    │
└─────┬───────┘
      ↓
┌─────────────┐
│  Confirm    │ ← User satisfied?
└─────────────┘
```

**Why it works:**
- Predictable behavior
- Easier testing per state
- Clear escalation points
- Cost control via state limits

---

## Model Selection Questions

### Q18: How do you choose between GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro?

**What interviewers look for:**
- Current model knowledge
- Decision framework
- Cost awareness

**Strong answer (December 2025):**

| Factor | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro |
|--------|--------|-------------------|----------------|
| Context window | 128K | 200K | 2M |
| Coding | Excellent | Best in class | Very good |
| Long context | Good | Good | Best in class |
| Vision | Yes | Yes | Yes |
| Pricing (input) | $2.50/1M | $3/1M | $1.25/1M |
| Pricing (output) | $10/1M | $15/1M | $5/1M |
| Latency (TTFT) | Fast | Fast | Medium |
| Function calling | Excellent | Excellent | Good |

**Selection framework:**

Choose **GPT-4o** when:
- Ecosystem integration matters (OpenAI tools)
- Balanced performance across tasks
- Need fastest time to first token

Choose **Claude 3.5 Sonnet** when:
- Code generation or analysis
- Complex reasoning tasks
- Need nuanced, detailed responses
- Safety/refusals are not problematic for use case

Choose **Gemini 1.5 Pro** when:
- Very long context (over 200K)
- Cost optimization is priority
- Video or audio understanding
- Multimodal grounding needed

**Sample Answer:**

"My model selection depends on the specific requirements. Here is how I think about it:

**For most production workloads**, I default to Claude 3.5 Sonnet or GPT-4o. They are both excellent general-purpose models with strong instruction following, good coding ability, and reliable function calling. Sonnet has a slight edge on coding tasks in my experience, while GPT-4o has better ecosystem integration if you are already in the OpenAI world.

**For long-context applications**, Gemini 1.5 Pro is the clear winner with its 1-2 million token context. If I am building a system that needs to process entire codebases or very long documents in a single call, Gemini is my choice. It is also the most cost-effective of the frontier models.

**For cost-sensitive high-volume applications**, I use GPT-4o-mini or Claude 3.5 Haiku. These are 10-20x cheaper than their larger siblings and handle straightforward tasks well. I often build cascading systems where simple queries go to these smaller models.

**For the most demanding reasoning tasks**, I consider o1 or Claude 3.5 Opus. These are expensive but provide measurable quality improvements on complex multi-step reasoning.

**My practical approach:**

1. Start prototyping with Claude Sonnet or GPT-4o since they are reliable and high-quality.
2. Evaluate on my specific task since benchmark rankings do not always predict task performance.
3. Build an abstraction layer so I can switch models easily.
4. Optimize costs by routing simpler requests to cheaper models once the system is stable.

I never rely solely on benchmark scores. A model that ranks lower on MMLU might excel on my specific domain."

---

### Q19: When would you use a small language model vs a frontier model?

**What interviewers look for:**
- Understanding of capability tradeoffs
- Cost optimization awareness
- Deployment considerations

**Strong answer:**

**Small models (under 10B params): Phi-3, Gemma 2, Llama 3.2, Qwen 2.5**

| Scenario | Use SLM | Use Frontier |
|----------|---------|--------------|
| Classification/routing | ✓ | |
| Simple extraction | ✓ | |
| On-device deployment | ✓ | |
| High volume, low margin | ✓ | |
| Latency under 100ms | ✓ | |
| Complex reasoning | | ✓ |
| Multi-step planning | | ✓ |
| Novel task generalization | | ✓ |
| Agentic tool selection | | ✓ |

**Cascading pattern:**
1. Route query through small classifier
2. Simple queries → SLM
3. Complex queries → Frontier model
4. Result: 70%+ cost reduction, minimal quality loss

**Deployment options for SLMs:**
- Cloud: Serverless endpoints (SageMaker, Vertex)
- Edge: ONNX, CoreML, TensorRT
- Local: Ollama, llama.cpp, vLLM

---

### Q20: Explain reasoning models (o1, DeepSeek-R1). When are they worth the cost?

**What interviewers look for:**
- Understanding of test-time compute
- Knowledge of capabilities and limitations
- Cost/benefit analysis

**Strong answer:**

**How reasoning models differ:**
- Spend more tokens "thinking" before answering
- Chain-of-thought is built into the model
- Trade latency and cost for accuracy on hard problems

**Performance profile (December 2025):**

| Model | MATH benchmark | Latency | Cost (output) |
|-------|---------------|---------|---------------|
| GPT-4o | ~76% | Fast | $10/1M |
| o1 | ~94% | 10-60s | $60/1M |
| o1-mini | ~90% | 5-30s | $12/1M |
| DeepSeek-R1 | ~92% | 10-40s | $2/1M |

**When worth the cost:**
- Mathematical proofs and formal reasoning
- Complex code debugging
- Scientific analysis
- Multi-step logical problems
- When correctness matters more than speed

**When NOT worth it:**
- Simple Q&A
- Content generation
- Latency-sensitive applications
- High volume use cases
- Tasks where GPT-4o/Claude already excel

---

### Q21: How do you evaluate and compare embedding models?

**What interviewers look for:**
- Knowledge of MTEB benchmark
- Understanding of practical evaluation
- Domain-specific considerations

**Strong answer:**

**MTEB (Massive Text Embedding Benchmark):**
- Standard benchmark for embedding quality
- Tasks: retrieval, classification, clustering, semantic similarity
- Leaderboard at huggingface.co/spaces/mteb/leaderboard

**Current top models (December 2025):**

| Model | MTEB Score | Dimensions | Max Tokens | Cost |
|-------|------------|------------|------------|------|
| OpenAI text-embedding-3-large | 64.6 | 3072 | 8191 | $0.13/1M |
| Voyage-3 | 67.8 | 1024 | 32000 | $0.06/1M |
| Cohere embed-v3 | 66.4 | 1024 | 512 | $0.10/1M |
| BGE-large-en-v1.5 | 63.9 | 1024 | 512 | Self-host |

**Practical evaluation approach:**
1. Start with MTEB as baseline
2. Create domain-specific test set
3. Evaluate retrieval precision on YOUR data
4. Consider: max token length, cost, dimensionality
5. Test multilingual if applicable

**Key insight:** MTEB scores are averages. A model ranking lower overall might excel on YOUR retrieval task. Always evaluate on domain data.

---

## Optimization Questions

### Q22: Explain the KV cache and why it matters

**What interviewers look for:**
- Technical understanding of transformer inference
- Memory calculation ability
- Optimization awareness

**Strong answer:**

**What is KV cache:**
During generation, the model computes Key and Value tensors for all previous tokens. Caching these avoids redundant computation on each new token.

**Why it matters:**
- Without cache: O(n²) compute per token
- With cache: O(n) compute per token
- Enables practical long-context generation

**Memory calculation:**
```
KV cache memory = 2 × layers × heads × head_dim × seq_len × batch × bytes

Example: Llama 2 70B, 8K context
= 2 × 80 × 64 × 128 × 8192 × 1 × 2 bytes
= ~10.7 GB per request
```

**Optimization techniques:**
1. **Grouped Query Attention (GQA):** Share K/V heads, reduce memory 4-8x
2. **PagedAttention:** Virtual memory for KV cache, reduce fragmentation
3. **Context caching:** Reuse cache for shared prefixes (system prompts)
4. **Quantize KV cache:** Store in FP8 or INT8

**Sample Answer:**

"The KV cache is fundamental to efficient LLM inference. Let me explain what it is and why it matters.

During autoregressive generation, for each new token, the model needs Key and Value tensors from all previous tokens to compute attention. Without caching, we would recompute these tensors for every previous token on every generation step, which is O(n squared) computation.

With KV cache, we store the Key and Value tensors after computing them once. Each new token only requires computing its own K and V, then attending to the cached values. This brings us to O(n) per token.

**The memory calculation:**

For a model like Llama 70B with 80 layers and GQA with 8 KV heads:
```
KV cache per token = 2 (K and V) x 80 layers x 8 heads x 128 dim x 2 bytes
                   = about 328 KB per token
```

At 8K context, that is 2.6 GB per request. With 100 concurrent requests, I need 260 GB just for KV cache, not counting model weights.

**Optimization techniques I use:**

1. **GQA/MQA**: Modern models like Llama 3 use Grouped Query Attention, sharing KV heads across multiple query heads. This reduces KV cache by 8x compared to full multi-head attention.

2. **PagedAttention** (used in vLLM): Instead of pre-allocating max sequence length, allocate pages dynamically. This eliminates memory fragmentation and can improve throughput 2-4x.

3. **Prefix caching**: For shared system prompts, compute KV cache once and reuse across requests. This is especially valuable for chat applications with long system prompts.

4. **KV cache quantization**: Store cache in INT8 or FP8 instead of FP16. This halves memory with minimal quality impact."

**Interview follow-up:** "What's the memory usage for serving 100 concurrent requests?"

---

### Q23: What is speculative decoding and when would you use it?

**What interviewers look for:**
- Understanding of the technique
- Knowledge of speedup tradeoffs
- Practical application

**Strong answer:**

**How it works:**
1. Small "draft" model generates K candidate tokens quickly
2. Large "target" model verifies all K tokens in one forward pass
3. Accept matching tokens, reject and regenerate from first mismatch
4. Net effect: Multiple tokens per target model call

**Speedup depends on:**
- Draft-target alignment (how often draft is correct)
- Draft model speed vs target
- Task complexity (easier tasks = higher acceptance)

**Typical results:**
- 2-3x speedup for well-aligned draft/target
- Exact same output as target-only (mathematically equivalent)

**When to use:**
- Latency-critical applications
- High volume serving
- Draft model available (same tokenizer required)
- Tasks with predictable patterns

**Alternatives:**
- Medusa: Multiple prediction heads instead of draft model
- Lookahead: Jacobi iteration for speculative tokens

---

### Q24: Compare batching strategies for LLM serving

**What interviewers look for:**
- Understanding of static vs dynamic batching
- Knowledge of continuous batching
- Awareness of vLLM and alternatives

**Strong answer:**

| Strategy | How It Works | Pros | Cons |
|----------|--------------|------|------|
| Static | Wait for N requests, process together | Simple | High latency at low load |
| Dynamic | Batch requests within time window | Adaptive | Still some waiting |
| Continuous | Add/remove requests mid-generation | Optimal GPU utilization | Complex implementation |
| Chunked prefill | Mix prefill and decode in batches | Balances TTFT and TPS | Recent technique |

**Continuous batching (vLLM):**
- Requests enter batch as soon as they arrive
- Completed requests exit immediately
- New requests fill freed slots
- Result: Near-optimal throughput at all load levels

**Key metrics to optimize:**
- TTFT (Time to First Token): User-perceived latency
- TPS (Tokens per Second): Throughput
- GPU utilization: Cost efficiency

**Framework comparison (December 2025):**

| Framework | Continuous Batching | PagedAttention | Multi-LoRA |
|-----------|---------------------|----------------|------------|
| vLLM | Yes | Yes | Yes |
| TGI | Yes | Yes | Yes |
| TensorRT-LLM | Yes | Yes | Limited |

---

### Q25: How do you optimize LLM inference costs?

**What interviewers look for:**
- Comprehensive cost reduction strategies
- Quantitative impact awareness
- Practical implementation experience

**Strong answer:**

**Optimization layers (in order of impact):**

1. **Model selection (50-90% savings)**
   - Use smallest model that meets quality bar
   - Cascade: cheap model first, escalate if needed
   - Fine-tuned small model often beats prompted large model

2. **Caching (30-80% reduction in API calls)**
   - Exact match cache for repeated queries
   - Semantic cache for similar queries
   - Prompt caching for shared prefixes (provider feature)

3. **Prompt optimization (20-50% token reduction)**
   - Shorter prompts with same effectiveness
   - Remove redundant instructions
   - Use structured output to reduce output length

4. **Batching (20-40% infra savings)**
   - Batch requests for throughput
   - Use batch APIs when latency allows
   - Off-peak processing for async tasks

5. **Infrastructure (variable)**
   - Spot instances for fault-tolerant workloads
   - Right-size GPU selection
   - Quantized models for self-hosted

**Measurement:**
- Track cost per query
- Track cost per user action
- Set alerting on cost spikes
- A/B test optimization changes

**Sample Answer:**

"I approach LLM cost optimization in layers, starting with the highest-impact changes.

**Layer 1: Model selection** has the biggest impact, potentially 50-90% savings. The question is: what is the cheapest model that meets my quality bar? I run evaluations to find this. Often GPT-4o-mini or Claude Haiku handles 60-70% of queries just fine, and I only route complex queries to frontier models.

**Layer 2: Caching** can reduce API calls by 30-80%. I implement two levels:
- Exact match cache for repeated queries
- Semantic cache for similar queries (if embedding similarity exceeds 0.95, return cached response)

For chat applications, prompt caching from providers like Anthropic is valuable since system prompts are cached on their side.

**Layer 3: Prompt optimization** reduces tokens 20-50%. I audit prompts regularly:
- Remove redundant instructions
- Use concise language
- Request structured output to limit response length
- Use few-shot examples sparingly

**Layer 4: Batching** saves 20-40% on infrastructure. For async workloads, I batch requests. OpenAI offers 50% discount on batch API. For sync workloads, continuous batching in vLLM maximizes GPU utilization.

**Layer 5: Infrastructure optimization** varies by setup. For self-hosted, I use quantized models (AWQ 4-bit), right-size GPU selection, and spot instances for fault-tolerant workloads.

**I always measure:**
- Cost per query (broken down by component)
- Cost per successful user action
- Token efficiency (output value per token spent)

I set alerts for cost spikes and A/B test any optimization to ensure quality is maintained."

---

### Q26: Explain quantization techniques for LLM deployment

**What interviewers look for:**
- Understanding of quantization methods
- Quality vs efficiency tradeoffs
- Practical deployment experience

**Strong answer:**

| Method | Bits | Memory Reduction | Quality Loss | Use Case |
|--------|------|------------------|--------------|----------|
| FP16 | 16 | 2x vs FP32 | None | Training, high-quality inference |
| INT8 (LLM.int8) | 8 | 2x vs FP16 | Minimal | Production serving |
| GPTQ | 4 | 4x vs FP16 | Small | Edge, cost-sensitive |
| AWQ | 4 | 4x vs FP16 | Smaller than GPTQ | Production 4-bit |
| GGUF Q4_K_M | 4 | 4x vs FP16 | Small | CPU inference, llama.cpp |

**How quantization works:**
- Reduce precision of weights (and optionally activations)
- Fewer bits = less memory = faster memory transfer
- Quality loss from rounding errors

**AWQ advantage:**
- Activation-aware: Protects high-impact weights
- Better quality than naive quantization
- Fast inference with optimized kernels

**Practical guidance:**
- Start with AWQ 4-bit for most deployments
- Use INT8 if 4-bit quality is insufficient
- GGUF for CPU-only deployment
- Always benchmark on YOUR tasks before deploying

---

## Evaluation Questions

### Q27: How do you evaluate LLM outputs when there is no ground truth?

**What interviewers look for:**
- Understanding of LLM-as-judge
- Knowledge of bias mitigation
- Practical evaluation pipeline design

**Strong answer:**

**LLM-as-Judge approach:**
1. Define evaluation criteria (fluency, relevance, accuracy, etc.)
2. Provide rubric with examples of each score
3. Have judge LLM score outputs
4. Aggregate across multiple judges or multiple passes

**Bias mitigations:**
- Position bias: Randomize order of options
- Verbosity bias: Normalize for length
- Self-enhancement: Use different model as judge
- Provide scoring rubric with examples

**Evaluation prompt structure:**
```
You are evaluating a response on a scale of 1-5 for relevance.

Scoring rubric:
1 - Completely irrelevant
2 - Tangentially related
3 - Partially relevant
4 - Mostly relevant
5 - Highly relevant

Question: {question}
Response: {response}

Score (1-5):
Reasoning:
```

**Calibration:**
- Include known good/bad examples
- Check inter-rater reliability
- Validate against human judgments on subset

**Sample Answer:**

"When there is no ground truth, I use LLM-as-judge as my primary evaluation method, with careful calibration.

**My approach:**

First, I define clear evaluation criteria. For a customer support bot, I might evaluate:
- Correctness: Is the information accurate?
- Relevance: Does it answer the question?
- Helpfulness: Would this actually help the user?
- Tone: Is it professional and empathetic?

Then I create a detailed rubric with examples at each score level. This is critical for consistency:

```
Helpfulness (1-5 scale):
5 - Fully resolves the user's issue with clear next steps
4 - Addresses main concern with minor gaps
3 - Partially helpful but missing key information
2 - Tangentially related but does not solve the problem
1 - Unhelpful or irrelevant
```

I include 2-3 examples of responses at each level so the judge LLM calibrates correctly.

**Bias mitigation is essential:**

- **Position bias**: If comparing two responses, I run the evaluation twice with swapped positions. If the winner changes, I mark it as a tie.
- **Length bias**: Some models prefer longer responses. I explicitly instruct to ignore length.
- **Self-preference**: I use a different model as judge than the one being evaluated. Claude judging GPT outputs, for example.

**Validation process:**

I take a sample of 50-100 evaluations and have humans rate them independently. I compute correlation between LLM-judge scores and human scores. If correlation is below 0.7, I revise my rubric and examples.

I also include 'calibration examples' with known scores in each batch. If the judge scores these correctly, I have more confidence in the other scores.

LLM-as-judge is not perfect, but with proper calibration it is practical for rapid iteration. For high-stakes decisions, I supplement with human evaluation."

---

### Q28: Explain the RAGAS evaluation framework

**What interviewers look for:**
- Knowledge of RAG-specific metrics
- Implementation understanding
- Practical usage

**Strong answer:**

**RAGAS metrics:**

| Metric | Measures | How Calculated |
|--------|----------|----------------|
| Faithfulness | Is answer grounded in context? | LLM checks if claims are supported |
| Answer Relevance | Does answer address question? | LLM generates questions from answer, compare to original |
| Context Relevance | Is retrieved context useful? | LLM rates relevance of each chunk |
| Context Recall | Did we get all needed info? | Compare retrieved to ground truth contexts |

**Implementation:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Prepare dataset
dataset = {
    "question": [...],
    "answer": [...],
    "contexts": [...],
    "ground_truth": [...]  # Optional
}

# Run evaluation
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

**Usage patterns:**
- Offline evaluation on test set
- Continuous monitoring sample in production
- A/B testing different RAG configurations
- Debugging retrieval vs generation issues

---

### Q29: How do you detect and handle hallucinations?

**What interviewers look for:**
- Understanding of hallucination types
- Detection strategies
- Mitigation techniques

**Strong answer:**

**Hallucination types:**
1. **Factual:** Incorrect facts about the world
2. **Faithfulness:** Claims not supported by provided context
3. **Fabrication:** Making up sources, citations, quotes

**Detection strategies:**

| Strategy | Approach | Tradeoff |
|----------|----------|----------|
| Cross-reference | Check against knowledge base | Coverage limited |
| Self-consistency | Generate multiple times, check agreement | Cost |
| Citation verification | Require and verify citations | Latency |
| NLI models | Check entailment between source and claim | Accuracy varies |
| Confidence calibration | LLM rates its own confidence | Unreliable for some models |

**Mitigation techniques:**
1. **Retrieval grounding:** Only answer from retrieved context
2. **Citation enforcement:** Force model to cite sources
3. **Abstention:** Allow "I don't know" responses
4. **Temperature:** Lower temperature reduces creativity/hallucination
5. **Guardrails:** Post-generation fact checking

**System prompt guidance:**
```
Only answer based on the provided context. 
If the context does not contain the information needed, say "I don't have information about that."
Always cite the source document for each claim.
```

**Sample Answer:**

"Hallucination is when the model generates content that is not grounded in reality or the provided context. I categorize it into three types:

1. **Factual hallucination**: Incorrect facts about the real world
2. **Faithfulness hallucination**: Claims not supported by the provided context (most relevant for RAG)
3. **Fabrication**: Making up citations, quotes, or sources that do not exist

**My detection strategies:**

**For RAG systems**, I check faithfulness using NLI models or LLM-as-judge. I extract claims from the response and verify each is entailed by the context. RAGAS faithfulness metric does exactly this.

**Self-consistency checking**: Generate the response multiple times with temperature above 0. If answers are inconsistent, confidence is low. High-confidence factual claims should be consistent.

**Citation verification**: If the model claims 'According to Document X...', I verify that Document X actually contains that information.

**My mitigation strategies:**

**1. Grounding in retrieval**: I instruct the model to only answer from provided context. My system prompt includes: 'If the information is not in the context, say you do not know.'

**2. Enable abstention**: Train or prompt the model to say 'I do not have information about that' rather than guessing. This is culturally difficult since models are trained to be helpful, but it is crucial.

**3. Force citations**: Require the model to cite specific sources for each claim. This makes hallucinations easier to spot and reduces their frequency.

**4. Temperature settings**: Lower temperature (0.1-0.3) for factual tasks reduces creative hallucination.

**5. Post-generation verification**: Run a fact-checking pass on the response before returning to the user. This adds latency but catches issues.

The key insight is that hallucination cannot be fully eliminated. I design systems that detect it and gracefully handle it rather than assuming it will not happen."

---

## Production and MLOps Questions

### Q30: How do you implement observability for LLM applications?

**What interviewers look for:**
- Understanding of what to measure
- Tracing implementation
- Practical tooling knowledge

**Strong answer:**

**Three pillars for LLM apps:**

1. **Logs**
   - Request/response (or hashes for privacy)
   - Model used, parameters
   - Token counts
   - Latency breakdown

2. **Metrics**
   - Request volume, latency (p50, p95, p99)
   - Token usage (input/output)
   - Cost per request
   - Error rates by type
   - Cache hit rates
   - Quality scores (sampled)

3. **Traces**
   - End-to-end request flow
   - Each LLM call with prompts/completions
   - Retrieval steps with chunks returned
   - Tool calls and results

**Tooling options:**
- LangSmith: LangChain native
- Langfuse: Open source
- OpenTelemetry: Standard instrumentation
- Weights & Biases: ML-focused
- Custom: OpenTelemetry + your stack

**Essential dashboard:**
- Request volume over time
- Latency percentiles
- Token usage and cost
- Error rate
- Quality score trends

**Sample Answer:**

"Observability for LLM applications requires adapting the three pillars of logs, metrics, and traces for the unique characteristics of LLM systems.

**Logging:**

I log every LLM call with:
- Request ID for correlation
- Model and parameters used
- Token counts (input and output)
- Latency (TTFT and total)
- Input and output content (or hashes if privacy-sensitive)

For RAG systems, I also log retrieved chunks and their scores so I can debug retrieval quality.

**Metrics:**

My core dashboard includes:
- Request volume and error rates
- Latency percentiles: p50, p95, p99
- Token usage: input tokens, output tokens, by model
- Cost: real-time cost tracking per request and daily totals
- Cache hit rates (if using caching)
- Quality scores: sampled LLM-as-judge scores over time

I set alerts for:
- Error rate exceeds 5%
- P95 latency exceeds SLA
- Cost spikes above 2x normal
- Quality score drops below threshold

**Tracing:**

End-to-end tracing is crucial for debugging. For a RAG request, my trace shows:
- User query received
- Embedding generated (latency)
- Vector search performed (latency, chunks retrieved)
- Reranking completed (latency, final chunks)
- LLM called (latency, tokens, model)
- Response returned

This lets me identify bottlenecks and debug quality issues by seeing exactly what context was used.

**Tooling:**

I use LangSmith or Langfuse for LLM-specific tracing since they understand prompts and completions. For metrics, I use standard tools like Prometheus and Grafana. For logs, I use a centralized system with structured logging.

The key insight is that LLM observability must include quality metrics, not just operational metrics. A system that is fast and available but producing low-quality responses is failing."

---

### Q31: Describe CI/CD for LLM applications

**What interviewers look for:**
- Understanding of what to test
- Prompt versioning awareness
- Evaluation integration

**Strong answer:**

**What changes in LLM apps:**
- Prompts (most frequent)
- Retrieved context (data updates)
- Model versions
- Parameters (temperature, etc.)
- Application code

**CI Pipeline:**
1. **Unit tests:** Core logic, data processing
2. **Prompt tests:** Specific scenarios with expected behaviors
3. **Evaluation suite:** Run RAGAS or custom metrics on test set
4. **Cost estimation:** Project cost impact of changes

**Prompt versioning:**
- Version all prompts in code or config
- Associate evaluation results with versions
- Enable rollback to previous versions

**CD considerations:**
- Gradual rollout (1% → 10% → 100%)
- Monitor quality metrics during rollout
- Automatic rollback triggers
- A/B test significant changes

**Evaluation gates:**
```yaml
quality_gates:
  faithfulness: >= 0.85
  answer_relevance: >= 0.80
  latency_p95: <= 2000ms
  cost_per_query: <= $0.05
```

---

### Q32: How do you handle rate limits and quotas?

**What interviewers look for:**
- Practical experience with API limits
- Graceful degradation strategies
- Multi-provider patterns

**Strong answer:**

**Rate limit types:**
- Requests per minute (RPM)
- Tokens per minute (TPM)
- Tokens per day (TPD)
- Concurrent requests

**Handling strategies:**

| Strategy | Implementation | Use Case |
|----------|---------------|----------|
| Queue with backoff | Queue requests, retry with exponential backoff | Standard handling |
| Request batching | Combine multiple queries | Reduce request count |
| Priority queues | Urgent requests get quota first | Mixed priority traffic |
| Multi-provider fallback | Route to backup provider | High availability |
| Caching | Return cached for repeated queries | Reduce redundant calls |
| Load shedding | Reject low-priority requests | Overload protection |

**Implementation example:**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError)
)
async def call_llm_with_retry(prompt):
    return await llm.generate(prompt)
```

**Monitoring:**
- Track rate limit errors
- Alert on approaching quotas
- Dashboard showing quota utilization

---

### Q33: Describe strategies for LLM application security

**What interviewers look for:**
- Comprehensive threat awareness
- Defense in depth approach
- Practical controls

**Strong answer:**

**Threat categories:**

| Layer | Threat | Mitigation |
|-------|--------|------------|
| Input | Prompt injection | Input validation, instruction hierarchy |
| Input | Jailbreaking | Refusal training, output filtering |
| Data | Context leakage | Tenant isolation, permission checks |
| Data | PII exposure | Detection, redaction, anonymization |
| Output | Harmful content | Output filtering, guardrails |
| Output | Hallucinated secrets | Never put secrets in prompts |

**Defense in depth:**
1. **Input validation:** Regex, length limits, encoding checks
2. **Input transformation:** Potentially paraphrase untrusted input
3. **Instruction hierarchy:** System > user separation
4. **Context filtering:** Permission-based retrieval
5. **Output filtering:** Content classifiers, PII detection
6. **Monitoring:** Anomaly detection on inputs/outputs

**Multi-tenant isolation (critical):**
- Tenant ID in all data
- Filter at retrieval time, not post-generation
- Scoped caches per tenant
- Audit logging with tenant context

---

## Ensemble Methods Questions

### Q40: When would you use Self-Consistency vs Best-of-N sampling?

**What they're testing:**
- Understanding of inference-time compute tradeoffs
- Knowledge of appropriate use cases for each technique
- Practical cost-accuracy considerations

**Approach:**
1. Define both techniques
2. Explain when each excels
3. Discuss the key differentiator: extractable answers vs open-ended

**Sample Answer:**

"These serve fundamentally different purposes:

**Self-Consistency** is for tasks with extractable, verifiable answers. I generate k reasoning paths with temperature 0.5-0.8, extract the final answer from each, and take a majority vote. This works for:
- Math problems (extract final number)
- Multiple choice (vote on labels)
- Short-form QA (vote on answers)

The key requirement is that I can compare answers for equality.

**Best-of-N** is for open-ended generation where there is no single right answer. I generate N samples, score each with a reward model, and select the best. This works for:
- Creative writing
- Code generation (many valid solutions)
- Explanations

Here I need a reward model or judge since I cannot just compare for equality.

**Key decision:** Can I extract and compare answers? If yes, use Self-Consistency. If no, use Best-of-N.

I would not use Self-Consistency for creative writing (no extractable answer) or Best-of-N for math (voting is simpler and cheaper than reward scoring)."

---

### Q41: How do you prevent reward hacking when using Best-of-N?

**What they're testing:**
- Awareness of reward model failure modes
- Understanding of ensemble techniques for robustness
- Practical mitigation strategies

**Approach:**
1. Define reward hacking
2. Explain why it happens
3. Provide multiple mitigation strategies

**Sample Answer:**

"Reward hacking is when the model exploits weaknesses in the reward model rather than genuinely improving quality. For example, the model might learn that longer responses score higher, so it pads with filler.

**My mitigations:**

1. **Reward model ensemble**: Use 3+ diverse reward models. A sample that hacks one RM is unlikely to hack all of them.

2. **Conservative aggregation**: Instead of mean score, use 25th percentile or minimum. This selects samples that score well across all RMs.

3. **Diversity monitoring**: If sample diversity drops, the model may be exploiting a narrow hack. I track embedding diversity across samples.

4. **Human calibration**: Periodically validate that RM-selected samples match human preferences.

5. **Multi-dimensional scoring**: Score on quality, safety, relevance separately. Require good scores on all dimensions.

The key insight is that any single reward signal can be gamed. Ensembles make gaming much harder."

---

### Q42: Design an evaluation system for comparing two LLMs on open-ended tasks.

**What they're testing:**
- Knowledge of LLM-as-judge techniques
- Awareness of evaluation biases
- Practical evaluation pipeline design

**Strong answer includes:**
- Panel of judges for reduced bias
- Pairwise comparison with positional debiasing
- Inter-rater agreement metrics
- Human calibration

**Sample Answer:**

"Comparing LLMs on open-ended tasks requires careful evaluation design to avoid biases.

**My approach:**

1. **Panel of diverse judges**: Use 3-5 models from different families (Claude, GPT-4, Gemini) as judges. Same-family models share biases, so diversity matters.

2. **Pairwise comparison with positional debiasing**: Models prefer the first position 60-70% of the time. I run each comparison twice with swapped positions. If winner changes with position, I mark it as a tie.

3. **Structured rubric**: Clear criteria with examples at each score level. This improves consistency across judges.

4. **Inter-rater agreement**: I track how often judges agree. Low agreement indicates the task is ambiguous or judges need calibration.

5. **Human validation**: I validate a sample of evaluations against human preferences. If correlation is below 0.7, I revise my rubric.

For statistical significance, I use at least 500 comparison pairs and compute confidence intervals on win rates."

---

### Q43: What is the difference between ensemble learning and model arbitration?

**What they're testing:**
- Conceptual clarity on aggregation vs selection
- Understanding when to use each approach

**Sample Answer:**

"These are fundamentally different approaches:

**Ensemble learning** combines outputs from all models into a blended prediction. The relationship is collaborative - models compensate for each other's errors. Methods include voting, averaging, stacking. The final output is a composite derived from all models.

**Model arbitration** selects a single best output from candidates. The relationship is competitive - outputs are judged against each other. Methods include reward model scoring, ranking, routing. The final output comes from one chosen winner.

**When to use each:**

Use **ensemble** when:
- There is a correct answer format (classification, math)
- You want robustness and reduced variance
- All models contribute useful signal

Use **arbitration** when:
- Output is open-ended (creative, explanations)
- You want best quality, not average quality
- You have a reliable scoring function

They can be combined: generate diverse candidates (benefits from ensemble thinking), then select the best (arbitration). A panel of judges uses ensemble for scoring, then arbitration for final selection."

---

### Q44: When would you use Multi-Agent Debate vs Mixture of Agents?

**What they're testing:**
- Understanding of multi-model coordination patterns
- Ability to match patterns to use cases

**Sample Answer:**

"These are different coordination patterns with different purposes:

**Multi-Agent Debate** is adversarial. Multiple models critique each other over 2-3 rounds. Each model sees others' answers and must defend or revise their position. Best for:
- Fact verification (catching hallucinations)
- Error correction (finding mistakes)
- Complex reasoning (stress-testing logic)

The value is adversarial pressure that catches errors.

**Mixture of Agents (MoA)** is collaborative. Layer 1 models generate diverse perspectives, Layer 2 aggregator synthesizes them. Best for:
- Complex synthesis (reports, summaries)
- Multi-domain problems (need different expertise)
- Creative tasks (want diverse ideas combined)

The value is combining complementary strengths.

**Decision:**
- Need to verify/challenge: Use Debate
- Need to synthesize/combine: Use MoA

For a financial report, I might use both: MoA to generate comprehensive analysis from different perspectives, then Debate to verify factual claims before publication."

---

## System Design Scenarios

### Scenario 1: Design a customer support chatbot

**Time:** 35 minutes

**Requirements:**
- 10,000 tickets/day
- Multi-language (5 languages)
- Access to product documentation and order history
- Integration with ticketing system
- Human handoff capability

**Strong answer structure:**

1. **Clarifying questions (2 min)**
   - What percentage of tickets should be fully automated?
   - What SLA for first response?
   - Are there compliance requirements?
   - What is the existing tech stack?

2. **High-level architecture (5 min)**
   - Draw: User → API Gateway → Chat Service → Agent → RAG + Tools → LLM
   - Identify key components

3. **Data pipeline (5 min)**
   - Documentation ingestion with chunking
   - Order history API integration
   - Multi-language embedding strategy

4. **Agent design (10 min)**
   - Intent classification first (route simple vs complex)
   - RAG for documentation queries
   - Tool use for order lookup, ticket creation
   - Escalation criteria
   - State machine for conversation flow

5. **Multi-language (5 min)**
   - Multilingual embedding model
   - Translation layer or multilingual LLM
   - Language detection on input

6. **Reliability and observability (5 min)**
   - Fallback to human on low confidence
   - Latency and quality monitoring
   - Cost tracking per conversation

7. **Scaling considerations (3 min)**
   - Cache frequent queries
   - Batch non-urgent operations
   - Auto-scaling based on ticket volume

---

### Scenario 2: Design a document processing pipeline

**Time:** 35 minutes

**Requirements:**
- 100,000 documents/day (PDF, images, scanned)
- Extract structured data (invoices, contracts, forms)
- 99% accuracy requirement
- HIPAA compliance

**Strong answer structure:**

1. **Clarifying questions**
   - What document types specifically?
   - What structured fields need extraction?
   - What is acceptable latency?
   - Human-in-the-loop for low confidence?

2. **Pipeline architecture**
   ```
   Upload → Classification → OCR/Extraction → Validation → Human Review → Output
   ```

3. **Document classification**
   - Fine-tuned classifier for document types
   - Route to type-specific extraction

4. **Extraction approach**
   - Document AI (Textract, Azure Doc Intelligence) for structured forms
   - Vision LLM for complex/variable layouts
   - Combine outputs for high accuracy

5. **Validation layer**
   - Schema validation
   - Cross-field consistency
   - Business rule checks
   - Confidence thresholds

6. **Human-in-the-loop**
   - Queue low-confidence extractions
   - Reviewer interface with corrections
   - Feedback loop to improve model

7. **HIPAA compliance**
   - PHI detection and handling
   - Encryption at rest and in transit
   - Audit logging
   - Access controls

---

### Scenario 3: Design a RAG system for enterprise search

**Time:** 35 minutes

**Requirements:**
- 10 million documents
- 50,000 employees
- Role-based access control
- Real-time document updates

**Key points to cover:**
1. Multi-tenant architecture with permission filtering
2. Chunking strategy for mixed document types
3. Hybrid search (dense + sparse)
4. Real-time indexing pipeline
5. Caching for common queries
6. Evaluation and quality monitoring

---

### Scenario 4: Design a code assistant

**Time:** 35 minutes

**Requirements:**
- IDE integration
- Repository-aware context
- Code generation and explanation
- Streaming responses

**Key points to cover:**
1. Repository indexing (code-specific chunking)
2. Context assembly (current file, imports, related files)
3. Latency optimization (caching, streaming)
4. Code-specific evaluation metrics
5. Privacy considerations for proprietary code

---

### Scenario 5: Design an AI-powered content moderation system

**Time:** 35 minutes

**Requirements:**
- 1 million posts/day
- Multi-modal (text, images, video)
- Low latency (under 500ms)
- Appeal workflow

**Key points to cover:**
1. Cascading classifiers (cheap → expensive)
2. Multi-modal processing pipeline
3. Threshold tuning for precision/recall
4. Human review queue
5. Feedback loop for model improvement

---

## Interview Tips Summary

1. **Always discuss tradeoffs** - No decision is free
2. **Lead with clarifying questions** - Scope the problem
3. **Think out loud** - Show your reasoning process
4. **Use real numbers** - Latency, cost, throughput
5. **Consider failure modes** - What can go wrong?
6. **End with monitoring** - How do you know it works?
7. **Acknowledge uncertainty** - It is okay to say "I would research this more"

---

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Anthropic Documentation](https://docs.anthropic.com/)
- Liu et al. "Lost in the Middle: How Language Models Use Long Contexts" 2023
- Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" 2023

---

*See also: [Answer Frameworks](02-answer-frameworks.md) | [Common Pitfalls](03-common-pitfalls.md) | [Whiteboard Exercises](04-whiteboard-exercises.md)*

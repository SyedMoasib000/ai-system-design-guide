# Context Management

Context management is how you decide what information to include in the LLM's input. With limited context windows and costs per token, effective context management is critical for production systems.

## Table of Contents

- [Context Window Fundamentals](#context-window-fundamentals)
- [Context Budget Allocation](#context-budget-allocation)
- [Context Selection Strategies](#context-selection-strategies)
- [Context Compression](#context-compression)
- [Long Context Handling](#long-context-handling)
- [Conversation Context](#conversation-context)
- [Production Patterns](#production-patterns)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Context Window Fundamentals

### What Is a Context Window

The context window is the maximum number of tokens the model can process in a single request (input + output combined).

| Model | Context Window | Practical Input Limit* |
|-------|----------------|----------------------|
| GPT-4o | 128K | ~100K |
| Claude 3.5 Sonnet | 200K | ~180K |
| Gemini 1.5 Pro | 1M | ~900K |
| Llama 3.1 70B | 128K | ~100K |

*Leave room for output tokens

### Context Window Costs

Larger context has tradeoffs:

| Aspect | Short Context | Long Context |
|--------|---------------|--------------|
| Latency | Fast | Slower (prefill time) |
| Cost | Lower | Higher (per token) |
| Accuracy | Focused | May "lose" information |
| Flexibility | Limited | More options |

### The "Lost in the Middle" Problem

Research shows models pay less attention to content in the middle of long contexts:

```
Attention distribution (simplified):
████████░░░░░░░░████████
^                      ^
Beginning           End (most recent)

Information in the middle may be underweighted.
```

**Mitigation:** Put most important information at the beginning and end.

---

## Context Budget Allocation

### Typical RAG Budget

For a standard RAG application with 8K context:

```
┌─────────────────────────────────────────────────────────────────┐
│                    8192 Token Budget                            │
├─────────────────────────────────────────────────────────────────┤
│ System prompt          │  500 tokens  │  6%                     │
│ Retrieved context      │ 4000 tokens  │ 49%                     │
│ Conversation history   │ 1500 tokens  │ 18%                     │
│ User message           │  500 tokens  │  6%                     │
│ Reserved for output    │ 1692 tokens  │ 21%                     │
└─────────────────────────────────────────────────────────────────┘
```

### Budget Allocation Code

```python
class ContextBudget:
    def __init__(self, max_tokens: int, output_reserve: int = 1000):
        self.max_tokens = max_tokens
        self.output_reserve = output_reserve
        self.available = max_tokens - output_reserve
        
        # Default allocations (can be adjusted)
        self.allocations = {
            "system": 0.10,      # 10%
            "retrieved": 0.50,   # 50%
            "history": 0.25,     # 25%
            "user": 0.15         # 15%
        }
    
    def get_budget(self, component: str) -> int:
        return int(self.available * self.allocations[component])
    
    def allocate(self, components: dict[str, str]) -> dict[str, str]:
        """Allocate context to components, truncating as needed."""
        result = {}
        
        for name, content in components.items():
            budget = self.get_budget(name)
            tokens = count_tokens(content)
            
            if tokens > budget:
                result[name] = truncate_to_tokens(content, budget)
            else:
                result[name] = content
        
        return result

# Usage
budget = ContextBudget(max_tokens=8192)
allocated = budget.allocate({
    "system": system_prompt,
    "retrieved": retrieved_docs,
    "history": conversation_history,
    "user": user_message
})
```

---

## Context Selection Strategies

### Strategy 1: Relevance-Based Selection

Select the most relevant content for the query:

```python
def select_by_relevance(
    query: str,
    candidates: list[str],
    max_tokens: int
) -> list[str]:
    # Score candidates by relevance
    query_embedding = embed(query)
    scored = []
    
    for candidate in candidates:
        embedding = embed(candidate)
        score = cosine_similarity(query_embedding, embedding)
        scored.append((candidate, score))
    
    # Sort by relevance
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Select until budget exhausted
    selected = []
    total_tokens = 0
    
    for candidate, score in scored:
        tokens = count_tokens(candidate)
        if total_tokens + tokens > max_tokens:
            break
        selected.append(candidate)
        total_tokens += tokens
    
    return selected
```

### Strategy 2: Recency-Based Selection

For conversations, prioritize recent messages:

```python
def select_by_recency(
    messages: list[dict],
    max_tokens: int,
    always_include_first: bool = True
) -> list[dict]:
    # Always include system message if present
    selected = []
    total_tokens = 0
    
    if always_include_first and messages and messages[0]["role"] == "system":
        selected.append(messages[0])
        total_tokens += count_tokens(messages[0]["content"])
        messages = messages[1:]
    
    # Add messages from most recent, going backward
    for message in reversed(messages):
        tokens = count_tokens(message["content"])
        if total_tokens + tokens > max_tokens:
            break
        selected.insert(1 if selected else 0, message)
        total_tokens += tokens
    
    return selected
```

### Strategy 3: Importance-Based Selection

Score content by importance, not just relevance:

```python
def select_by_importance(
    documents: list[Document],
    query: str,
    max_tokens: int
) -> list[Document]:
    scored = []
    
    for doc in documents:
        # Multi-factor scoring
        relevance = compute_relevance(query, doc)
        recency = compute_recency_score(doc.date)
        authority = doc.source_authority  # Pre-computed
        
        # Weighted combination
        score = (
            0.5 * relevance +
            0.3 * recency +
            0.2 * authority
        )
        scored.append((doc, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Select within budget
    return select_within_budget(scored, max_tokens)
```

### Strategy 4: Diversity-Based Selection

Avoid redundant information:

```python
def select_with_diversity(
    documents: list[Document],
    query: str,
    max_tokens: int,
    diversity_weight: float = 0.3
) -> list[Document]:
    selected = []
    selected_embeddings = []
    
    candidates = [(doc, compute_relevance(query, doc)) for doc in documents]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    total_tokens = 0
    
    for doc, relevance in candidates:
        tokens = count_tokens(doc.text)
        if total_tokens + tokens > max_tokens:
            continue
        
        # Check diversity against already selected
        if selected_embeddings:
            embedding = embed(doc.text)
            max_similarity = max(
                cosine_similarity(embedding, e) 
                for e in selected_embeddings
            )
            
            # Penalize documents too similar to already selected
            diversity_score = 1 - max_similarity
            final_score = (1 - diversity_weight) * relevance + diversity_weight * diversity_score
            
            if final_score < 0.5:  # Threshold
                continue
            
            selected_embeddings.append(embedding)
        else:
            selected_embeddings.append(embed(doc.text))
        
        selected.append(doc)
        total_tokens += tokens
    
    return selected
```

---

## Context Compression

### Summarization-Based Compression

```python
def compress_with_summarization(
    documents: list[str],
    target_tokens: int,
    preserve_key_facts: bool = True
) -> str:
    current_tokens = sum(count_tokens(d) for d in documents)
    
    if current_tokens <= target_tokens:
        return "\n\n".join(documents)
    
    # Calculate compression ratio needed
    ratio = target_tokens / current_tokens
    
    # Summarize each document proportionally
    compressed = []
    for doc in documents:
        doc_tokens = count_tokens(doc)
        target_doc_tokens = int(doc_tokens * ratio)
        
        summary = llm.generate(
            f"Summarize this in approximately {target_doc_tokens} tokens. "
            f"Preserve all key facts and numbers.\n\n{doc}"
        )
        compressed.append(summary)
    
    return "\n\n".join(compressed)
```

### Extraction-Based Compression

Extract only relevant parts:

```python
def extract_relevant_sentences(
    document: str,
    query: str,
    max_sentences: int = 10
) -> str:
    sentences = split_sentences(document)
    query_embedding = embed(query)
    
    scored = []
    for sentence in sentences:
        embedding = embed(sentence)
        score = cosine_similarity(query_embedding, embedding)
        scored.append((sentence, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, _ in scored[:max_sentences]]
    
    # Maintain original order
    original_order = [s for s in sentences if s in top_sentences]
    
    return " ".join(original_order)
```

### LLMLingua-Style Compression

Remove less important tokens:

```python
def compress_tokens(
    text: str,
    compression_ratio: float = 0.5
) -> str:
    """
    Compress text by removing less important tokens.
    Uses a small model to score token importance.
    """
    tokens = tokenize(text)
    
    # Score each token's importance
    importance_scores = score_token_importance(tokens)
    
    # Keep top tokens by importance
    num_to_keep = int(len(tokens) * compression_ratio)
    
    # Sort by importance but preserve relative order
    indexed_scores = list(enumerate(importance_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    keep_indices = set(i for i, _ in indexed_scores[:num_to_keep])
    
    compressed_tokens = [t for i, t in enumerate(tokens) if i in keep_indices]
    
    return detokenize(compressed_tokens)
```

---

## Long Context Handling

### Hierarchical Context

For very long documents, use a hierarchical approach:

```python
class HierarchicalContext:
    def __init__(self, document: str):
        self.document = document
        self.sections = self.split_into_sections(document)
        self.section_summaries = self.create_summaries(self.sections)
    
    def get_context(self, query: str, detail_budget: int) -> str:
        # First, include all summaries
        context = "Document Overview:\n"
        for i, summary in enumerate(self.section_summaries):
            context += f"\nSection {i+1}: {summary}\n"
        
        # Then, add detailed content for most relevant sections
        relevant_sections = self.find_relevant_sections(query, top_k=2)
        
        context += "\n\nDetailed Sections:\n"
        for section_idx in relevant_sections:
            section_content = self.truncate(
                self.sections[section_idx], 
                detail_budget // len(relevant_sections)
            )
            context += f"\n[Section {section_idx+1}]\n{section_content}\n"
        
        return context
```

### Map-Reduce for Long Documents

Process long documents in chunks, then combine:

```python
def map_reduce_query(
    document: str,
    query: str,
    chunk_size: int = 4000
) -> str:
    # Map: Process each chunk
    chunks = split_into_chunks(document, chunk_size)
    chunk_answers = []
    
    for chunk in chunks:
        answer = llm.generate(
            f"Based on this text, answer the question if possible.\n\n"
            f"Text: {chunk}\n\n"
            f"Question: {query}\n\n"
            f"If the answer is not in this text, respond with 'NOT_FOUND'."
        )
        if answer != "NOT_FOUND":
            chunk_answers.append(answer)
    
    if not chunk_answers:
        return "Information not found in document."
    
    # Reduce: Combine answers
    combined = llm.generate(
        f"Combine these partial answers into a single coherent response.\n\n"
        f"Question: {query}\n\n"
        f"Partial answers:\n" + 
        "\n".join(f"- {a}" for a in chunk_answers)
    )
    
    return combined
```

### Sliding Window with Memory

```python
class SlidingWindowContext:
    def __init__(self, window_size: int = 4000, overlap: int = 500):
        self.window_size = window_size
        self.overlap = overlap
        self.memory = []  # Key facts extracted from previous windows
    
    def process_long_document(self, document: str, query: str) -> str:
        chunks = self.create_overlapping_chunks(document)
        
        for chunk in chunks:
            # Include memory in context
            context = f"Previous key facts:\n{self.format_memory()}\n\n"
            context += f"Current section:\n{chunk}"
            
            # Extract key facts from this chunk
            facts = self.extract_key_facts(chunk, query)
            self.memory.extend(facts)
            
            # Trim memory if too long
            self.memory = self.memory[-20:]  # Keep last 20 facts
        
        # Final answer using all accumulated knowledge
        return self.generate_final_answer(query)
```

---

## Conversation Context

### Conversation Summarization

```python
class ConversationManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages = []
        self.summary = ""
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.maybe_compress()
    
    def maybe_compress(self):
        total_tokens = sum(count_tokens(m["content"]) for m in self.messages)
        
        if total_tokens > self.max_tokens * 0.8:
            # Summarize older messages
            old_messages = self.messages[:-4]  # Keep last 4
            recent_messages = self.messages[-4:]
            
            summary_prompt = f"""
            Previous conversation summary: {self.summary}
            
            New messages to incorporate:
            {self.format_messages(old_messages)}
            
            Create an updated summary that captures all key information.
            """
            
            self.summary = llm.generate(summary_prompt)
            self.messages = recent_messages
    
    def get_context(self) -> list[dict]:
        context = []
        
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary}"
            })
        
        context.extend(self.messages)
        return context
```

### Selective History

```python
def select_relevant_history(
    messages: list[dict],
    current_query: str,
    max_messages: int = 10
) -> list[dict]:
    # Always include system message
    selected = [m for m in messages if m["role"] == "system"]
    
    # Score other messages by relevance to current query
    other_messages = [m for m in messages if m["role"] != "system"]
    query_embedding = embed(current_query)
    
    scored = []
    for msg in other_messages:
        embedding = embed(msg["content"])
        score = cosine_similarity(query_embedding, embedding)
        scored.append((msg, score))
    
    # Take top by relevance, but maintain chronological order
    scored.sort(key=lambda x: x[1], reverse=True)
    relevant = [m for m, _ in scored[:max_messages]]
    
    # Restore chronological order
    for msg in other_messages:
        if msg in relevant:
            selected.append(msg)
    
    return selected
```

---

## Production Patterns

### Dynamic Context Assembly

```python
class ContextAssembler:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = get_tokenizer(config["model"])
    
    def assemble(
        self,
        system_prompt: str,
        user_query: str,
        retrieved_docs: list[str],
        conversation_history: list[dict]
    ) -> list[dict]:
        budget = ContextBudget(
            max_tokens=self.config["max_context"],
            output_reserve=self.config["max_output"]
        )
        
        messages = []
        used_tokens = 0
        
        # 1. System prompt (required)
        system_tokens = count_tokens(system_prompt)
        messages.append({"role": "system", "content": system_prompt})
        used_tokens += system_tokens
        
        # 2. Retrieved context (high priority)
        retrieved_budget = budget.get_budget("retrieved")
        selected_docs = self.select_docs(
            retrieved_docs, user_query, retrieved_budget
        )
        
        if selected_docs:
            context_content = "Relevant information:\n\n" + "\n\n".join(selected_docs)
            messages.append({"role": "system", "content": context_content})
            used_tokens += count_tokens(context_content)
        
        # 3. Conversation history (fill remaining)
        history_budget = budget.available - used_tokens - count_tokens(user_query)
        selected_history = self.select_history(
            conversation_history, history_budget
        )
        messages.extend(selected_history)
        
        # 4. Current user message
        messages.append({"role": "user", "content": user_query})
        
        return messages
```

### Context Caching

```python
class ContextCache:
    """Cache expensive context computations."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.embedding_cache = TTLCache(maxsize=10000, ttl=ttl_seconds)
        self.summary_cache = TTLCache(maxsize=1000, ttl=ttl_seconds)
    
    def get_embedding(self, text: str) -> list[float]:
        key = hash(text)
        if key not in self.embedding_cache:
            self.embedding_cache[key] = embed(text)
        return self.embedding_cache[key]
    
    def get_summary(self, document_id: str, document: str) -> str:
        if document_id not in self.summary_cache:
            self.summary_cache[document_id] = summarize(document)
        return self.summary_cache[document_id]
```

---

## Interview Questions

### Q: How do you handle context window limits in a RAG system?

**Strong answer:**
Multiple strategies working together:

**1. Smart selection:**
- Retrieve more than needed (top 20)
- Rerank to get best 5
- Select by relevance within token budget

**2. Compression when needed:**
- Summarize long passages
- Extract relevant sentences only
- Use hierarchical summaries for very long docs

**3. Budget allocation:**
```python
# Example for 8K context
system: 500 tokens
retrieved: 4000 tokens (4-5 chunks)
history: 1500 tokens
user query: 500 tokens
output reserve: 1500 tokens
```

**4. Quality over quantity:**
- Better to have 3 highly relevant chunks than 10 mediocre ones
- The model can only attend to so much effectively

### Q: What is the "lost in the middle" problem and how do you address it?

**Strong answer:**
Research shows LLMs pay more attention to the beginning and end of context, with reduced attention to the middle.

**Implications:**
- Important context in the middle may be ignored
- Long context does not mean all of it is equally used

**Mitigations:**

1. **Strategic ordering:**
   - Put most important content first
   - Also include key info at the end
   - Avoid burying critical facts in the middle

2. **Limit context length:**
   - Less context = less middle to lose
   - Quality over quantity

3. **Redundancy for critical info:**
   - Repeat key facts in different positions
   - Summarize key points at the start

4. **Use models designed for long context:**
   - Gemini 1.5, Claude with better attention
   - But still verify with your specific task

---

## References

- Liu et al. "Lost in the Middle: How Language Models Use Long Contexts" (2023)
- Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios" (2023)
- Anthropic Long Context Guide: https://docs.anthropic.com/claude/docs/long-context-tips

---

*Previous: [Prompt Engineering](01-prompt-engineering.md) | Next: [System Prompts](03-system-prompts.md)*

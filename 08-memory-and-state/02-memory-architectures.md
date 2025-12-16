# Memory Architectures for LLM Applications

Memory is essential for LLM applications that need context beyond a single turn. This chapter covers memory patterns from simple caching to persistent long-term storage.

## Table of Contents

- [Memory Requirements](#memory-requirements)
- [Short-Term Memory](#short-term-memory)
- [Long-Term Memory](#long-term-memory)
- [Semantic Memory](#semantic-memory)
- [Memory Management](#memory-management)
- [Caching Strategies](#caching-strategies)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Memory Requirements

### Types of Memory in LLM Systems

| Memory Type | Scope | Persistence | Example |
|-------------|-------|-------------|---------|
| **Context Window** | Current request | None | Conversation so far |
| **Session Memory** | User session | Session-scoped | Chat history |
| **User Memory** | Across sessions | Persistent | Preferences, facts |
| **Semantic Memory** | All users | Persistent | Knowledge base |
| **Procedural Memory** | System-wide | Persistent | Learned patterns |

### Memory Challenges

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Context limit | Cannot fit all history | Summarization |
| Retrieval relevance | Wrong memories recalled | Better indexing |
| Staleness | Outdated information | TTL, freshness tracking |
| Privacy | Cross-user leakage | Strict isolation |
| Scale | Growing memory size | Pruning, archival |

---

## Short-Term Memory

### Conversation Buffer

```python
class ConversationBuffer:
    """Simple buffer that stores recent messages."""
    
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, message: dict):
        self.messages.append(message)
        
        # Trim if exceeds limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> list[dict]:
        return self.messages.copy()
    
    def clear(self):
        self.messages = []
```

### Token-Limited Buffer

```python
class TokenLimitedBuffer:
    """Buffer that respects token limits."""
    
    def __init__(self, max_tokens: int = 4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add(self, message: dict):
        self.messages.append(message)
        self._trim_to_limit()
    
    def _trim_to_limit(self):
        while self._count_tokens() > self.max_tokens and len(self.messages) > 1:
            # Remove oldest message (keep at least one)
            self.messages.pop(0)
    
    def _count_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            total += count_tokens(msg.get("content", ""))
        return total
```

### Sliding Window with Summary

```python
class SummarizingBuffer:
    """Keep recent messages + summary of older context."""
    
    def __init__(
        self,
        keep_recent: int = 10,
        summarize_after: int = 20
    ):
        self.keep_recent = keep_recent
        self.summarize_after = summarize_after
        self.messages = []
        self.summary = None
    
    def add(self, message: dict):
        self.messages.append(message)
        
        if len(self.messages) > self.summarize_after:
            self._summarize_old_messages()
    
    async def _summarize_old_messages(self):
        # Keep recent, summarize the rest
        old_messages = self.messages[:-self.keep_recent]
        self.messages = self.messages[-self.keep_recent:]
        
        # Create summary
        old_summary = self.summary or ""
        new_summary = await self._create_summary(old_summary, old_messages)
        self.summary = new_summary
    
    def get_context(self) -> list[dict]:
        context = []
        
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        context.extend(self.messages)
        return context
```

---

## Long-Term Memory

### Persistent Memory Store

```python
class LongTermMemory:
    """Store facts and preferences that persist across sessions."""
    
    def __init__(self, storage, vector_store):
        self.storage = storage  # Key-value store
        self.vector_store = vector_store  # For semantic retrieval
    
    async def store_fact(
        self,
        user_id: str,
        fact: str,
        category: str,
        confidence: float = 1.0
    ):
        fact_id = generate_id()
        
        # Store structured data
        await self.storage.set(f"fact:{user_id}:{fact_id}", {
            "fact": fact,
            "category": category,
            "confidence": confidence,
            "created_at": datetime.now(),
            "last_accessed": datetime.now()
        })
        
        # Index for retrieval
        embedding = await embed(fact)
        await self.vector_store.upsert(
            id=fact_id,
            vector=embedding,
            payload={
                "user_id": user_id,
                "fact": fact,
                "category": category
            }
        )
    
    async def recall(
        self,
        user_id: str,
        query: str,
        k: int = 5
    ) -> list[dict]:
        # Semantic search for relevant facts
        embedding = await embed(query)
        
        results = await self.vector_store.search(
            query_vector=embedding,
            top_k=k,
            filter={"user_id": user_id}
        )
        
        # Update access time
        for result in results:
            await self._update_access_time(user_id, result.id)
        
        return [r.payload for r in results]
```

### Memory Extraction

```python
class MemoryExtractor:
    """Extract memorable facts from conversations."""
    
    async def extract_facts(self, conversation: list[dict]) -> list[dict]:
        prompt = """
Analyze this conversation and extract facts worth remembering:
- User preferences
- Personal information shared
- Important dates or events
- Stated goals or intentions

Conversation:
{conversation}

Return as JSON array:
[
    {{"fact": "User prefers dark mode", "category": "preference", "confidence": 0.9}},
    ...
]
"""
        
        result = await self.llm.generate(
            prompt.format(conversation=self.format_conversation(conversation))
        )
        
        return json.loads(result)
    
    async def process_conversation_end(
        self,
        user_id: str,
        conversation: list[dict],
        memory: LongTermMemory
    ):
        # Extract facts
        facts = await self.extract_facts(conversation)
        
        # Store each fact
        for fact in facts:
            await memory.store_fact(
                user_id=user_id,
                fact=fact["fact"],
                category=fact["category"],
                confidence=fact["confidence"]
            )
```

---

## Semantic Memory

### Knowledge Graph Memory

```python
class KnowledgeGraphMemory:
    """Store facts as a knowledge graph for relationship queries."""
    
    def __init__(self, graph_db):
        self.graph = graph_db
    
    async def add_relationship(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object: str
    ):
        await self.graph.query("""
            MERGE (s:Entity {name: $subject, user_id: $user_id})
            MERGE (o:Entity {name: $object, user_id: $user_id})
            CREATE (s)-[r:RELATION {type: $predicate}]->(o)
        """, {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "user_id": user_id
        })
    
    async def query_relationships(
        self,
        user_id: str,
        entity: str
    ) -> list[dict]:
        result = await self.graph.query("""
            MATCH (s:Entity {name: $entity, user_id: $user_id})-[r]->(o)
            RETURN s.name as subject, type(r) as predicate, o.name as object
            UNION
            MATCH (s)-[r]->(o:Entity {name: $entity, user_id: $user_id})
            RETURN s.name as subject, type(r) as predicate, o.name as object
        """, {"entity": entity, "user_id": user_id})
        
        return result
```

### Episodic Memory

```python
class EpisodicMemory:
    """Store complete episodes (conversations, tasks) for learning."""
    
    def __init__(self, storage, vector_store):
        self.storage = storage
        self.vector_store = vector_store
    
    async def store_episode(
        self,
        user_id: str,
        episode: dict
    ):
        episode_id = generate_id()
        
        # Create summary for indexing
        summary = await self.summarize_episode(episode)
        
        # Store full episode
        await self.storage.set(f"episode:{episode_id}", {
            "user_id": user_id,
            "summary": summary,
            "content": episode,
            "outcome": episode.get("outcome"),
            "timestamp": datetime.now()
        })
        
        # Index summary
        embedding = await embed(summary)
        await self.vector_store.upsert(
            id=episode_id,
            vector=embedding,
            payload={
                "user_id": user_id,
                "summary": summary,
                "outcome": episode.get("outcome")
            }
        )
    
    async def recall_similar(
        self,
        user_id: str,
        situation: str,
        k: int = 3
    ) -> list[dict]:
        embedding = await embed(situation)
        
        results = await self.vector_store.search(
            query_vector=embedding,
            top_k=k,
            filter={"user_id": user_id}
        )
        
        episodes = []
        for result in results:
            full_episode = await self.storage.get(f"episode:{result.id}")
            episodes.append(full_episode)
        
        return episodes
```

---

## Memory Management

### Memory Pruning

```python
class MemoryPruner:
    """Remove stale or low-value memories."""
    
    def __init__(
        self,
        max_age_days: int = 365,
        min_access_count: int = 2,
        max_memories_per_user: int = 1000
    ):
        self.max_age_days = max_age_days
        self.min_access_count = min_access_count
        self.max_memories = max_memories_per_user
    
    async def prune(self, user_id: str, memory_store):
        # Get all memories for user
        memories = await memory_store.get_all(user_id)
        
        # Filter candidates for deletion
        to_delete = []
        
        for memory in memories:
            age = (datetime.now() - memory["created_at"]).days
            
            # Delete old, rarely accessed memories
            if age > self.max_age_days and memory["access_count"] < self.min_access_count:
                to_delete.append(memory["id"])
        
        # If still over limit, delete by score
        remaining = len(memories) - len(to_delete)
        if remaining > self.max_memories:
            scored = self.score_memories(
                [m for m in memories if m["id"] not in to_delete]
            )
            to_delete.extend([m["id"] for m in scored[self.max_memories:]])
        
        # Delete
        for memory_id in to_delete:
            await memory_store.delete(memory_id)
    
    def score_memories(self, memories: list[dict]) -> list[dict]:
        """Score by importance: recency * access_count * confidence."""
        for m in memories:
            age_days = (datetime.now() - m["created_at"]).days
            recency_score = 1 / (1 + age_days / 30)
            m["score"] = recency_score * m["access_count"] * m.get("confidence", 1.0)
        
        return sorted(memories, key=lambda m: m["score"], reverse=True)
```

---

## Caching Strategies

### Semantic Cache

```python
class SemanticCache:
    """Cache responses for semantically similar queries."""
    
    def __init__(self, vector_store, threshold: float = 0.95, ttl: int = 3600):
        self.vector_store = vector_store
        self.threshold = threshold
        self.ttl = ttl
    
    async def get(self, query: str) -> str | None:
        embedding = await embed(query)
        
        results = await self.vector_store.search(
            query_vector=embedding,
            top_k=1,
            filter={"expires_at": {"$gt": datetime.now()}}
        )
        
        if results and results[0].score > self.threshold:
            return results[0].payload["response"]
        
        return None
    
    async def set(self, query: str, response: str):
        embedding = await embed(query)
        
        await self.vector_store.upsert(
            id=generate_id(),
            vector=embedding,
            payload={
                "query": query,
                "response": response,
                "expires_at": datetime.now() + timedelta(seconds=self.ttl)
            }
        )
```

---

## Interview Questions

### Q: How do you implement memory for a conversational AI?

**Strong answer:**

"I implement memory at multiple levels:

**Short-term (session):**
- Conversation buffer with token limits
- Sliding window with summarization for long conversations
- Stored in Redis or session storage

**Long-term (persistent):**
- Extract facts from conversations (preferences, important info)
- Store with embeddings for semantic retrieval
- Recall relevant facts at conversation start

**Implementation:**
```python
# At conversation start
user_context = await long_term_memory.recall(user_id, initial_query)

# During conversation
conversation_memory.add(message)
context = user_context + conversation_memory.get_context()

# At conversation end
facts = await extract_facts(conversation)
await long_term_memory.store(user_id, facts)
```

**Key considerations:**
- Privacy: strict user isolation
- Relevance: only surface pertinent memories
- Freshness: prune stale information
- Scale: limit per-user memory growth"

### Q: How do you handle context that exceeds the token limit?

**Strong answer:**

"Several strategies depending on use case:

**Summarization:** Compress older messages into a summary. Keep recent messages verbatim.

**Selective retrieval:** Store all history externally, retrieve only relevant portions using embeddings.

**Hierarchical summarization:** Create summaries at different granularities. Recent: full text. Older: paragraph summary. Ancient: one-line summary.

**Example implementation:**
```python
if total_tokens > limit:
    # Keep recent 5 turns verbatim
    recent = messages[-10:]
    
    # Summarize older messages
    older = messages[:-10]
    summary = await summarize(older)
    
    # Combine
    context = [{'role': 'system', 'content': f'Previous context: {summary}'}] + recent
```

The key is preserving the information that matters for the current task while staying within limits."

---

## References

- MemGPT: https://arxiv.org/abs/2310.08560
- LangChain Memory: https://python.langchain.com/docs/modules/memory/

---

*Next: [Caching](02-caching.md)*

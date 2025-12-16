# Memory and State Management

LLM systems often need to maintain state across interactions. This chapter covers conversation memory, agent state, and persistence patterns for production systems.

## Table of Contents

- [Why Memory Matters](#why-memory-matters)
- [Memory Types](#memory-types)
- [Conversation Memory](#conversation-memory)
- [Agent State Management](#agent-state-management)
- [Persistence Patterns](#persistence-patterns)
- [Memory in Multi-Turn Applications](#memory-in-multi-turn-applications)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Memory Matters

### The Stateless Challenge

LLMs are stateless by default. Each API call is independent. To build useful applications, we must manage state ourselves.

| Application | State Needed |
|-------------|--------------|
| Chatbot | Conversation history |
| Agent | Task progress, tool results |
| RAG | Session context, user preferences |
| Assistant | Long-term user knowledge |

### Memory Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Full history | Complete context | Token limits, cost |
| Summarization | Compact, unlimited | Loses details |
| RAG on history | Relevant retrieval | Latency, complexity |
| Hybrid | Flexible | Implementation effort |

---

## Memory Types

### Short-Term Memory

Current conversation or task context:

```python
class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Trim if exceeds limit
        if len(self.messages) > self.max_messages:
            # Keep system message if present
            if self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
            else:
                self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> list[dict]:
        return self.messages.copy()
    
    def clear(self):
        self.messages = []
```

### Long-Term Memory

Persistent knowledge across sessions:

```python
class LongTermMemory:
    def __init__(self, vector_store, user_id: str):
        self.vector_store = vector_store
        self.user_id = user_id
    
    async def store(self, memory: str, metadata: dict = None):
        embedding = await embed(memory)
        await self.vector_store.upsert(
            id=generate_id(),
            vector=embedding,
            payload={
                "user_id": self.user_id,
                "content": memory,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )
    
    async def recall(self, query: str, top_k: int = 5) -> list[str]:
        query_embedding = await embed(query)
        results = await self.vector_store.search(
            query_vector=query_embedding,
            filter={"user_id": self.user_id},
            limit=top_k
        )
        return [r.payload["content"] for r in results]
```

### Working Memory

Active task state:

```python
@dataclass
class WorkingMemory:
    task_goal: str
    current_step: int
    completed_steps: list[str]
    pending_actions: list[str]
    tool_results: dict[str, any]
    scratchpad: str  # For agent reasoning
    
    def to_context(self) -> str:
        return f"""
Task: {self.task_goal}
Progress: Step {self.current_step}
Completed: {', '.join(self.completed_steps)}
Pending: {', '.join(self.pending_actions)}

Previous results:
{json.dumps(self.tool_results, indent=2)}

Notes:
{self.scratchpad}
"""
```

---

## Conversation Memory

### Buffer Memory

Keep last N messages:

```python
class BufferMemory:
    def __init__(self, max_messages: int = 10):
        self.buffer = deque(maxlen=max_messages)
    
    def add_user_message(self, content: str):
        self.buffer.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        self.buffer.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> list[dict]:
        return list(self.buffer)
```

### Summary Memory

Summarize older messages:

```python
class SummaryMemory:
    def __init__(self, max_recent: int = 4, llm_client = None):
        self.summary = ""
        self.recent_messages = []
        self.max_recent = max_recent
        self.llm = llm_client
    
    async def add_message(self, message: dict):
        self.recent_messages.append(message)
        
        if len(self.recent_messages) > self.max_recent * 2:
            await self.compress()
    
    async def compress(self):
        # Summarize older messages
        old_messages = self.recent_messages[:-self.max_recent]
        self.recent_messages = self.recent_messages[-self.max_recent:]
        
        summary_prompt = f"""
        Previous summary: {self.summary}
        
        New messages to incorporate:
        {self.format_messages(old_messages)}
        
        Create an updated summary capturing key information and context.
        Be concise but preserve important details.
        """
        
        self.summary = await self.llm.generate(summary_prompt)
    
    def get_context(self) -> list[dict]:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary}"
            })
        context.extend(self.recent_messages)
        return context
```

### Token-Based Memory

Manage by token count, not message count:

```python
class TokenMemory:
    def __init__(self, max_tokens: int = 4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_message(self, message: dict):
        self.messages.append(message)
        self.trim_to_fit()
    
    def trim_to_fit(self):
        while self.total_tokens() > self.max_tokens and len(self.messages) > 1:
            # Remove oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    self.messages.pop(i)
                    break
    
    def total_tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.messages)
```

---

## Agent State Management

### Checkpointing

Save and restore agent state:

```python
@dataclass
class AgentState:
    task_id: str
    messages: list[dict]
    tool_calls: list[dict]
    current_step: str
    iteration: int
    metadata: dict

class AgentCheckpointer:
    def __init__(self, storage):
        self.storage = storage
    
    async def save(self, state: AgentState):
        await self.storage.put(
            key=f"agent:{state.task_id}",
            value=state.to_dict()
        )
    
    async def load(self, task_id: str) -> AgentState:
        data = await self.storage.get(f"agent:{task_id}")
        if data:
            return AgentState.from_dict(data)
        return None
    
    async def list_checkpoints(self, task_id: str) -> list[str]:
        return await self.storage.list_versions(f"agent:{task_id}")
    
    async def rollback(self, task_id: str, version: str) -> AgentState:
        data = await self.storage.get_version(f"agent:{task_id}", version)
        return AgentState.from_dict(data)
```

### Event Sourcing for Agents

Record all actions for replay:

```python
@dataclass
class AgentEvent:
    timestamp: datetime
    event_type: str  # thought, action, observation, error
    content: dict

class EventSourcedAgent:
    def __init__(self, task_id: str, event_store):
        self.task_id = task_id
        self.event_store = event_store
        self.events = []
    
    async def record(self, event_type: str, content: dict):
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            content=content
        )
        self.events.append(event)
        await self.event_store.append(self.task_id, event)
    
    async def rebuild_state(self) -> AgentState:
        events = await self.event_store.get_all(self.task_id)
        
        state = AgentState(task_id=self.task_id)
        for event in events:
            state = self.apply_event(state, event)
        
        return state
    
    def apply_event(self, state: AgentState, event: AgentEvent) -> AgentState:
        if event.event_type == "thought":
            state.messages.append({"role": "assistant", "content": event.content["thought"]})
        elif event.event_type == "action":
            state.tool_calls.append(event.content)
        elif event.event_type == "observation":
            state.messages.append({"role": "tool", "content": event.content["result"]})
        return state
```

---

## Persistence Patterns

### Session Storage

```python
class SessionStore:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600 * 24  # 24 hours
    
    async def save_session(self, session_id: str, data: dict):
        await self.redis.setex(
            f"session:{session_id}",
            self.ttl,
            json.dumps(data)
        )
    
    async def get_session(self, session_id: str) -> dict:
        data = await self.redis.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return None
    
    async def extend_ttl(self, session_id: str):
        await self.redis.expire(f"session:{session_id}", self.ttl)
```

### Database-Backed Memory

```python
class DatabaseMemory:
    def __init__(self, db_pool):
        self.db = db_pool
    
    async def save_conversation(
        self,
        conversation_id: str,
        user_id: str,
        messages: list[dict]
    ):
        await self.db.execute("""
            INSERT INTO conversations (id, user_id, messages, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (id) DO UPDATE
            SET messages = $3, updated_at = NOW()
        """, conversation_id, user_id, json.dumps(messages))
    
    async def get_conversation(self, conversation_id: str) -> list[dict]:
        row = await self.db.fetchrow("""
            SELECT messages FROM conversations WHERE id = $1
        """, conversation_id)
        
        if row:
            return json.loads(row["messages"])
        return []
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10
    ) -> list[dict]:
        rows = await self.db.fetch("""
            SELECT id, messages, updated_at
            FROM conversations
            WHERE user_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
        """, user_id, limit)
        
        return [dict(row) for row in rows]
```

---

## Memory in Multi-Turn Applications

### Hybrid Memory Architecture

```python
class HybridMemory:
    def __init__(
        self,
        short_term_limit: int = 10,
        summary_llm = None,
        vector_store = None
    ):
        self.short_term = BufferMemory(short_term_limit)
        self.summary = SummaryMemory(llm_client=summary_llm)
        self.long_term = LongTermMemory(vector_store) if vector_store else None
    
    async def add_turn(self, user_msg: str, assistant_msg: str):
        # Add to short-term
        self.short_term.add_user_message(user_msg)
        self.short_term.add_assistant_message(assistant_msg)
        
        # Update summary
        await self.summary.add_message({"role": "user", "content": user_msg})
        await self.summary.add_message({"role": "assistant", "content": assistant_msg})
        
        # Extract and store important facts
        if self.long_term:
            facts = await self.extract_facts(user_msg, assistant_msg)
            for fact in facts:
                await self.long_term.store(fact)
    
    async def get_context(self, current_query: str) -> list[dict]:
        context = []
        
        # Add summary of older conversation
        if self.summary.summary:
            context.append({
                "role": "system",
                "content": f"Conversation history:\n{self.summary.summary}"
            })
        
        # Add recalled long-term memories if relevant
        if self.long_term:
            memories = await self.long_term.recall(current_query, top_k=3)
            if memories:
                context.append({
                    "role": "system",
                    "content": f"Relevant past context:\n" + "\n".join(memories)
                })
        
        # Add recent messages
        context.extend(self.short_term.get_messages())
        
        return context
```

### Entity Memory

Track entities mentioned in conversation:

```python
class EntityMemory:
    def __init__(self):
        self.entities = {}  # entity_name -> entity_info
    
    async def extract_and_store(self, message: str, llm):
        # Use LLM to extract entities
        extraction_prompt = f"""
        Extract named entities from this message.
        Return as JSON: {{"entities": [{{"name": "...", "type": "...", "info": "..."}}]}}
        
        Message: {message}
        """
        
        result = await llm.generate(extraction_prompt)
        entities = json.loads(result)["entities"]
        
        for entity in entities:
            name = entity["name"].lower()
            if name in self.entities:
                # Merge information
                self.entities[name]["info"] += f"; {entity['info']}"
            else:
                self.entities[name] = entity
    
    def get_context(self) -> str:
        if not self.entities:
            return ""
        
        lines = ["Known entities:"]
        for name, info in self.entities.items():
            lines.append(f"- {name} ({info['type']}): {info['info']}")
        
        return "\n".join(lines)
```

---

## Interview Questions

### Q: How do you handle conversation memory that exceeds context limits?

**Strong answer:**

"Context limits are a fundamental constraint. I use a tiered approach:

**Tier 1 - Recent messages**: Keep the last 4-6 turns verbatim. These are most likely relevant to the current exchange.

**Tier 2 - Summarization**: Older messages get summarized. I use the LLM to create a running summary that captures key points, decisions, and context without the verbosity.

**Tier 3 - Semantic retrieval**: For very long conversations or when needing specific past context, I embed messages and retrieve relevant ones based on the current query.

**Implementation:**
```python
context = []
# Always include recent messages
context.extend(recent_messages[-6:])
# Include summary of older conversation
if summary:
    context.insert(0, {'role': 'system', 'content': f'Prior context: {summary}'})
# Retrieve relevant past messages if needed
if needs_specific_recall:
    relevant = vector_search(current_query, past_messages)
    context.insert(1, format_recalled_context(relevant))
```

The key insight is that not all history is equally valuable. Recent context matters most, and intelligent compression preserves what matters while fitting within limits."

### Q: How would you implement memory for a long-running agent?

**Strong answer:**

"Long-running agents need durable state that survives failures. I implement this with checkpointing and event sourcing.

**Checkpointing**: After each significant step, I save the complete agent state: messages, tool results, current step, and scratchpad. If the agent fails, I restore from the last checkpoint and resume.

**Event sourcing**: I log every action (thoughts, tool calls, observations) as immutable events. This gives me full replay capability and auditability.

**Implementation:**
```python
class LongRunningAgent:
    async def step(self):
        # Record the thought
        thought = await self.think()
        await self.event_store.append('thought', thought)
        
        # Execute action
        action = await self.decide_action(thought)
        await self.event_store.append('action', action)
        
        result = await self.execute(action)
        await self.event_store.append('observation', result)
        
        # Checkpoint after each complete step
        await self.checkpoint()
```

**Benefits:**
- Resilience: Resume from any failure
- Debugging: Replay exact sequence of events
- Cost control: Do not repeat completed work
- Auditability: Full trace of agent behavior

This is essentially how systems like LangGraph and CrewAI implement durable execution for agents."

---

## References

- LangChain Memory: https://python.langchain.com/docs/modules/memory/
- LangGraph Checkpointing: https://langchain-ai.github.io/langgraph/
- Redis for Session Management: https://redis.io/docs/

---

*Next: [Caching Strategies](02-caching.md)*

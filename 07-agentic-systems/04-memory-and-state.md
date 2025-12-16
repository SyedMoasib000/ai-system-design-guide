# Agent Memory and State

Long-running agents need persistent memory across interactions. This chapter covers memory architectures specifically for agentic systems.

## Table of Contents

- [Memory Requirements for Agents](#memory-requirements-for-agents)
- [Working Memory](#working-memory)
- [Episodic Memory](#episodic-memory)
- [Semantic Memory](#semantic-memory)
- [Procedural Memory](#procedural-memory)
- [Memory Management](#memory-management)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Memory Requirements for Agents

### Memory Types for Agents

| Memory Type | Purpose | Persistence | Example |
|-------------|---------|-------------|---------|
| **Working** | Current task state | Session | Current goal, step, observations |
| **Episodic** | Past experiences | Long-term | Previous task attempts, outcomes |
| **Semantic** | Facts and knowledge | Long-term | User preferences, domain knowledge |
| **Procedural** | Learned skills | Long-term | Successful tool patterns |

### Agent State vs Conversation Memory

Agents need more than conversation history:

```python
@dataclass
class AgentState:
    # Task state
    goal: str
    current_step: int
    total_steps: int
    
    # Execution state
    messages: list[dict]
    tool_calls: list[dict]
    observations: list[dict]
    
    # Planning state
    plan: list[str]
    completed_steps: list[str]
    pending_steps: list[str]
    
    # Working memory
    scratchpad: str
    variables: dict
    
    # Metadata
    started_at: datetime
    last_updated: datetime
    iteration_count: int
    total_tokens_used: int
```

---

## Working Memory

### Scratchpad Pattern

```python
class AgentScratchpad:
    """
    Temporary working memory for current task.
    Included in prompt for agent reasoning.
    """
    
    def __init__(self):
        self.notes = []
        self.variables = {}
        self.intermediate_results = []
    
    def add_note(self, note: str):
        self.notes.append({
            "content": note,
            "timestamp": datetime.now()
        })
    
    def set_variable(self, name: str, value: any):
        self.variables[name] = value
    
    def add_result(self, step: str, result: any):
        self.intermediate_results.append({
            "step": step,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def to_prompt(self) -> str:
        sections = []
        
        if self.notes:
            sections.append("## Working Notes\n" + "\n".join(
                f"- {n['content']}" for n in self.notes
            ))
        
        if self.variables:
            sections.append("## Variables\n" + "\n".join(
                f"- {k}: {v}" for k, v in self.variables.items()
            ))
        
        if self.intermediate_results:
            sections.append("## Intermediate Results\n" + "\n".join(
                f"- {r['step']}: {r['result']}" for r in self.intermediate_results[-5:]
            ))
        
        return "\n\n".join(sections)
```

### Goal Stack

```python
class GoalStack:
    """
    Track hierarchical goals and subgoals.
    """
    
    def __init__(self):
        self.stack = []  # Stack of goals
    
    def push_goal(self, goal: str, parent: str = None):
        self.stack.append({
            "goal": goal,
            "parent": parent,
            "status": "active",
            "started_at": datetime.now()
        })
    
    def complete_goal(self, goal: str, result: any):
        for item in self.stack:
            if item["goal"] == goal:
                item["status"] = "completed"
                item["result"] = result
                item["completed_at"] = datetime.now()
                break
    
    def get_current_goal(self) -> str:
        for item in reversed(self.stack):
            if item["status"] == "active":
                return item["goal"]
        return None
    
    def to_context(self) -> str:
        lines = ["## Goal Hierarchy"]
        for item in self.stack:
            status = "✓" if item["status"] == "completed" else "→"
            indent = "  " * self.get_depth(item)
            lines.append(f"{indent}{status} {item['goal']}")
        return "\n".join(lines)
```

---

## Episodic Memory

### Experience Store

```python
class EpisodicMemory:
    """
    Store and retrieve past task experiences.
    Helps agent learn from history.
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    async def store_episode(self, episode: dict):
        # Create embedding of the episode summary
        summary = self.create_summary(episode)
        embedding = await embed(summary)
        
        await self.vector_store.upsert(
            id=episode["id"],
            vector=embedding,
            payload={
                "task": episode["task"],
                "outcome": episode["outcome"],
                "success": episode["success"],
                "steps_taken": episode["steps"],
                "lessons": episode.get("lessons", []),
                "timestamp": episode["timestamp"]
            }
        )
    
    async def recall_similar(self, current_task: str, k: int = 3) -> list[dict]:
        embedding = await embed(current_task)
        
        results = await self.vector_store.search(
            query_vector=embedding,
            limit=k,
            filter={"success": True}  # Prefer successful examples
        )
        
        return [r.payload for r in results]
    
    def create_summary(self, episode: dict) -> str:
        return f"""
Task: {episode['task']}
Outcome: {episode['outcome']}
Success: {episode['success']}
Key steps: {', '.join(episode['steps'][:5])}
"""
```

### Learning from Failures

```python
class FailureMemory:
    """
    Remember failures to avoid repeating them.
    """
    
    async def store_failure(
        self,
        task: str,
        action: dict,
        error: str,
        context: dict
    ):
        failure = {
            "task": task,
            "failed_action": action,
            "error": error,
            "context_summary": self.summarize_context(context),
            "timestamp": datetime.now()
        }
        
        await self.failure_store.save(failure)
    
    async def get_warnings(self, task: str, proposed_action: dict) -> list[str]:
        # Check if we have failed similarly before
        similar_failures = await self.find_similar_failures(task, proposed_action)
        
        warnings = []
        for failure in similar_failures:
            warnings.append(
                f"Warning: A similar action failed before. "
                f"Error was: {failure['error']}"
            )
        
        return warnings
```

---

## Semantic Memory

### Fact Store

```python
class SemanticMemory:
    """
    Store factual knowledge learned during execution.
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    async def learn_fact(
        self,
        fact: str,
        source: str,
        confidence: float,
        category: str
    ):
        # Extract structured information
        entities = await self.extract_entities(fact)
        
        fact_record = {
            "content": fact,
            "source": source,
            "confidence": confidence,
            "category": category,
            "entities": entities,
            "learned_at": datetime.now()
        }
        
        await self.storage.save(fact_record)
    
    async def recall_facts(
        self,
        query: str,
        category: str = None,
        min_confidence: float = 0.7
    ) -> list[dict]:
        # Semantic search for relevant facts
        results = await self.storage.search(
            query=query,
            filter={
                "confidence": {"$gte": min_confidence},
                **({"category": category} if category else {})
            }
        )
        
        return results
```

### User Preference Memory

```python
class UserPreferenceMemory:
    """
    Remember user preferences across sessions.
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    async def update_preference(
        self,
        user_id: str,
        preference_type: str,
        value: any,
        confidence: float
    ):
        key = f"{user_id}:{preference_type}"
        
        existing = await self.storage.get(key)
        if existing:
            # Update with moving average
            new_confidence = (existing["confidence"] + confidence) / 2
            await self.storage.update(key, {
                "value": value,
                "confidence": new_confidence,
                "updated_at": datetime.now()
            })
        else:
            await self.storage.set(key, {
                "value": value,
                "confidence": confidence,
                "created_at": datetime.now()
            })
    
    async def get_preferences(self, user_id: str) -> dict:
        all_prefs = await self.storage.get_by_prefix(f"{user_id}:")
        return {
            p["type"]: p["value"]
            for p in all_prefs
            if p["confidence"] > 0.6
        }
```

---

## Procedural Memory

### Successful Patterns

```python
class ProceduralMemory:
    """
    Remember successful action sequences (skills).
    """
    
    async def store_skill(
        self,
        skill_name: str,
        trigger: str,
        action_sequence: list[dict],
        success_rate: float
    ):
        skill = {
            "name": skill_name,
            "trigger": trigger,
            "actions": action_sequence,
            "success_rate": success_rate,
            "usage_count": 1,
            "created_at": datetime.now()
        }
        
        await self.skill_store.save(skill)
    
    async def find_skill(self, situation: str) -> dict | None:
        # Find applicable skill
        skills = await self.skill_store.search(situation)
        
        # Rank by success rate and recency
        ranked = sorted(
            skills,
            key=lambda s: s["success_rate"] * 0.7 + self.recency_score(s) * 0.3,
            reverse=True
        )
        
        return ranked[0] if ranked else None
    
    async def update_skill_performance(
        self,
        skill_name: str,
        success: bool
    ):
        skill = await self.skill_store.get(skill_name)
        
        # Update success rate with exponential moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * skill["success_rate"]
        
        await self.skill_store.update(skill_name, {
            "success_rate": new_rate,
            "usage_count": skill["usage_count"] + 1
        })
```

---

## Memory Management

### Memory Consolidation

```python
class MemoryConsolidator:
    """
    Periodically consolidate and summarize memories.
    """
    
    async def consolidate(self, agent_id: str):
        # Get recent episodic memories
        episodes = await self.episodic.get_recent(agent_id, days=7)
        
        # Extract patterns
        patterns = await self.extract_patterns(episodes)
        
        # Store as procedural memory (skills)
        for pattern in patterns:
            if pattern["frequency"] > 3 and pattern["success_rate"] > 0.8:
                await self.procedural.store_skill(
                    skill_name=pattern["name"],
                    trigger=pattern["trigger"],
                    action_sequence=pattern["actions"],
                    success_rate=pattern["success_rate"]
                )
        
        # Summarize and archive old episodes
        old_episodes = await self.episodic.get_older_than(agent_id, days=30)
        summary = await self.summarize_episodes(old_episodes)
        await self.archive.store(agent_id, summary)
        await self.episodic.delete(old_episodes)
```

### Memory Budget

```python
class MemoryBudget:
    """
    Manage memory within token limits.
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
    
    def allocate(self) -> dict:
        return {
            "working_memory": int(self.max_tokens * 0.4),
            "episodic_memory": int(self.max_tokens * 0.25),
            "semantic_memory": int(self.max_tokens * 0.2),
            "procedural_memory": int(self.max_tokens * 0.15)
        }
    
    def select_memories(
        self,
        available: dict[str, list],
        budget: dict[str, int]
    ) -> dict[str, list]:
        selected = {}
        
        for memory_type, items in available.items():
            type_budget = budget[memory_type]
            
            # Rank by relevance and recency
            ranked = self.rank_by_importance(items)
            
            # Select until budget exhausted
            selected[memory_type] = []
            tokens_used = 0
            for item in ranked:
                item_tokens = count_tokens(str(item))
                if tokens_used + item_tokens <= type_budget:
                    selected[memory_type].append(item)
                    tokens_used += item_tokens
        
        return selected
```

---

## Interview Questions

### Q: How do you design memory for a long-running agent?

**Strong answer:**

"I use a multi-tier memory architecture inspired by human memory:

**Working memory** holds the current task state: goal, steps taken, observations, scratchpad notes. This is always in context. I keep it concise to leave room for other content.

**Episodic memory** stores past task experiences. When facing a similar task, I retrieve relevant episodes to learn from past successes and avoid past failures. This is vector-indexed for semantic retrieval.

**Semantic memory** contains learned facts: user preferences, domain knowledge, tool capabilities. This persists across sessions.

**Procedural memory** captures successful action patterns (skills). If the agent has successfully completed similar tasks before, it can reuse that pattern rather than reasoning from scratch.

**Key implementation details:**
- Checkpointing for crash recovery
- Memory consolidation to convert episodes into skills
- Budget allocation to fit within context limits
- Recency and relevance scoring for retrieval

The goal is that the agent improves over time by learning from experience, not just operating statelessly."

### Q: How do you handle memory that exceeds the context window?

**Strong answer:**

"Several strategies:

**Summarization:** Compress older conversation turns into a summary. Keep recent turns verbatim, summarize older ones.

**Selective retrieval:** Do not include all memory in every prompt. Use semantic search to retrieve only relevant episodic and semantic memories for the current task.

**Hierarchical memory:** Store different granularities. Fine-grained recent events, summarized older events, highly compressed ancient events.

**Budget allocation:** Divide context into budgets. Working memory gets 40%, episodic 25%, semantic 20%, procedural 15%. Strict limits prevent any category from dominating.

**External persistence:** Store full memories externally (database, vector store). Only load what is needed for current context.

The key insight is that not all memory is equally valuable for every moment. Intelligent selection beats dumping everything into context."

---

## References

- Cognitive Architecture for Agents: https://arxiv.org/abs/2309.02427
- LangGraph Memory: https://langchain-ai.github.io/langgraph/

---

*Next: [Human-in-the-Loop](05-human-in-the-loop.md)*

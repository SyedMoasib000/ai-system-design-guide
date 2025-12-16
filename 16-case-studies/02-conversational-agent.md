# Case Study: Customer Support Conversational Agent

This case study walks through designing a production customer support agent for a B2B SaaS company.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Requirements Analysis](#requirements-analysis)
- [Architecture Design](#architecture-design)
- [Component Deep Dives](#component-deep-dives)
- [Reliability Patterns](#reliability-patterns)
- [Evaluation and Monitoring](#evaluation-and-monitoring)
- [Cost Analysis](#cost-analysis)
- [Lessons Learned](#lessons-learned)
- [Interview Walkthrough](#interview-walkthrough)

---

## Problem Statement

**Company:** B2B SaaS platform with 50K enterprise customers

**Current state:**
- 500K support tickets per month
- Average response time: 4 hours
- Customer satisfaction (CSAT): 72%
- Support team: 100 agents

**Goal:**
- Reduce response time to < 5 minutes for common queries
- Improve CSAT to > 85%
- Handle 60% of tickets without human intervention
- Maintain quality for escalated tickets

---

## Requirements Analysis

### Functional Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| Query understanding | Classify intent, extract entities | P0 |
| Knowledge retrieval | Search product docs, FAQs, past tickets | P0 |
| Account context | Access user's subscription, history | P0 |
| Response generation | Natural, accurate, helpful responses | P0 |
| Conversation memory | Multi-turn context | P0 |
| Action execution | Create tickets, trigger workflows | P1 |
| Human escalation | Seamless handoff when needed | P0 |
| Billing inquiries | Handle sensitive financial data | P1 |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| Latency (TTFT) | < 1s | User expectation for chat |
| Latency (full) | < 5s | Maintain engagement |
| Availability | 99.9% | Business-critical |
| Accuracy | > 95% | Customer trust |
| Escalation rate | < 40% | Cost efficiency |
| CSAT | > 85% | Business goal |

### Security Requirements

- No PII in logs
- Tenant isolation (customers only see their data)
- Audit trail for all actions
- SOC 2 compliance

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CUSTOMER SUPPORT AGENT                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   Web/App   │────▶│   Gateway   │────▶│    Auth     │        │
│  │   Client    │     │             │     │  + Tenant   │        │
│  └─────────────┘     └──────┬──────┘     └─────────────┘        │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   ORCHESTRATION LAYER                     │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Intent        Query          Response    Workflow │  │   │
│  │  │  Classifier → Router →        Generator → Engine   │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  Knowledge  │     │   Account   │     │   Action    │        │
│  │    Base     │     │   Context   │     │   Tools     │        │
│  │   (RAG)     │     │   Service   │     │             │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Conversation Flow

```
User Message
    │
    ▼
┌─────────────────┐
│ Intent Classify │─── billing, technical, account, general, escalation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Routing   │─── Which knowledge sources? Which tools?
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
┌───────┐ ┌───────┐ ┌──────────┐
│  RAG  │ │Account│ │ Actions  │
│ Query │ │Context│ │ (if any) │
└───┬───┘ └───┬───┘ └────┬─────┘
    │         │          │
    └────┬────┴──────────┘
         │
         ▼
┌─────────────────┐
│    Generate     │
│    Response     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Safety Check   │─── PII, harmful, off-topic
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Confidence     │─── Low confidence? Escalate
│    Check        │
└────────┬────────┘
         │
         ▼
    Response / Escalation
```

---

## Component Deep Dives

### Intent Classification

```python
class IntentClassifier:
    INTENTS = [
        "billing",
        "technical_issue",
        "account_management",
        "feature_request",
        "general_question",
        "escalation_request"
    ]
    
    async def classify(self, message: str, history: list[dict]) -> dict:
        prompt = f"""
Classify this customer support message into one of these intents:
{', '.join(self.INTENTS)}

Conversation history:
{self.format_history(history[-3:])}

Current message: {message}

Return JSON: {{"intent": "...", "confidence": 0.0-1.0, "entities": {{}}}}
"""
        
        result = await self.llm.generate(prompt, response_format="json")
        return json.loads(result)
```

### Knowledge Base (RAG Pipeline)

```python
class SupportKnowledgeBase:
    def __init__(self):
        self.sources = {
            "docs": ProductDocsIndex(),
            "faqs": FAQIndex(),
            "tickets": ResolvedTicketsIndex(),
            "release_notes": ReleaseNotesIndex()
        }
    
    async def retrieve(
        self,
        query: str,
        intent: str,
        tenant_id: str
    ) -> list[dict]:
        # Select sources based on intent
        if intent == "billing":
            sources = ["faqs", "docs"]
        elif intent == "technical_issue":
            sources = ["docs", "tickets", "release_notes"]
        else:
            sources = ["faqs", "docs"]
        
        # Parallel retrieval from selected sources
        results = await asyncio.gather(*[
            self.sources[s].search(query, tenant_id=tenant_id)
            for s in sources
        ])
        
        # Merge and rerank
        all_results = [r for source_results in results for r in source_results]
        reranked = await self.reranker.rerank(query, all_results)
        
        return reranked[:5]
```

### Account Context Service

```python
class AccountContextService:
    async def get_context(self, user_id: str, tenant_id: str) -> dict:
        # Fetch relevant account information
        account = await self.account_db.get(tenant_id)
        user = await self.user_db.get(user_id)
        
        # Recent activity
        recent_tickets = await self.tickets_db.get_recent(user_id, limit=5)
        recent_actions = await self.activity_log.get_recent(user_id, limit=10)
        
        # Subscription details
        subscription = await self.billing.get_subscription(tenant_id)
        
        return {
            "account_name": account["name"],
            "plan": subscription["plan"],
            "user_role": user["role"],
            "recent_tickets": self.summarize_tickets(recent_tickets),
            "account_age_days": (datetime.now() - account["created_at"]).days,
            "is_trial": subscription["is_trial"]
        }
```

### Response Generation

```python
class ResponseGenerator:
    async def generate(
        self,
        query: str,
        context: list[dict],
        account_context: dict,
        history: list[dict]
    ) -> dict:
        system_prompt = f"""
You are a helpful customer support agent for [Company Name].

Account context:
- Customer: {account_context['account_name']}
- Plan: {account_context['plan']}
- Role: {account_context['user_role']}

Guidelines:
- Be helpful, professional, and empathetic
- Use the provided context to answer accurately
- If you are unsure, say so and offer to escalate
- Never make up information not in the context
- For billing issues, be extra careful with numbers
- Always end with an offer to help further
"""
        
        context_str = self.format_context(context)
        
        response = await self.llm.generate(
            system=system_prompt,
            messages=history + [
                {"role": "system", "content": f"Relevant information:\n{context_str}"},
                {"role": "user", "content": query}
            ],
            temperature=0.3  # Lower for accuracy
        )
        
        # Extract confidence
        confidence = await self.assess_confidence(query, response, context)
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [c["source"] for c in context]
        }
```

---

## Reliability Patterns

### Confidence-Based Escalation

```python
class EscalationHandler:
    def __init__(self, confidence_threshold: float = 0.7):
        self.threshold = confidence_threshold
    
    async def check_escalation(
        self,
        response: dict,
        intent: str,
        user_request: str
    ) -> dict:
        should_escalate = False
        reason = None
        
        # Low confidence
        if response["confidence"] < self.threshold:
            should_escalate = True
            reason = "low_confidence"
        
        # Explicit escalation request
        if intent == "escalation_request":
            should_escalate = True
            reason = "user_requested"
        
        # Sensitive topics
        if await self.is_sensitive(user_request):
            should_escalate = True
            reason = "sensitive_topic"
        
        if should_escalate:
            return await self.create_escalation(response, reason)
        
        return {"escalate": False, "response": response}
    
    async def is_sensitive(self, message: str) -> bool:
        sensitive_keywords = [
            "legal", "lawsuit", "lawyer",
            "refund", "cancel subscription",
            "competitor", "data breach"
        ]
        return any(kw in message.lower() for kw in sensitive_keywords)
```

### Multi-Turn Memory

```python
class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.redis = Redis()
    
    async def get_history(self, session_id: str) -> list[dict]:
        key = f"conversation:{session_id}"
        history = await self.redis.get(key)
        if history:
            return json.loads(history)
        return []
    
    async def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str
    ):
        history = await self.get_history(session_id)
        
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        
        # Trim to max turns
        if len(history) > self.max_turns * 2:
            history = history[-(self.max_turns * 2):]
        
        await self.redis.setex(
            f"conversation:{session_id}",
            3600,  # 1 hour TTL
            json.dumps(history)
        )
```

---

## Evaluation and Monitoring

### Quality Metrics

```python
class QualityMonitor:
    def __init__(self, sample_rate: float = 0.05):
        self.sample_rate = sample_rate
        self.judge = LLMJudge()
    
    async def evaluate(self, conversation: dict):
        if random.random() > self.sample_rate:
            return
        
        scores = await self.judge.evaluate(
            query=conversation["user_message"],
            response=conversation["assistant_message"],
            context=conversation["context"],
            criteria={
                "relevance": "Does the response address the user's question?",
                "accuracy": "Is the information correct based on the context?",
                "helpfulness": "Would this response help the user?",
                "tone": "Is the tone professional and empathetic?"
            }
        )
        
        # Record metrics
        for criterion, score in scores.items():
            metrics.record(f"quality_{criterion}", score)
```

### Dashboard Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Latency (TTFT) | < 1s | 0.8s |
| Latency (full) | < 5s | 3.2s |
| Accuracy | > 95% | 94.3% |
| Escalation rate | < 40% | 38% |
| CSAT | > 85% | 87% |
| Resolution rate | > 60% | 62% |

---

## Cost Analysis

### Per-Conversation Cost Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| Intent classification | $0.001 | GPT-4o-mini |
| RAG retrieval | $0.002 | Embeddings + search |
| Reranking | $0.003 | Cross-encoder |
| Response generation | $0.015 | Claude 3.5 Sonnet |
| Quality sampling | $0.001 | 5% sample rate |
| **Total** | **~$0.022** | Per conversation |

### Monthly Cost Projection

| Item | Calculation | Cost |
|------|-------------|------|
| Conversations | 500K × $0.022 | $11,000 |
| Infrastructure | Fixed | $2,000 |
| Human escalations | 190K × $5 (human cost) | $950,000 |
| **Total** | | $963,000 |
| **Savings vs all-human** | 500K × $5 - $963K | $1.5M/year |

---

## Lessons Learned

### What Worked

1. **Intent-based routing** reduced latency by focusing retrieval on relevant sources
2. **Confidence-based escalation** maintained quality while reducing human load
3. **Account context** made responses more personalized and accurate
4. **Lower temperature (0.3)** improved consistency for support responses

### What Did Not Work Initially

1. **Single model for everything** - routing to different models for different tasks improved quality
2. **Too high escalation threshold** - started at 0.9 confidence, causing too many escalations
3. **Full conversation history** - exceeded context limits, switched to summarization

### Recommendations

1. Start with high escalation rate and lower gradually as confidence improves
2. Monitor CSAT by escalation reason to identify weak areas
3. Retrain embeddings on support-specific vocabulary
4. Build feedback loop: agents tag escalated conversations for training data

---

## Interview Walkthrough

**Interviewer:** "Design an AI customer support system for a SaaS company."

**Strong response pattern:**

1. **Clarify requirements** (2 min)
   - "What's the ticket volume? What channels? What's the current CSAT?"

2. **State constraints explicitly**
   - "Key constraints: accuracy over speed, seamless escalation, tenant isolation"

3. **High-level architecture** (3 min)
   - Draw the flow: intent → routing → RAG → generation → safety → response/escalation

4. **Deep dive on critical component** (5 min)
   - "Let me detail the confidence-based escalation..."

5. **Address reliability** (3 min)
   - "For reliability, I would use self-consistency for billing queries, multi-provider fallback"

6. **Metrics and monitoring** (2 min)
   - "Key metrics: CSAT, resolution rate, escalation rate, accuracy sampling"

7. **Cost consideration** (1 min)
   - "At 500K conversations/month, cost per conversation matters. Model routing helps."

---

## References

- Anthropic Customer Support Best Practices: https://docs.anthropic.com/claude/docs/customer-service
- LangChain Conversational Agents: https://python.langchain.com/docs/use_cases/chatbots

---

*Next: [Code Assistant Case Study](03-code-assistant.md)*

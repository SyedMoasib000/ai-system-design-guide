# Human-in-the-Loop Patterns

Effective AI systems know when to involve humans. This chapter covers patterns for human oversight, escalation, and collaboration.

## Table of Contents

- [Why Human-in-the-Loop](#why-human-in-the-loop)
- [Escalation Patterns](#escalation-patterns)
- [Approval Workflows](#approval-workflows)
- [Feedback Integration](#feedback-integration)
- [Graceful Handoff](#graceful-handoff)
- [Monitoring and Intervention](#monitoring-and-intervention)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Human-in-the-Loop

### When AI Alone Is Not Enough

| Scenario | Why Human Needed |
|----------|------------------|
| Low confidence | AI uncertain, human judgment needed |
| High stakes | Consequences too severe for AI-only |
| Edge cases | Outside training distribution |
| Policy exceptions | Requires discretion |
| Learning opportunity | Human decision improves future AI |
| Legal/compliance | Human accountability required |

### The Human-AI Spectrum

```
Fully Automated ◄─────────────────────────────────► Fully Manual
      │                    │                              │
      │                    │                              │
   AI decides          AI proposes,                   Human does
   and acts           human approves                  everything
      │                    │                              │
  Low stakes          Medium stakes                High stakes
  High confidence     Mixed confidence            Low confidence
```

---

## Escalation Patterns

### Confidence-Based Escalation

```python
class ConfidenceEscalator:
    def __init__(
        self,
        high_threshold: float = 0.9,
        low_threshold: float = 0.6
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    async def process(self, query: str, response: str, confidence: float) -> dict:
        if confidence >= self.high_threshold:
            # High confidence: auto-respond
            return {
                "action": "respond",
                "response": response,
                "escalated": False
            }
        
        elif confidence >= self.low_threshold:
            # Medium confidence: respond but flag for review
            await self.queue_for_review(query, response, confidence)
            return {
                "action": "respond",
                "response": response,
                "escalated": False,
                "flagged_for_review": True
            }
        
        else:
            # Low confidence: escalate before responding
            return {
                "action": "escalate",
                "escalated": True,
                "message": "I want to make sure I give you accurate information. Let me connect you with a specialist.",
                "escalation_reason": "low_confidence"
            }
```

### Topic-Based Escalation

```python
class TopicEscalator:
    ALWAYS_ESCALATE = [
        "legal_advice",
        "medical_diagnosis",
        "financial_advice",
        "safety_critical",
        "harassment_report"
    ]
    
    ESCALATE_IF_COMPLEX = [
        "billing_dispute",
        "account_security",
        "refund_request"
    ]
    
    async def check(self, query: str, intent: str, complexity: str) -> dict:
        if intent in self.ALWAYS_ESCALATE:
            return {
                "escalate": True,
                "reason": f"topic_{intent}",
                "priority": "high"
            }
        
        if intent in self.ESCALATE_IF_COMPLEX and complexity == "high":
            return {
                "escalate": True,
                "reason": f"complex_{intent}",
                "priority": "medium"
            }
        
        return {"escalate": False}
```

### Cumulative Frustration Detection

```python
class FrustrationDetector:
    """
    Detect when user is frustrated and escalate proactively.
    """
    
    def __init__(self):
        self.indicators = {
            "repeated_questions": 0,
            "negative_sentiment": 0,
            "explicit_frustration": False,
            "caps_lock": 0,
            "short_responses": 0
        }
    
    async def analyze(self, message: str, history: list[dict]) -> dict:
        # Check for repeated similar questions
        if self.is_repeat_question(message, history):
            self.indicators["repeated_questions"] += 1
        
        # Sentiment analysis
        sentiment = await self.analyze_sentiment(message)
        if sentiment < -0.5:
            self.indicators["negative_sentiment"] += 1
        
        # Explicit frustration markers
        frustration_phrases = ["this is frustrating", "not helpful", "talk to human", "useless"]
        if any(phrase in message.lower() for phrase in frustration_phrases):
            self.indicators["explicit_frustration"] = True
        
        # Calculate frustration score
        score = (
            self.indicators["repeated_questions"] * 0.3 +
            self.indicators["negative_sentiment"] * 0.2 +
            (1.0 if self.indicators["explicit_frustration"] else 0) * 0.5
        )
        
        if score > 0.7:
            return {
                "escalate": True,
                "reason": "user_frustration",
                "score": score
            }
        
        return {"escalate": False, "score": score}
```

---

## Approval Workflows

### Pre-Action Approval

```python
class ApprovalWorkflow:
    """
    Require human approval before executing certain actions.
    """
    
    REQUIRES_APPROVAL = {
        "delete_data": "high",
        "send_email": "medium",
        "modify_billing": "high",
        "external_api_call": "low"
    }
    
    async def request_approval(
        self,
        action: dict,
        context: dict,
        requester: str
    ) -> ApprovalRequest:
        severity = self.REQUIRES_APPROVAL.get(action["type"], "low")
        
        request = ApprovalRequest(
            id=generate_id(),
            action=action,
            context=context,
            requester=requester,
            severity=severity,
            requested_at=datetime.now(),
            expires_at=datetime.now() + self.get_timeout(severity)
        )
        
        await self.store_request(request)
        await self.notify_approvers(request, severity)
        
        return request
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout: int = 3600
    ) -> ApprovalResult:
        try:
            result = await asyncio.wait_for(
                self.poll_for_decision(request_id),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return ApprovalResult(
                approved=False,
                reason="timeout",
                request_id=request_id
            )
    
    async def execute_with_approval(
        self,
        action: dict,
        context: dict,
        requester: str
    ) -> dict:
        # Check if approval needed
        if action["type"] not in self.REQUIRES_APPROVAL:
            return await self.execute(action)
        
        # Request approval
        request = await self.request_approval(action, context, requester)
        
        # Wait for decision
        result = await self.wait_for_approval(request.id)
        
        if result.approved:
            return await self.execute(action)
        else:
            return {
                "status": "rejected",
                "reason": result.reason,
                "approver": result.approver
            }
```

### Batch Review

```python
class BatchReviewWorkflow:
    """
    Queue items for batch human review.
    """
    
    def __init__(self, batch_size: int = 50):
        self.queue = []
        self.batch_size = batch_size
    
    async def add_for_review(self, item: dict):
        self.queue.append({
            "item": item,
            "added_at": datetime.now(),
            "status": "pending"
        })
        
        if len(self.queue) >= self.batch_size:
            await self.notify_reviewers()
    
    async def get_batch(self, reviewer_id: str) -> list[dict]:
        # Assign batch to reviewer
        batch = self.queue[:self.batch_size]
        for item in batch:
            item["assigned_to"] = reviewer_id
            item["assigned_at"] = datetime.now()
        
        return batch
    
    async def submit_reviews(
        self,
        reviewer_id: str,
        reviews: list[dict]
    ):
        for review in reviews:
            # Update item with review
            item = self.find_item(review["item_id"])
            item["review"] = review
            item["reviewed_by"] = reviewer_id
            item["reviewed_at"] = datetime.now()
            
            # Apply decision
            await self.apply_decision(item)
            
            # Train on feedback
            await self.feedback_loop(item)
```

---

## Feedback Integration

### Active Learning Loop

```python
class ActiveLearningLoop:
    """
    Use human decisions to improve AI models.
    """
    
    def __init__(self):
        self.feedback_store = FeedbackStore()
        self.model_trainer = ModelTrainer()
    
    async def record_feedback(
        self,
        query: str,
        ai_response: str,
        human_action: str,
        human_response: str = None
    ):
        feedback = {
            "query": query,
            "ai_response": ai_response,
            "human_action": human_action,  # approved, edited, rejected
            "human_response": human_response,
            "timestamp": datetime.now()
        }
        
        await self.feedback_store.save(feedback)
        
        # If correction, add to training queue
        if human_action in ["edited", "rejected"]:
            await self.add_to_training_queue(feedback)
    
    async def add_to_training_queue(self, feedback: dict):
        training_example = {
            "input": feedback["query"],
            "preferred_output": feedback["human_response"] or feedback["ai_response"],
            "rejected_output": feedback["ai_response"] if feedback["human_action"] == "rejected" else None
        }
        
        await self.model_trainer.queue_example(training_example)
    
    async def trigger_retraining(self):
        # Triggered when enough examples accumulated
        examples = await self.model_trainer.get_queued_examples()
        if len(examples) >= 1000:
            await self.model_trainer.fine_tune(examples)
```

### Quality Metrics from Human Feedback

```python
class HumanFeedbackMetrics:
    def compute_metrics(self, feedback_data: list[dict]) -> dict:
        total = len(feedback_data)
        
        approved = sum(1 for f in feedback_data if f["action"] == "approved")
        edited = sum(1 for f in feedback_data if f["action"] == "edited")
        rejected = sum(1 for f in feedback_data if f["action"] == "rejected")
        
        return {
            "approval_rate": approved / total,
            "edit_rate": edited / total,
            "rejection_rate": rejected / total,
            "auto_acceptable_rate": approved / total,  # Could skip human
            "needs_improvement_rate": (edited + rejected) / total
        }
```

---

## Graceful Handoff

### Warm Handoff Pattern

```python
class WarmHandoff:
    """
    Transfer to human with full context.
    """
    
    async def handoff(
        self,
        conversation: list[dict],
        user_info: dict,
        reason: str
    ) -> HandoffResult:
        # Summarize conversation for human
        summary = await self.summarize_conversation(conversation)
        
        # Get recommended actions
        recommendations = await self.get_recommendations(conversation)
        
        # Find available agent
        agent = await self.find_available_agent(
            skill_required=self.get_required_skill(reason),
            priority=self.get_priority(reason)
        )
        
        # Create handoff package
        package = {
            "conversation_summary": summary,
            "full_conversation": conversation,
            "user_info": user_info,
            "handoff_reason": reason,
            "ai_recommendations": recommendations,
            "sentiment": await self.analyze_sentiment(conversation),
            "key_entities": await self.extract_entities(conversation)
        }
        
        # Notify user
        user_message = f"""
I'm connecting you with {agent.name} who can better help with this. 
I've shared our conversation so you won't need to repeat yourself.
One moment please...
"""
        
        await self.send_to_user(user_message)
        
        # Connect to agent
        await self.connect_to_agent(agent, package)
        
        return HandoffResult(
            success=True,
            agent=agent,
            wait_time=agent.estimated_wait
        )
```

### Seamless Transition

```python
class SeamlessTransition:
    """
    User should not notice when transitioning to human.
    """
    
    async def transition(self, session_id: str, to_human: bool):
        if to_human:
            # Keep same UI, just change backend
            await self.notify_agent_pool(session_id)
            
            # While waiting for human, AI can still help
            await self.send_message(
                session_id,
                "I'm looking into this further. One moment..."
            )
        else:
            # Transitioning back to AI
            await self.update_ai_context(session_id)
```

---

## Monitoring and Intervention

### Real-Time Monitoring

```python
class ConversationMonitor:
    """
    Monitor ongoing AI conversations for intervention needs.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            "turns_without_resolution": 5,
            "sentiment_score": -0.7,
            "repeated_intent": 3
        }
    
    async def monitor(self, conversation: list[dict]) -> list[Alert]:
        alerts = []
        
        # Check conversation length
        if len(conversation) > self.alert_thresholds["turns_without_resolution"] * 2:
            alerts.append(Alert(
                type="long_conversation",
                severity="medium",
                message="Conversation exceeds expected length"
            ))
        
        # Check sentiment trend
        recent_sentiment = await self.analyze_recent_sentiment(conversation[-4:])
        if recent_sentiment < self.alert_thresholds["sentiment_score"]:
            alerts.append(Alert(
                type="negative_sentiment",
                severity="high",
                message="User sentiment is negative"
            ))
        
        # Check for repeated intents
        intents = [m.get("intent") for m in conversation if m.get("intent")]
        if self.has_repetition(intents):
            alerts.append(Alert(
                type="repeated_intent",
                severity="high",
                message="User repeating same request"
            ))
        
        return alerts
```

### Supervisor Override

```python
class SupervisorOverride:
    """
    Allow supervisors to intervene in AI conversations.
    """
    
    async def take_over(self, supervisor_id: str, session_id: str):
        # Pause AI
        await self.pause_ai(session_id)
        
        # Transfer control
        await self.transfer_control(session_id, supervisor_id)
        
        # Log intervention
        await self.log_intervention(supervisor_id, session_id, "takeover")
    
    async def inject_response(
        self,
        supervisor_id: str,
        session_id: str,
        response: str
    ):
        # Replace AI response with supervisor response
        await self.send_as_ai(session_id, response)
        
        # Log
        await self.log_intervention(supervisor_id, session_id, "inject", response)
```

---

## Interview Questions

### Q: When should an AI system escalate to a human?

**Strong answer:**

"I use multiple triggers for escalation:

**Confidence-based:** If the AI's confidence falls below a threshold (typically 0.6-0.7), escalate. The model knows what it does not know.

**Topic-based:** Certain topics always require human judgment: legal, medical, safety-critical, or sensitive personal situations.

**User-triggered:** If the user explicitly asks for a human or shows frustration (repeated questions, negative sentiment, explicit complaints).

**Policy-based:** Some actions require human approval regardless of AI confidence: financial transactions, data deletion, external communications.

**Error accumulation:** If the AI has had multiple failures in a conversation (tool errors, replannings), something is wrong.

The escalation message matters: 'I want to make sure you get the best help. Let me connect you with a specialist.' This maintains trust rather than admitting failure.

I track escalation rate as a key metric. Too high means the AI needs improvement. Too low might mean the threshold is wrong or users are not getting help when they need it."

### Q: How do you design the handoff from AI to human?

**Strong answer:**

"A good handoff preserves context and sets the human up for success.

**Warm handoff elements:**
1. Summary of conversation (what was discussed, what was tried)
2. User information and history
3. AI's assessment and recommendations
4. Sentiment analysis (is user frustrated?)
5. Key entities extracted (order numbers, issues mentioned)

**User experience:**
- Seamless transition, same interface
- Do not make user repeat themselves
- Set expectations ('One moment while I connect you')
- If wait time, provide estimate

**Agent experience:**
- All context visible before accepting
- AI recommendations as starting point
- Easy way to correct AI if it was wrong (training data)

The handoff should feel like the human was listening all along. No cold starts, no repeated explanations."

---

## References

- Anthropic Human-AI Interaction: https://docs.anthropic.com/
- LangGraph Human-in-the-Loop: https://langchain-ai.github.io/langgraph/

---

*Next: [Agent Evaluation](06-evaluation-and-monitoring.md)*

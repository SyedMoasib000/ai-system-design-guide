# Case Study: Content Moderation at Scale

This case study covers designing an AI-powered content moderation system for a social platform handling millions of posts daily.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Requirements Analysis](#requirements-analysis)
- [Architecture Design](#architecture-design)
- [Classification Pipeline](#classification-pipeline)
- [Human-in-the-Loop](#human-in-the-loop)
- [Adversarial Robustness](#adversarial-robustness)
- [Results and Metrics](#results-and-metrics)
- [Interview Walkthrough](#interview-walkthrough)

---

## Problem Statement

**Company:** Social media platform with 50M daily active users

**Current state:**
- 10M posts per day
- 500 human moderators
- Average review time: 4 hours
- False positive rate: 15%
- Harmful content reaching users: 2%

**Goals:**
- Reduce harmful content exposure to < 0.1%
- Review priority content in < 15 minutes
- Reduce false positive rate to < 5%
- Scale without linear moderator growth

---

## Requirements Analysis

### Content Categories

| Category | Severity | Action | Latency |
|----------|----------|--------|---------|
| CSAM | Critical | Block + Report | Immediate |
| Violence/Gore | High | Block + Review | < 1 min |
| Hate speech | High | Block + Review | < 5 min |
| Harassment | Medium | Review + Warn | < 15 min |
| Spam | Medium | Deprioritize | < 1 hour |
| Misinformation | Medium | Label + Review | < 1 hour |
| Adult content | Low | Age-gate | < 1 hour |

### Accuracy Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall (harmful) | > 99% | Minimize harm exposure |
| Precision | > 95% | Minimize false positives |
| Latency (critical) | < 1 min | Prevent spread |
| Latency (standard) | < 15 min | Balance resources |

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONTENT MODERATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │   Content   │                                                │
│  │   Ingestion │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   TIER 1: FAST FILTERS                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
│  │  │  Hash    │  │ Keyword  │  │  Known   │              │    │
│  │  │ Matching │  │ Blocklist│  │ Patterns │              │    │
│  │  └──────────┘  └──────────┘  └──────────┘              │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │ Blocked           │ Pass              │ Elevated      │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────────────────────────────┐    │
│  │   Block +   │    │          TIER 2: ML MODELS          │    │
│  │   Report    │    │  ┌────────┐  ┌────────┐  ┌────────┐│    │
│  └─────────────┘    │  │ Vision │  │  Text  │  │ Multi- ││    │
│                     │  │ Model  │  │ Model  │  │ modal  ││    │
│                     │  └────────┘  └────────┘  └────────┘│    │
│                     └──────────────────┬──────────────────┘    │
│                                        │                        │
│         ┌──────────────────────────────┼──────────────────┐    │
│         │ High Confidence              │ Low Confidence   │    │
│         ▼                              ▼                   │    │
│  ┌─────────────┐              ┌─────────────────────────┐ │    │
│  │ Auto Action │              │    TIER 3: LLM REVIEW   │ │    │
│  └─────────────┘              │  (nuanced cases)        │ │    │
│                               └────────────┬────────────┘ │    │
│                                            │               │    │
│                        ┌───────────────────┼──────────────┐│    │
│                        │ Confident         │ Uncertain    ││    │
│                        ▼                   ▼              ││    │
│                 ┌─────────────┐    ┌─────────────┐       ││    │
│                 │ Auto Action │    │   Human     │       ││    │
│                 └─────────────┘    │   Review    │       ││    │
│                                    └─────────────┘       ││    │
│                                                          ││    │
└──────────────────────────────────────────────────────────┘│    │
```

### Processing Tiers

| Tier | Method | Latency | Cost | Coverage |
|------|--------|---------|------|----------|
| 1 | Hash/keyword | < 10ms | $0.0001 | 5% blocked |
| 2 | ML classifiers | < 100ms | $0.001 | 85% auto-decided |
| 3 | LLM review | < 3s | $0.01 | 8% nuanced |
| 4 | Human review | Minutes | $0.50 | 2% escalated |

---

## Classification Pipeline

### Tier 1: Fast Filters

```python
class FastFilters:
    """
    Immediate blocking for known harmful content.
    No false positives for matches.
    """
    
    def __init__(self):
        self.hash_db = PhotoDNADatabase()  # CSAM detection
        self.keyword_filter = KeywordBlocklist()
        self.pattern_matcher = RegexPatterns()
    
    async def filter(self, content: Content) -> FilterResult:
        # CSAM hash matching (highest priority)
        if content.has_media:
            hash_match = await self.hash_db.check(content.media_hashes)
            if hash_match:
                return FilterResult(
                    action="block_report",
                    reason="csam_hash_match",
                    confidence=1.0,
                    tier=1
                )
        
        # Keyword blocklist
        if content.text:
            keyword_match = self.keyword_filter.check(content.text)
            if keyword_match and keyword_match.severity == "critical":
                return FilterResult(
                    action="block_review",
                    reason=f"keyword_{keyword_match.category}",
                    confidence=0.99,
                    tier=1
                )
        
        # Pattern matching (phone numbers in suspicious context, etc)
        pattern_match = self.pattern_matcher.check(content.text)
        if pattern_match:
            return FilterResult(
                action="elevate",
                reason=f"pattern_{pattern_match.type}",
                confidence=pattern_match.confidence,
                tier=1
            )
        
        return FilterResult(action="continue", tier=1)
```

### Tier 2: ML Classification

```python
class MLClassificationPipeline:
    """
    Fast, specialized ML models for content classification.
    """
    
    def __init__(self):
        self.text_classifier = TextSafetyClassifier()
        self.image_classifier = ImageSafetyClassifier()
        self.multimodal = MultimodalClassifier()
    
    async def classify(self, content: Content) -> ClassificationResult:
        tasks = []
        
        if content.text:
            tasks.append(self.classify_text(content.text))
        
        if content.has_images:
            tasks.append(self.classify_images(content.images))
        
        if content.text and content.has_images:
            tasks.append(self.classify_multimodal(content))
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        final_scores = self.aggregate_scores(results)
        
        # Determine action based on confidence
        for category, score in final_scores.items():
            if score > 0.95:  # High confidence
                return ClassificationResult(
                    action=self.get_action(category),
                    category=category,
                    confidence=score,
                    tier=2,
                    needs_review=False
                )
            elif score > 0.7:  # Medium confidence
                return ClassificationResult(
                    action="elevate_llm",
                    category=category,
                    confidence=score,
                    tier=2,
                    needs_review=True
                )
        
        # Low confidence on all categories = likely safe
        return ClassificationResult(
            action="allow",
            confidence=1 - max(final_scores.values()),
            tier=2
        )
```

### Tier 3: LLM Review

```python
class LLMReviewer:
    """
    LLM for nuanced content that ML models cannot handle.
    """
    
    def __init__(self):
        self.model = ClaudeSonnet()
        self.policy_context = self.load_policy_context()
    
    async def review(
        self,
        content: Content,
        ml_result: ClassificationResult
    ) -> ReviewResult:
        prompt = f"""
You are a content moderator reviewing potentially harmful content.

Content policies:
{self.policy_context}

Content to review:
Text: {content.text}
{f"Image description: {content.image_description}" if content.has_images else ""}

ML classifier flagged this as potentially: {ml_result.category}
ML confidence: {ml_result.confidence}

Analyze this content and determine:
1. Does it violate any policy? If so, which one?
2. What is the severity (critical/high/medium/low)?
3. What action should be taken (block/warn/allow)?
4. Confidence in your decision (high/medium/low)?

Consider context, intent, and potential for harm.
Return your analysis as JSON.
"""
        
        response = await self.model.generate(prompt)
        decision = json.loads(response)
        
        # Map confidence to action
        if decision["confidence"] == "low":
            return ReviewResult(
                action="human_review",
                reason=decision["analysis"],
                llm_decision=decision,
                tier=3
            )
        
        return ReviewResult(
            action=decision["action"],
            category=decision.get("violated_policy"),
            severity=decision["severity"],
            confidence=decision["confidence"],
            tier=3
        )
```

---

## Human-in-the-Loop

### Review Queue Management

```python
class ReviewQueueManager:
    """
    Prioritize and route content to human moderators.
    """
    
    def __init__(self):
        self.queues = {
            "critical": PriorityQueue(),  # CSAM, violence - immediate
            "high": PriorityQueue(),      # Hate speech - < 15 min
            "standard": PriorityQueue(),  # Other violations - < 1 hour
            "appeals": PriorityQueue()    # User appeals
        }
    
    async def enqueue(self, content: Content, result: ReviewResult):
        priority = self.calculate_priority(content, result)
        
        item = ReviewItem(
            content_id=content.id,
            content=content,
            ai_analysis=result,
            priority=priority,
            enqueued_at=datetime.now()
        )
        
        queue_name = self.get_queue(result.severity)
        await self.queues[queue_name].put(item)
        
        # Alert if critical
        if queue_name == "critical":
            await self.alert_moderators(item)
    
    def calculate_priority(self, content: Content, result: ReviewResult) -> float:
        priority = 0.0
        
        # Severity weight
        severity_weights = {"critical": 100, "high": 50, "medium": 20, "low": 5}
        priority += severity_weights.get(result.severity, 0)
        
        # Reach weight (viral content prioritized)
        priority += min(content.reach_score * 10, 50)
        
        # Confidence inverse (less confident = higher priority)
        priority += (1 - result.confidence) * 30
        
        return priority
```

### Moderator Interface

```python
class ModeratorDecision:
    async def submit(
        self,
        moderator_id: str,
        content_id: str,
        decision: str,
        reason: str,
        notes: str = None
    ):
        # Record decision
        await self.store_decision({
            "content_id": content_id,
            "moderator_id": moderator_id,
            "decision": decision,
            "reason": reason,
            "notes": notes,
            "ai_recommendation": await self.get_ai_result(content_id),
            "decided_at": datetime.now()
        })
        
        # Execute action
        await self.execute_action(content_id, decision)
        
        # Update ML models with feedback
        await self.feedback_loop.record(
            content_id=content_id,
            ai_prediction=await self.get_ai_result(content_id),
            human_decision=decision
        )
```

---

## Adversarial Robustness

### Evasion Techniques and Defenses

| Evasion Technique | Defense |
|-------------------|---------|
| Character substitution (h@te) | Normalization + homoglyph mapping |
| Image text (text in images) | OCR pipeline |
| Invisible characters | Unicode normalization |
| Context manipulation | Multi-turn analysis |
| Encoded content | Decoding pipeline |
| Adversarial images | Robust vision models |

### Defensive Pipeline

```python
class AdversarialDefense:
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.ocr = OCRPipeline()
        self.decoder = ContentDecoder()
    
    def preprocess(self, content: Content) -> Content:
        processed = content.copy()
        
        # Normalize text
        if processed.text:
            processed.text = self.normalizer.normalize(processed.text)
            processed.text = self.decoder.decode_obfuscation(processed.text)
        
        # Extract text from images
        if processed.has_images:
            for image in processed.images:
                extracted_text = self.ocr.extract(image)
                if extracted_text:
                    processed.text = f"{processed.text}\n[IMAGE TEXT]: {extracted_text}"
        
        return processed
    
    def normalize(self, text: str) -> str:
        # Homoglyph normalization
        text = self.homoglyph_map(text)
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Remove zero-width characters
        text = re.sub(r"[\u200b-\u200f\u2028-\u202f]", "", text)
        
        # Leetspeak normalization
        text = self.leetspeak_decode(text)
        
        return text
```

---

## Results and Metrics

### Performance Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Harmful content exposure | 2% | 0.08% | 96% reduction |
| Review latency (critical) | 4 hours | 8 minutes | 30x faster |
| False positive rate | 15% | 4.2% | 72% reduction |
| Moderator efficiency | 50/day | 200/day | 4x increase |

### Cost Analysis

| Component | Monthly Cost | Per 10M Posts |
|-----------|--------------|---------------|
| Tier 1 filters | $1,000 | $0.10 |
| Tier 2 ML | $15,000 | $1.50 |
| Tier 3 LLM | $8,000 | $0.80 |
| Human review | $150,000 | $15.00 |
| **Total** | **$174,000** | **$17.40** |

*Human review still dominates cost but focused on hard cases*

---

## Interview Walkthrough

**Interviewer:** "Design a content moderation system for a social media platform."

**Strong response:**

1. **Clarify scale and requirements** (1 min)
   - "What's the volume? What content types? What's acceptable false positive rate?"
   - "Any regulatory requirements (CSAM reporting, GDPR)?"

2. **Multi-tier architecture** (3 min)
   - "I would use a cascade of increasing sophistication:"
   - "Tier 1: Hash matching, keyword filters - instant, certain"
   - "Tier 2: ML classifiers - fast, specialized"
   - "Tier 3: LLM review - nuanced, context-aware"
   - "Tier 4: Human review - final arbiter"
   - "Each tier handles what the previous cannot"

3. **Prioritization is key** (2 min)
   - "Not all harmful content is equal. CSAM and violence need immediate action. Hate speech is priority but not instant. Spam can wait."
   - "Priority queue based on severity, reach, and confidence"

4. **Human-in-the-loop design** (2 min)
   - "Humans for low-confidence decisions and appeals"
   - "AI handles 95%+ automatically to make human review economically viable"
   - "Feedback loop: human decisions improve ML models"

5. **Adversarial robustness** (2 min)
   - "Users will evade detection. Defenses include:"
   - "Text normalization for obfuscation"
   - "OCR for text in images"
   - "Continuous model updates as evasion evolves"

6. **Metrics** (1 min)
   - "Primary: harmful content exposure rate (target < 0.1%)"
   - "Secondary: false positive rate (user experience)"
   - "Operational: review latency, moderator throughput"

---

## References

- Meta Content Moderation: https://transparency.fb.com/
- Google Perspective API: https://perspectiveapi.com/
- OpenAI Moderation: https://platform.openai.com/docs/guides/moderation

---

*Next: [Appendix A: LLM Pricing Reference](../appendices/a-pricing-reference.md)*

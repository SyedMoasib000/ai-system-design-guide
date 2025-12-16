# Guardrails and Safety

Guardrails are systems that constrain LLM behavior to ensure safe, reliable outputs. This chapter covers safety mechanisms, content filtering, and reliability patterns for production systems.

## Table of Contents

- [Why Guardrails Matter](#why-guardrails-matter)
- [Types of Guardrails](#types-of-guardrails)
- [Input Guardrails](#input-guardrails)
- [Output Guardrails](#output-guardrails)
- [Hallucination Mitigation](#hallucination-mitigation)
- [Structured Output Validation](#structured-output-validation)
- [Fallback Strategies](#fallback-strategies)
- [Guardrail Frameworks](#guardrail-frameworks)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Guardrails Matter

### The Reliability Challenge

LLMs are probabilistic and can produce:
- Factually incorrect information (hallucination)
- Harmful or inappropriate content
- Off-topic or unhelpful responses
- Inconsistent formatting
- Leaked sensitive information

### Business Impact

| Risk | Consequence |
|------|-------------|
| Harmful content | Legal liability, reputation damage |
| Hallucination | User harm, trust erosion |
| PII leakage | Compliance violations, fines |
| Off-topic responses | Poor user experience |
| Format errors | Application crashes |

---

## Types of Guardrails

### Guardrail Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Input                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INPUT GUARDRAILS                                │
│  • Topic filtering                                              │
│  • PII detection                                                │
│  • Jailbreak detection                                          │
│  • Input validation                                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Generation                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT GUARDRAILS                               │
│  • Content filtering                                            │
│  • Factuality checking                                          │
│  • Format validation                                            │
│  • Relevance checking                                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Response                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input Guardrails

### Topic Classification

Block off-topic or prohibited requests:

```python
class TopicGuardrail:
    def __init__(self, allowed_topics: list[str], model: str = "gpt-4o-mini"):
        self.allowed_topics = allowed_topics
        self.classifier = TopicClassifier(model)
    
    def check(self, user_input: str) -> GuardrailResult:
        topic = self.classifier.classify(user_input)
        
        if topic in self.allowed_topics:
            return GuardrailResult(passed=True)
        
        return GuardrailResult(
            passed=False,
            reason=f"Topic '{topic}' is not supported",
            suggested_response="I can only help with questions about our products and services."
        )

# Usage
guardrail = TopicGuardrail(
    allowed_topics=["product_info", "billing", "technical_support", "general"]
)
result = guardrail.check("How do I cook pasta?")
# Result: passed=False, topic outside allowed scope
```

### PII Detection

Detect and handle personally identifiable information:

```python
class PIIGuardrail:
    def __init__(self):
        self.patterns = {
            "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
    
    def check(self, text: str) -> GuardrailResult:
        detected = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = len(matches)
        
        if detected:
            return GuardrailResult(
                passed=False,
                reason=f"PII detected: {detected}",
                suggested_action="redact"
            )
        
        return GuardrailResult(passed=True)
    
    def redact(self, text: str) -> str:
        redacted = text
        for pii_type, pattern in self.patterns.items():
            redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)
        return redacted
```

### Jailbreak Detection

Detect attempts to bypass safety measures:

```python
class JailbreakDetector:
    def __init__(self):
        self.classifier = load_jailbreak_classifier()
        
        # Known patterns (supplement with ML classifier)
        self.patterns = [
            r"ignore.*previous.*instructions",
            r"pretend.*you.*are",
            r"act.*as.*if",
            r"you.*are.*now",
            r"DAN.*mode",
            r"developer.*mode",
            r"jailbreak",
            r"bypass.*filter",
        ]
    
    def check(self, text: str) -> GuardrailResult:
        # Pattern matching
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason="Potential jailbreak attempt detected",
                    confidence=0.8
                )
        
        # ML classifier for sophisticated attempts
        score = self.classifier.predict(text)
        if score > 0.7:
            return GuardrailResult(
                passed=False,
                reason="ML classifier flagged as jailbreak",
                confidence=score
            )
        
        return GuardrailResult(passed=True)
```

### Input Length and Rate Limiting

```python
class InputLimitsGuardrail:
    def __init__(
        self,
        max_tokens: int = 4000,
        max_requests_per_minute: int = 20
    ):
        self.max_tokens = max_tokens
        self.max_rpm = max_requests_per_minute
        self.request_counts = defaultdict(list)
    
    def check(self, text: str, user_id: str) -> GuardrailResult:
        # Token limit
        tokens = count_tokens(text)
        if tokens > self.max_tokens:
            return GuardrailResult(
                passed=False,
                reason=f"Input too long: {tokens} tokens (max {self.max_tokens})"
            )
        
        # Rate limit
        now = time.time()
        recent = [t for t in self.request_counts[user_id] if now - t < 60]
        self.request_counts[user_id] = recent
        
        if len(recent) >= self.max_rpm:
            return GuardrailResult(
                passed=False,
                reason="Rate limit exceeded"
            )
        
        self.request_counts[user_id].append(now)
        return GuardrailResult(passed=True)
```

---

## Output Guardrails

### Content Safety Filter

```python
class ContentSafetyGuardrail:
    def __init__(self):
        self.categories = [
            "hate",
            "violence",
            "sexual",
            "self_harm",
            "illegal_activity"
        ]
        self.classifier = load_content_classifier()
    
    def check(self, response: str) -> GuardrailResult:
        scores = self.classifier.predict(response)
        
        flagged = {cat: score for cat, score in scores.items() if score > 0.7}
        
        if flagged:
            return GuardrailResult(
                passed=False,
                reason=f"Content flagged: {flagged}",
                suggested_response="I cannot provide that type of content."
            )
        
        return GuardrailResult(passed=True)

# Using OpenAI Moderation API
def check_with_openai(text: str) -> GuardrailResult:
    response = openai.Moderation.create(input=text)
    result = response["results"][0]
    
    if result["flagged"]:
        categories = [k for k, v in result["categories"].items() if v]
        return GuardrailResult(
            passed=False,
            reason=f"Flagged categories: {categories}"
        )
    
    return GuardrailResult(passed=True)
```

### Relevance Check

Ensure response addresses the question:

```python
class RelevanceGuardrail:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def check(self, query: str, response: str) -> GuardrailResult:
        # Embedding similarity
        query_emb = embed(query)
        response_emb = embed(response)
        similarity = cosine_similarity(query_emb, response_emb)
        
        if similarity < self.threshold:
            return GuardrailResult(
                passed=False,
                reason=f"Low relevance score: {similarity:.2f}",
                suggested_action="regenerate"
            )
        
        return GuardrailResult(passed=True, metadata={"relevance": similarity})
```

### Factuality Check (for RAG)

```python
class FactualityGuardrail:
    def __init__(self):
        self.nli_model = load_nli_model()
    
    def check(self, response: str, context: str) -> GuardrailResult:
        # Split response into claims
        claims = self.extract_claims(response)
        
        unsupported = []
        for claim in claims:
            # Check if claim is entailed by context
            result = self.nli_model.predict(premise=context, hypothesis=claim)
            
            if result["label"] == "contradiction":
                unsupported.append({"claim": claim, "issue": "contradicts context"})
            elif result["label"] == "neutral" and result["confidence"] > 0.8:
                unsupported.append({"claim": claim, "issue": "not supported"})
        
        if unsupported:
            return GuardrailResult(
                passed=False,
                reason="Response contains unsupported claims",
                metadata={"unsupported_claims": unsupported}
            )
        
        return GuardrailResult(passed=True)
```

---

## Hallucination Mitigation

### Multi-Layer Approach

```python
class HallucinationGuard:
    def __init__(self):
        self.strategies = [
            self.check_context_grounding,
            self.check_self_consistency,
            self.check_confidence_signals
        ]
    
    def check(self, query: str, response: str, context: str) -> GuardrailResult:
        issues = []
        
        for strategy in self.strategies:
            result = strategy(query, response, context)
            if not result.passed:
                issues.append(result.reason)
        
        if issues:
            return GuardrailResult(
                passed=False,
                reason="; ".join(issues)
            )
        
        return GuardrailResult(passed=True)
    
    def check_context_grounding(self, query, response, context) -> GuardrailResult:
        # Use LLM to verify grounding
        prompt = f"""
        Context: {context}
        
        Response: {response}
        
        Is every factual claim in the response supported by the context?
        Answer YES or NO, then explain.
        """
        
        result = llm.generate(prompt)
        
        if result.startswith("NO"):
            return GuardrailResult(passed=False, reason="Ungrounded claims detected")
        
        return GuardrailResult(passed=True)
    
    def check_self_consistency(self, query, response, context) -> GuardrailResult:
        # Generate multiple responses and check consistency
        responses = [
            llm.generate(query, context=context, temperature=0.7)
            for _ in range(3)
        ]
        
        # Check if responses are semantically similar
        embeddings = [embed(r) for r in responses]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
        
        avg_similarity = sum(similarities) / len(similarities)
        
        if avg_similarity < 0.7:
            return GuardrailResult(
                passed=False,
                reason=f"Low self-consistency: {avg_similarity:.2f}"
            )
        
        return GuardrailResult(passed=True)
```

### Abstention Strategy

Train the model to say "I don't know":

```python
ABSTENTION_PROMPT = """
You are a helpful assistant. Answer based only on the provided context.

IMPORTANT RULES:
1. If the answer is not in the context, say "I don't have information about that."
2. If you are uncertain, express your uncertainty.
3. Never make up facts not present in the context.
4. It is better to abstain than to be wrong.

Context:
{context}

Question: {question}

Answer:
"""

class AbstentionDetector:
    def __init__(self):
        self.abstention_phrases = [
            "i don't have information",
            "i cannot find",
            "not mentioned in",
            "i'm not sure",
            "i don't know",
            "no information available"
        ]
    
    def is_abstention(self, response: str) -> bool:
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.abstention_phrases)
```

---

## Structured Output Validation

### JSON Schema Validation

```python
from jsonschema import validate, ValidationError

class StructuredOutputGuardrail:
    def __init__(self, schema: dict):
        self.schema = schema
    
    def check(self, response: str) -> GuardrailResult:
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            return GuardrailResult(
                passed=False,
                reason=f"Invalid JSON: {e}",
                suggested_action="retry_with_format_instruction"
            )
        
        # Validate against schema
        try:
            validate(instance=data, schema=self.schema)
        except ValidationError as e:
            return GuardrailResult(
                passed=False,
                reason=f"Schema validation failed: {e.message}",
                suggested_action="retry_with_format_instruction"
            )
        
        return GuardrailResult(passed=True, data=data)

# Usage
product_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"}
    },
    "required": ["name", "price"]
}

guardrail = StructuredOutputGuardrail(product_schema)
```

### Retry with Correction

```python
class StructuredOutputRetry:
    def __init__(self, schema: dict, max_retries: int = 3):
        self.schema = schema
        self.max_retries = max_retries
        self.guardrail = StructuredOutputGuardrail(schema)
    
    def generate_with_validation(self, prompt: str) -> dict:
        for attempt in range(self.max_retries):
            response = llm.generate(prompt)
            result = self.guardrail.check(response)
            
            if result.passed:
                return result.data
            
            # Add correction instruction
            prompt = f"""
            {prompt}
            
            Your previous response had this error: {result.reason}
            
            Please fix and respond with valid JSON matching the schema.
            Previous response: {response}
            
            Corrected response:
            """
        
        raise ValueError("Failed to generate valid structured output")
```

---

## Fallback Strategies

### Graceful Degradation

```python
class FallbackChain:
    def __init__(self, strategies: list):
        self.strategies = strategies
    
    def execute(self, query: str, context: str) -> Response:
        for strategy in self.strategies:
            try:
                result = strategy.generate(query, context)
                
                if self.is_acceptable(result):
                    return Response(
                        content=result,
                        source=strategy.name,
                        confidence="high"
                    )
            except Exception as e:
                self.log_error(strategy.name, e)
                continue
        
        # All strategies failed
        return Response(
            content="I apologize, but I am unable to help with that request right now.",
            source="fallback",
            confidence="none"
        )

# Usage
fallback = FallbackChain([
    PrimaryLLM(model="gpt-4o"),
    SecondaryLLM(model="claude-3.5-sonnet"),
    CachedResponses(),
    HumanEscalation()
])
```

### Human Escalation

```python
class HumanEscalationGuardrail:
    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold
    
    def check(self, response: str, confidence: float) -> GuardrailResult:
        if confidence < self.threshold:
            return GuardrailResult(
                passed=False,
                reason="Low confidence response",
                suggested_action="escalate_to_human",
                metadata={"confidence": confidence}
            )
        
        return GuardrailResult(passed=True)

def handle_low_confidence(query: str, response: str, metadata: dict):
    # Create ticket for human review
    ticket = create_support_ticket(
        query=query,
        ai_response=response,
        confidence=metadata["confidence"],
        priority="normal"
    )
    
    return f"I want to make sure I give you accurate information. I've escalated your question to our team. Ticket: {ticket.id}"
```

---

## Guardrail Frameworks

### NeMo Guardrails (NVIDIA)

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Define rails in Colang
"""
define user ask about competitors
    "What do you think about [competitor]?"
    "Is [competitor] better?"

define bot refuse competitor discussion
    "I'm focused on helping you with our products. Is there something specific I can help you with?"

define flow
    user ask about competitors
    bot refuse competitor discussion
"""

response = rails.generate(messages=[{"role": "user", "content": user_message}])
```

### Guardrails AI

```python
from guardrails import Guard
from guardrails.validators import ValidJSON, ToxicLanguage

guard = Guard.from_string(
    validators=[
        ValidJSON(on_fail="reask"),
        ToxicLanguage(threshold=0.8, on_fail="filter")
    ],
    prompt="""
    Extract product information as JSON:
    {
        "name": string,
        "price": number
    }
    
    Product description: ${description}
    """
)

result = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    description=product_description
)
```

---

## Interview Questions

### Q: How do you prevent hallucination in a production RAG system?

**Strong answer:**
Multi-layer approach:

**1. Retrieval quality:**
- High-quality retrieval is the first defense
- If we retrieve wrong context, model will hallucinate
- Use reranking to ensure relevance

**2. Prompt engineering:**
- Explicit instruction: "Answer only from context"
- Encourage abstention: "If not in context, say you don't know"
- Low temperature (0.1-0.3)

**3. Output validation:**
- Factuality checking: NLI model or LLM judge
- Citation verification: Check claims against sources
- Self-consistency: Multiple samples should agree

**4. Abstention strategy:**
- Train/prompt model to say "I don't know"
- Detect low-confidence responses
- Escalate to human when uncertain

**5. Monitoring:**
- Track hallucination rate in production
- User feedback on accuracy
- Regular evaluation on test set

### Q: Design a guardrail system for a customer service chatbot.

**Strong answer:**
I would implement guardrails at input and output:

**Input guardrails:**
1. Topic filter: Only allow product/service questions
2. PII detection: Redact or warn about sensitive data
3. Jailbreak detection: Block manipulation attempts
4. Rate limiting: Prevent abuse

**Output guardrails:**
1. Content safety: No harmful/inappropriate content
2. Relevance check: Response addresses the question
3. Brand voice: Consistent tone and messaging
4. Factuality: Claims supported by knowledge base

**Fallback chain:**
```
Primary LLM → Backup LLM → Canned responses → Human escalation
```

**Monitoring:**
- Log all guardrail triggers
- Alert on high block rates
- Regular review of blocked content
- User satisfaction tracking

---

## References

- NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails
- Guardrails AI: https://github.com/guardrails-ai/guardrails
- OpenAI Moderation: https://platform.openai.com/docs/guides/moderation
- Llama Guard: https://ai.meta.com/research/publications/llama-guard/

---

*Next: [Reliability Patterns](02-reliability-patterns.md)*

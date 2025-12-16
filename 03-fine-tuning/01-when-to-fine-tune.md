# When to Fine-Tune

Fine-tuning LLMs is powerful but often unnecessary. This chapter helps you decide when fine-tuning is the right approach.

## Table of Contents

- [Fine-Tuning Decision Framework](#fine-tuning-decision-framework)
- [Alternatives to Fine-Tuning](#alternatives-to-fine-tuning)
- [When Fine-Tuning Helps](#when-fine-tuning-helps)
- [When Fine-Tuning Hurts](#when-fine-tuning-hurts)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Fine-Tuning Decision Framework

### Decision Tree

```
Start: Can prompting solve this problem?
    │
    ├── Yes ───────────────────────────────────────► Use prompting
    │
    └── No ──► Is RAG sufficient?
                    │
                    ├── Yes ───────────────────────► Use RAG
                    │
                    └── No ──► Do you have quality training data?
                                    │
                                    ├── No ────────► Collect data first
                                    │
                                    └── Yes ──► Is the task narrow and well-defined?
                                                    │
                                                    ├── No ────────► Consider few-shot
                                                    │
                                                    └── Yes ──► Fine-tune
```

### Quick Assessment

| Criterion | Prompting | RAG | Fine-Tuning |
|-----------|-----------|-----|-------------|
| Need domain knowledge | ✓ (in prompt) | ✓ (retrieved) | ✓ (learned) |
| Need specific style/format | ✓ (examples) | ✗ | ✓ (best) |
| Need specific behavior | ✓ (instructions) | ✗ | ✓ (best) |
| Response latency critical | ✓ | ✗ (retrieval adds latency) | ✓ |
| Knowledge updates frequently | ✓ | ✓ (best) | ✗ (retraining needed) |
| Cost per request matters | ✗ (long prompts costly) | ✗ (retrieval + prompt) | ✓ (shorter prompts) |
| Training data available | N/A | N/A | Required |

---

## Alternatives to Fine-Tuning

### Prompting First

Many problems solved through fine-tuning are actually prompting problems:

```python
# BEFORE: "We need to fine-tune for consistent JSON output"
# AFTER: Just use better prompting

STRUCTURED_PROMPT = """
Extract information from the text and return as JSON.

Required format:
```json
{
    "name": "string",
    "date": "YYYY-MM-DD",
    "amount": number,
    "category": "one of: A, B, C"
}
```

Text to process:
{text}

Return only the JSON, no other text.
"""

# Or use structured output features
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)
```

### RAG Before Fine-Tuning

For knowledge-based tasks:

| Scenario | Solution |
|----------|----------|
| "Model does not know about our products" | RAG with product documentation |
| "Model does not know our policies" | RAG with policy documents |
| "Model does not know recent events" | RAG with news/update feeds |
| "Model does not know company terminology" | Glossary in system prompt |

Fine-tuning knowledge is problematic because:
- Knowledge becomes stale
- No source attribution
- Difficult to update
- Can hallucinate "learned" knowledge

### Few-Shot Learning

```python
# Instead of fine-tuning for style, use few-shot examples

FEW_SHOT_PROMPT = """
Convert customer queries to SQL. Here are examples:

Query: How many orders last month?
SQL: SELECT COUNT(*) FROM orders WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)

Query: Top 5 customers by revenue
SQL: SELECT customer_id, SUM(total) as revenue FROM orders GROUP BY customer_id ORDER BY revenue DESC LIMIT 5

Query: {user_query}
SQL:"""
```

---

## When Fine-Tuning Helps

### 1. Consistent Output Format

When you need very specific, consistent formatting that prompting cannot achieve:

```python
# Medical report summarization with specific format
# Thousands of examples teach the exact structure

training_example = {
    "messages": [
        {"role": "system", "content": "Summarize medical reports in the department format."},
        {"role": "user", "content": "Patient: John Doe, 45M. Presented with chest pain..."},
        {"role": "assistant", "content": "SUMMARY:\n• Chief Complaint: Chest pain\n• Assessment: Possible angina\n• Plan: ECG, troponin levels..."}
    ]
}
```

### 2. Specialized Domain Language

When the model needs to understand domain-specific jargon:

```python
# Legal document analysis with legal terminology
# The model learns specialized usage

training_example = {
    "messages": [
        {"role": "user", "content": "What is the force majeure clause in this contract?"},
        {"role": "assistant", "content": "The force majeure clause (Section 8.2) excuses performance..."}
    ]
}
```

### 3. Specific Behavior Patterns

When you need the model to behave in specific ways consistently:

```python
# Customer service with specific escalation triggers
# The model learns when to escalate vs. handle

training_example = {
    "messages": [
        {"role": "user", "content": "I've been waiting 3 weeks for my order!"},
        {"role": "assistant", "content": "I sincerely apologize for this delay. [ESCALATE: Order > 2 weeks late]..."}
    ]
}
```

### 4. Reducing Prompt Length

When long prompts are costly:

```python
# BEFORE: 2000 token system prompt with examples and instructions
# Cost: $0.01 per request at high volume

# AFTER: Fine-tuned model, 100 token prompt
# Cost: $0.001 per request + amortized training cost
```

### 5. Latency Optimization

Fine-tuned models skip few-shot processing:

| Approach | Prompt Tokens | Response Latency |
|----------|---------------|------------------|
| Few-shot (10 examples) | 3000 | 800ms |
| Fine-tuned | 200 | 200ms |

---

## When Fine-Tuning Hurts

### 1. Knowledge That Changes

```python
# BAD: Fine-tuning on company policies
# These change and you'll need to retrain

# GOOD: RAG with policy documents
# Update documents, no retraining needed
```

### 2. Small Dataset

```python
# BAD: Fine-tuning on 50 examples
# Likely to overfit, learn artifacts

# Minimum recommended:
# - GPT-4 fine-tuning: 50-100 examples minimum
# - Good results: 500-1000 examples
# - Great results: 5000+ examples
```

### 3. Broad, Varied Tasks

```python
# BAD: Single fine-tune for "general customer service"
# Too broad, hard to capture in training data

# GOOD: Fine-tune for specific narrow tasks
# "Extract shipping info from emails"
# "Classify support tickets into 10 categories"
```

### 4. Need for Explainability

```python
# Fine-tuning is a black box
# You cannot trace WHY the model responded a way

# If you need: "Why did you recommend this?"
# Use RAG with source attribution instead
```

---

## Cost-Benefit Analysis

### Training Costs

| Provider | Model | Training Cost |
|----------|-------|---------------|
| OpenAI | GPT-4o-mini | $3.00 / 1M tokens |
| OpenAI | GPT-4o | $25.00 / 1M tokens |
| Anthropic | Claude | Not available |
| Google | Gemini | Not available |

⚠️ Verify current pricing - rates change frequently.

### Breakeven Analysis

```python
def calculate_breakeven(
    training_examples: int,
    avg_tokens_per_example: int,
    training_cost_per_token: float,
    prompt_savings_per_request: int,
    inference_cost_per_token: float
) -> int:
    # Training cost
    total_training_tokens = training_examples * avg_tokens_per_example
    training_cost = total_training_tokens * training_cost_per_token
    
    # Savings per request
    savings_per_request = prompt_savings_per_request * inference_cost_per_token
    
    # Breakeven requests
    return int(training_cost / savings_per_request)

# Example:
# 1000 examples, 500 tokens each, $25/1M training
# Save 1500 tokens per request, $15/1M inference
breakeven = calculate_breakeven(1000, 500, 25e-6, 1500, 15e-6)
# = ~555,555 requests to break even
```

### ROI Calculation

| Scenario | Training Cost | Monthly Requests | Monthly Savings | Payback Period |
|----------|---------------|------------------|-----------------|----------------|
| Small | $500 | 10K | $150 | 3.3 months |
| Medium | $2,000 | 100K | $1,500 | 1.3 months |
| Large | $5,000 | 1M | $15,000 | 10 days |

---

## Interview Questions

### Q: When would you choose fine-tuning over RAG?

**Strong answer:**

"Fine-tuning and RAG solve different problems:

**Choose RAG when:**
- You need to inject knowledge (facts, documents, policies)
- Knowledge changes frequently
- You need source attribution
- Training data is limited

**Choose fine-tuning when:**
- You need consistent behavior/style
- You need to reduce prompt length for cost or latency
- You have 500+ high-quality examples
- The task is narrow and well-defined

**Example decision:**
- Customer FAQ bot: RAG (knowledge-based, documents change)
- SQL generation from natural language: Fine-tuning (consistent format, narrow task)
- Medical coding: Both (fine-tune for format, RAG for code definitions)

I always try prompting and few-shot first. Fine-tuning is a last resort when those fail, not a first choice."

### Q: What are the risks of fine-tuning?

**Strong answer:**

"Several risks to consider:

**Overfitting:** With small datasets, the model memorizes training examples rather than learning generalizable patterns. Results look good on training data but fail on real inputs.

**Catastrophic forgetting:** The model may lose general capabilities it had before. If you fine-tune on only one narrow task, it might become worse at everything else.

**Training data artifacts:** The model learns artifacts in your data. If your examples have a specific formatting quirk, the model learns that as required behavior.

**Stale knowledge:** Knowledge fine-tuned into weights cannot be updated without retraining. This is problematic for dynamic information.

**Evaluation difficulty:** It is hard to comprehensively test a fine-tuned model for all edge cases. It may fail unpredictably on inputs outside the training distribution.

**Mitigation:** Thorough evaluation on held-out set, A/B testing against base model, monitoring in production for drift."

---

## References

- OpenAI Fine-Tuning Guide: https://platform.openai.com/docs/guides/fine-tuning
- When to Fine-Tune: https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning

---

*Next: [Data Preparation for Fine-Tuning](02-data-preparation.md)*

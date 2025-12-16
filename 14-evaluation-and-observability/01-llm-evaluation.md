# LLM Evaluation

Evaluating LLM systems is fundamentally different from traditional ML. This chapter covers metrics, methodologies, and practical approaches for measuring quality in production.

## Table of Contents

- [Why LLM Evaluation Is Hard](#why-llm-evaluation-is-hard)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Automated Evaluation Methods](#automated-evaluation-methods)
- [LLM-as-Judge](#llm-as-judge)
- [Human Evaluation](#human-evaluation)
- [RAG-Specific Evaluation](#rag-specific-evaluation)
- [Building Evaluation Pipelines](#building-evaluation-pipelines)
- [Production Monitoring](#production-monitoring)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why LLM Evaluation Is Hard

### The Fundamental Challenge

Traditional ML has clear metrics (accuracy, F1, AUC). LLM outputs are open-ended text where "correct" is subjective.

| Traditional ML | LLM Systems |
|----------------|-------------|
| Single correct answer | Many valid responses |
| Objective metrics | Subjective quality |
| Easy to automate | Requires judgment |
| Static test sets | Need diverse scenarios |

### Multiple Dimensions of Quality

A response can be:
- Correct but poorly written
- Well-written but incomplete
- Complete but not relevant
- Relevant but unsafe

You need to measure multiple dimensions independently.

---

## Evaluation Dimensions

### Core Dimensions

| Dimension | What It Measures | How to Evaluate |
|-----------|------------------|-----------------|
| **Correctness** | Factually accurate? | Ground truth, LLM judge |
| **Relevance** | Answers the question? | LLM judge, human |
| **Completeness** | All aspects covered? | Checklist, LLM judge |
| **Coherence** | Well-structured, logical? | LLM judge, human |
| **Conciseness** | Appropriately brief? | Token count, LLM judge |
| **Safety** | No harmful content? | Classifiers, LLM judge |
| **Helpfulness** | Actually useful? | Human feedback |

### Task-Specific Dimensions

**For RAG:**
- Faithfulness: Grounded in retrieved context?
- Attribution: Proper citations?
- No hallucination: Nothing made up?

**For Code Generation:**
- Executability: Does it run?
- Correctness: Passes tests?
- Style: Follows conventions?

**For Summarization:**
- Coverage: Key points included?
- Factual consistency: No introduced errors?
- Compression: Appropriate length reduction?

---

## Automated Evaluation Methods

### Exact Match

Simplest approach, rarely sufficient alone:

```python
def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())
```

**Use for:** Multiple choice, classification, entity extraction

### Contains Keywords

```python
def keyword_match(prediction: str, required_keywords: list[str]) -> float:
    prediction_lower = prediction.lower()
    matches = sum(1 for kw in required_keywords if kw.lower() in prediction_lower)
    return matches / len(required_keywords)
```

**Use for:** Checking specific facts are mentioned

### Semantic Similarity

```python
def semantic_similarity(prediction: str, reference: str) -> float:
    pred_embedding = embed(prediction)
    ref_embedding = embed(reference)
    return cosine_similarity(pred_embedding, ref_embedding)
```

**Use for:** Paraphrase detection, general similarity
**Limitation:** High similarity does not mean correct

### ROUGE (Summarization)

Measures n-gram overlap:

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

def evaluate_summary(prediction: str, reference: str) -> dict:
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }
```

**Limitation:** Measures overlap, not quality

### Code Execution

For code generation, execution is ground truth:

```python
def evaluate_code(prediction: str, test_cases: list[dict]) -> dict:
    try:
        exec(prediction, globals())
    except SyntaxError as e:
        return {"syntax_valid": False, "error": str(e)}
    
    passed = 0
    for test in test_cases:
        try:
            result = eval(test["call"])
            if result == test["expected"]:
                passed += 1
        except Exception:
            pass
    
    return {
        "syntax_valid": True,
        "tests_passed": passed,
        "tests_total": len(test_cases),
        "pass_rate": passed / len(test_cases)
    }
```

---

## LLM-as-Judge

Use an LLM to evaluate another LLM's outputs.

### Basic Judge Prompt

```python
JUDGE_PROMPT = """
Evaluate the following response to the user's question.

Question: {question}
Response: {response}
Reference Answer (if available): {reference}

Rate the response on these criteria (1-5 scale):

1. Correctness: Is the information accurate?
2. Relevance: Does it address the question?
3. Completeness: Are all aspects covered?
4. Clarity: Is it well-written and clear?

For each criterion, provide:
- Score (1-5)
- Brief justification

Output as JSON:
{
    "correctness": {"score": X, "reason": "..."},
    "relevance": {"score": X, "reason": "..."},
    "completeness": {"score": X, "reason": "..."},
    "clarity": {"score": X, "reason": "..."},
    "overall": X
}
"""

def llm_judge(question: str, response: str, reference: str = None) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        reference=reference or "Not provided"
    )
    
    result = judge_model.generate(prompt)
    return json.loads(result)
```

### Pairwise Comparison

Compare two responses directly:

```python
PAIRWISE_PROMPT = """
Compare these two responses to the question and determine which is better.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider:
- Correctness
- Helpfulness
- Clarity
- Completeness

Output your choice (A or B) and explain why.

Choice:
"""

def pairwise_judge(question: str, response_a: str, response_b: str) -> dict:
    prompt = PAIRWISE_PROMPT.format(
        question=question,
        response_a=response_a,
        response_b=response_b
    )
    
    result = judge_model.generate(prompt)
    choice = "A" if "A" in result[:10] else "B"
    
    return {"winner": choice, "explanation": result}
```

### Judge Calibration

LLM judges have biases:

| Bias | Description | Mitigation |
|------|-------------|------------|
| Position bias | Prefers first or last option | Randomize order |
| Length bias | Prefers longer responses | Instruct to ignore length |
| Self-preference | Prefers own model's outputs | Use different judge model |
| Format bias | Prefers certain formats | Diverse training examples |

```python
def calibrated_pairwise_judge(question: str, response_a: str, response_b: str) -> dict:
    # Run twice with swapped positions
    result1 = pairwise_judge(question, response_a, response_b)
    result2 = pairwise_judge(question, response_b, response_a)
    
    # Check consistency
    result2_adjusted = "A" if result2["winner"] == "B" else "B"
    
    if result1["winner"] == result2_adjusted:
        return {"winner": result1["winner"], "confidence": "high"}
    else:
        return {"winner": "tie", "confidence": "low"}
```

---

## Human Evaluation

### When to Use Human Evaluation

| Use Case | Automate? | Human? |
|----------|-----------|--------|
| Rapid iteration | Yes | Spot check |
| Final quality assessment | Support | Yes |
| Subjective quality | No | Yes |
| Safety evaluation | Classifier | Review |
| Edge cases | No | Yes |

### Annotation Guidelines

```markdown
# Response Quality Annotation Guide

## Task
Rate the AI response quality on a 1-5 scale.

## Scale
5 - Excellent: Fully correct, helpful, well-written
4 - Good: Mostly correct, helpful, minor issues
3 - Acceptable: Correct but could be better
2 - Poor: Significant issues, partially helpful
1 - Unacceptable: Wrong, unhelpful, or harmful

## Instructions
1. Read the user question carefully
2. Read the AI response
3. Check for factual accuracy (if verifiable)
4. Assess helpfulness for the user's goal
5. Note any issues (inaccuracies, missing info, unclear)
6. Assign a score

## Examples
[Include 3-5 annotated examples at each score level]
```

### Inter-Annotator Agreement

```python
from sklearn.metrics import cohen_kappa_score

def calculate_agreement(annotator1: list, annotator2: list) -> dict:
    kappa = cohen_kappa_score(annotator1, annotator2)
    
    exact_agreement = sum(a == b for a, b in zip(annotator1, annotator2))
    exact_pct = exact_agreement / len(annotator1)
    
    return {
        "cohens_kappa": kappa,
        "exact_agreement": exact_pct,
        "interpretation": interpret_kappa(kappa)
    }

def interpret_kappa(kappa: float) -> str:
    if kappa < 0.2: return "Poor"
    if kappa < 0.4: return "Fair"
    if kappa < 0.6: return "Moderate"
    if kappa < 0.8: return "Substantial"
    return "Almost perfect"
```

---

## RAG-Specific Evaluation

### RAGAS Metrics

RAGAS provides standard RAG evaluation metrics:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def evaluate_rag(
    questions: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[str]
) -> dict:
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,      # Is answer grounded in context?
            answer_relevancy,  # Does answer address question?
            context_precision, # Are retrieved contexts relevant?
            context_recall     # Did we retrieve all needed context?
        ]
    )
    
    return result
```

### Faithfulness Evaluation

Check if response is grounded in context:

```python
FAITHFULNESS_PROMPT = """
Given the context and the response, determine if every claim in the 
response is supported by the context.

Context:
{context}

Response:
{response}

For each sentence in the response:
1. Extract the factual claims
2. Check if each claim is supported by the context
3. Mark as SUPPORTED or UNSUPPORTED

Output:
- Total claims: X
- Supported claims: Y
- Faithfulness score: Y/X
- Unsupported claims: [list]
"""

def evaluate_faithfulness(context: str, response: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(context=context, response=response)
    result = judge_model.generate(prompt)
    return parse_faithfulness_result(result)
```

### Context Relevance

Evaluate retrieved context quality:

```python
def evaluate_context_relevance(query: str, contexts: list[str]) -> dict:
    scores = []
    
    for context in contexts:
        prompt = f"""
        Query: {query}
        Context: {context}
        
        Is this context relevant to answering the query?
        Rate from 1-5 and explain.
        """
        
        result = judge_model.generate(prompt)
        score = extract_score(result)
        scores.append(score)
    
    return {
        "individual_scores": scores,
        "mean_relevance": sum(scores) / len(scores),
        "contexts_above_threshold": sum(1 for s in scores if s >= 3)
    }
```

---

## Building Evaluation Pipelines

### Evaluation Dataset Structure

```python
@dataclass
class EvalSample:
    id: str
    input: str
    expected_output: str  # Optional ground truth
    context: list[str]    # For RAG
    metadata: dict        # Category, difficulty, etc.

eval_dataset = [
    EvalSample(
        id="q001",
        input="What is the capital of France?",
        expected_output="Paris",
        context=[],
        metadata={"category": "factual", "difficulty": "easy"}
    ),
    # ... more samples
]
```

### Automated Evaluation Pipeline

```python
class EvaluationPipeline:
    def __init__(
        self,
        system_under_test,
        evaluators: list[Evaluator],
        dataset: list[EvalSample]
    ):
        self.sut = system_under_test
        self.evaluators = evaluators
        self.dataset = dataset
    
    def run(self) -> EvalReport:
        results = []
        
        for sample in self.dataset:
            # Get prediction
            prediction = self.sut.generate(sample.input)
            
            # Run all evaluators
            scores = {}
            for evaluator in self.evaluators:
                score = evaluator.evaluate(
                    input=sample.input,
                    prediction=prediction,
                    reference=sample.expected_output,
                    context=sample.context
                )
                scores[evaluator.name] = score
            
            results.append({
                "id": sample.id,
                "input": sample.input,
                "prediction": prediction,
                "scores": scores,
                "metadata": sample.metadata
            })
        
        return self.compile_report(results)
    
    def compile_report(self, results: list) -> EvalReport:
        # Aggregate by category, compute statistics
        report = EvalReport()
        
        for metric in self.evaluators:
            scores = [r["scores"][metric.name] for r in results]
            report.add_metric(metric.name, {
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores),
                "min": min(scores),
                "max": max(scores)
            })
        
        # Breakdown by category
        for category in set(r["metadata"]["category"] for r in results):
            category_results = [r for r in results if r["metadata"]["category"] == category]
            report.add_breakdown(category, self.aggregate(category_results))
        
        return report
```

---

## Production Monitoring

### Key Metrics to Track

```python
PRODUCTION_METRICS = {
    # Quality metrics (sample-based)
    "llm_judge_score": "Mean LLM judge score on sampled responses",
    "faithfulness": "RAG faithfulness on sampled responses",
    
    # User signals
    "thumbs_up_rate": "Positive feedback / total feedback",
    "regeneration_rate": "How often users regenerate",
    "copy_rate": "How often users copy responses",
    
    # Operational
    "error_rate": "Failed generations / total",
    "latency_p50": "Median response time",
    "latency_p99": "99th percentile response time",
    "tokens_per_response": "Average output length",
    
    # Cost
    "cost_per_request": "Average cost per request",
    "daily_cost": "Total daily API spend"
}
```

### Online Evaluation

```python
class OnlineEvaluator:
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
    
    def maybe_evaluate(self, request: dict, response: str) -> None:
        if random.random() > self.sample_rate:
            return
        
        # Async evaluation
        asyncio.create_task(self.evaluate_async(request, response))
    
    async def evaluate_async(self, request: dict, response: str):
        scores = await self.llm_judge(request["query"], response)
        
        # Log to monitoring system
        self.log_metrics({
            "correctness": scores["correctness"],
            "relevance": scores["relevance"],
            "timestamp": datetime.now()
        })
        
        # Alert on low scores
        if scores["overall"] < 3:
            self.alert_low_quality(request, response, scores)
```

### Drift Detection

```python
def detect_quality_drift(
    current_scores: list[float],
    baseline_scores: list[float],
    threshold: float = 0.1
) -> dict:
    current_mean = statistics.mean(current_scores)
    baseline_mean = statistics.mean(baseline_scores)
    
    drift = abs(current_mean - baseline_mean)
    is_significant = drift > threshold
    
    # Statistical test
    stat, p_value = stats.ttest_ind(current_scores, baseline_scores)
    
    return {
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift": drift,
        "is_significant": is_significant,
        "p_value": p_value
    }
```

---

## Interview Questions

### Q: How would you evaluate a RAG system?

**Strong answer:**
I would evaluate at multiple levels:

**1. Retrieval quality:**
- Precision@K: Are retrieved docs relevant?
- Recall@K: Did we find all relevant docs?
- MRR: Is the best doc ranked highly?

**2. Generation quality:**
- Faithfulness: Is response grounded in context?
- Relevance: Does it answer the question?
- Completeness: All aspects addressed?

**3. End-to-end:**
- Answer correctness vs ground truth
- User satisfaction (thumbs up/down)

**Tools:**
- RAGAS for automated metrics
- LLM-as-judge for subjective quality
- Human evaluation for gold standard

**Process:**
1. Create evaluation dataset (100+ examples)
2. Run automated metrics on every change
3. LLM judge for deeper analysis
4. Human review for final validation
5. Monitor in production continuously

### Q: What are the limitations of LLM-as-judge?

**Strong answer:**
Several known biases and limitations:

**Biases:**
- Position bias: Prefers first option in comparisons
- Length bias: Prefers longer responses
- Self-preference: May prefer own model's style
- Format bias: Influenced by formatting

**Mitigations:**
- Swap positions and check consistency
- Use different model as judge
- Calibrate with human annotations
- Multiple judge prompts

**When unreliable:**
- Highly domain-specific content
- Subtle factual errors
- Cultural/contextual nuances
- Safety edge cases

**Best practice:**
- Use for rapid iteration
- Calibrate against human judgments
- Do not rely solely on LLM judges
- Human review for high-stakes decisions

---

## References

- Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
- Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- RAGAS: https://docs.ragas.io/
- OpenAI Evals: https://github.com/openai/evals

---

*Next: [Observability](02-observability.md)*

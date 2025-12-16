# Prompt Optimization

Systematic approaches to improving prompt effectiveness through iteration, testing, and optimization.

## Table of Contents

- [Optimization Principles](#optimization-principles)
- [Iteration Workflow](#iteration-workflow)
- [Optimization Techniques](#optimization-techniques)
- [Testing and Evaluation](#testing-and-evaluation)
- [DSPy: Programmatic Optimization](#dspy-programmatic-optimization)
- [Prompt Versioning](#prompt-versioning)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Optimization Principles

### The Prompt Engineering Loop

```
┌─────────────────────────────────────────────────────────┐
│                PROMPT OPTIMIZATION LOOP                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Define  │───▶│  Develop │───▶│  Test    │         │
│   │ Criteria │    │  Prompt  │    │  Prompt  │         │
│   └──────────┘    └──────────┘    └────┬─────┘         │
│                                        │                │
│        ┌───────────────────────────────┤                │
│        │                               ▼                │
│   ┌────┴─────┐    ┌──────────┐    ┌──────────┐         │
│   │  Deploy  │◀───│  Select  │◀───│ Analyze  │         │
│   │          │    │  Best    │    │ Results  │         │
│   └──────────┘    └──────────┘    └──────────┘         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Metrics to Optimize

| Metric | Description | Measurement |
|--------|-------------|-------------|
| Accuracy | Correctness of output | Ground truth comparison |
| Relevance | Addresses the query | LLM-as-judge scoring |
| Format compliance | Follows output structure | Schema validation |
| Latency | Response time | Timing measurements |
| Cost | Token usage | Token counting |
| Safety | Avoids harmful content | Safety classifiers |

---

## Iteration Workflow

### Baseline Establishment

```python
class PromptOptimizer:
    def __init__(self, task: str, eval_dataset: list[dict]):
        self.task = task
        self.eval_dataset = eval_dataset
        self.prompt_versions = []
        self.results = []
    
    async def establish_baseline(self, initial_prompt: str) -> dict:
        version = {
            "id": "v0",
            "prompt": initial_prompt,
            "timestamp": datetime.now()
        }
        self.prompt_versions.append(version)
        
        # Run evaluation
        results = await self.evaluate(initial_prompt)
        self.results.append({
            "version": "v0",
            "metrics": results
        })
        
        return results
```

### Systematic Variation

```python
async def generate_variations(self, base_prompt: str) -> list[str]:
    """Generate prompt variations to test."""
    variations = []
    
    # Variation 1: Add explicit examples
    variations.append(self.add_examples(base_prompt))
    
    # Variation 2: Add structured output format
    variations.append(self.add_output_format(base_prompt))
    
    # Variation 3: Add reasoning instruction
    variations.append(self.add_chain_of_thought(base_prompt))
    
    # Variation 4: Simplify language
    variations.append(self.simplify(base_prompt))
    
    # Variation 5: Add constraints
    variations.append(self.add_constraints(base_prompt))
    
    return variations

def add_chain_of_thought(self, prompt: str) -> str:
    cot_prefix = """
Think through this step by step:
1. First, identify the key information
2. Then, analyze the requirements
3. Finally, provide your answer

"""
    return cot_prefix + prompt
```

---

## Optimization Techniques

### Technique 1: Few-Shot Example Selection

```python
class ExampleSelector:
    """Select the most effective examples for few-shot prompting."""
    
    def __init__(self, example_bank: list[dict]):
        self.example_bank = example_bank
        self.embeddings = self.embed_examples(example_bank)
    
    def select_similar(self, query: str, k: int = 3) -> list[dict]:
        """Select examples most similar to the query."""
        query_embedding = embed(query)
        similarities = [
            cosine_similarity(query_embedding, e)
            for e in self.embeddings
        ]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.example_bank[i] for i in top_indices]
    
    def select_diverse(self, k: int = 3) -> list[dict]:
        """Select diverse examples for broad coverage."""
        selected = [self.example_bank[0]]
        while len(selected) < k:
            # Find example most different from selected
            best_score = -1
            best_idx = None
            for i, ex in enumerate(self.example_bank):
                if ex in selected:
                    continue
                min_sim = min(
                    cosine_similarity(self.embeddings[i], self.embeddings[self.example_bank.index(s)])
                    for s in selected
                )
                if min_sim > best_score:
                    best_score = min_sim
                    best_idx = i
            selected.append(self.example_bank[best_idx])
        return selected
```

### Technique 2: Instruction Refinement

```python
class InstructionRefiner:
    """Iteratively refine instructions based on failure cases."""
    
    async def refine(self, prompt: str, failures: list[dict]) -> str:
        refine_prompt = f"""
Current prompt:
{prompt}

This prompt has failed on these cases:
{self.format_failures(failures)}

Analyze why these failures occurred and suggest improvements to the prompt.
Return the improved prompt.
"""
        
        improved = await self.llm.generate(refine_prompt)
        return improved
    
    def format_failures(self, failures: list[dict]) -> str:
        formatted = []
        for f in failures:
            formatted.append(f"""
Input: {f['input']}
Expected: {f['expected']}
Got: {f['actual']}
Issue: {f.get('analysis', 'Unknown')}
""")
        return "\n---\n".join(formatted)
```

### Technique 3: Output Format Optimization

```python
def optimize_output_format(task_type: str) -> str:
    """Generate appropriate output format instruction."""
    
    formats = {
        "classification": """
Output your answer in this exact format:
Category: [category name]
Confidence: [high/medium/low]
Reasoning: [one sentence explanation]
""",
        "extraction": """
Output as JSON:
{
    "field1": "extracted value",
    "field2": "extracted value"
}
""",
        "generation": """
Structure your response as:
## Summary
[Brief summary]

## Details
[Detailed explanation]

## Conclusion
[Key takeaway]
""",
        "qa": """
Answer the question directly, then provide supporting evidence.
Format:
Answer: [direct answer]
Evidence: [supporting facts from context]
"""
    }
    
    return formats.get(task_type, "")
```

---

## Testing and Evaluation

### A/B Testing Framework

```python
class PromptABTest:
    def __init__(self, control_prompt: str, treatment_prompt: str):
        self.control = control_prompt
        self.treatment = treatment_prompt
        self.results = {"control": [], "treatment": []}
    
    async def run_test(self, test_cases: list[dict], evaluator) -> dict:
        for case in test_cases:
            # Randomly assign to control or treatment
            if random.random() < 0.5:
                response = await self.llm.generate(
                    self.control + "\n" + case["input"]
                )
                score = await evaluator.score(case, response)
                self.results["control"].append(score)
            else:
                response = await self.llm.generate(
                    self.treatment + "\n" + case["input"]
                )
                score = await evaluator.score(case, response)
                self.results["treatment"].append(score)
        
        return self.analyze_results()
    
    def analyze_results(self) -> dict:
        control_mean = np.mean(self.results["control"])
        treatment_mean = np.mean(self.results["treatment"])
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            self.results["control"],
            self.results["treatment"]
        )
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "difference": treatment_mean - control_mean,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "recommendation": "treatment" if treatment_mean > control_mean and p_value < 0.05 else "control"
        }
```

### Regression Testing

```python
class PromptRegressionTests:
    """Ensure prompt changes do not break existing functionality."""
    
    def __init__(self, golden_examples: list[dict]):
        self.golden_examples = golden_examples
    
    async def run(self, prompt: str) -> dict:
        results = {
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for example in self.golden_examples:
            response = await self.llm.generate(
                prompt + "\n" + example["input"]
            )
            
            if self.matches_expected(response, example["expected"]):
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "input": example["input"],
                    "expected": example["expected"],
                    "actual": response
                })
        
        results["pass_rate"] = results["passed"] / len(self.golden_examples)
        return results
```

---

## DSPy: Programmatic Optimization

DSPy treats prompts as learnable parameters and optimizes them automatically.

### Basic DSPy Pattern

```python
import dspy

# Configure the LM
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)

# Define a signature (input -> output)
class QuestionAnswer(dspy.Signature):
    """Answer the question based on the context."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create a module
qa_module = dspy.ChainOfThought(QuestionAnswer)

# Use it
result = qa_module(
    context="The capital of France is Paris.",
    question="What is the capital of France?"
)
```

### Optimizing with DSPy

```python
from dspy.teleprompt import BootstrapFewShot

# Define a metric
def metric(example, prediction, trace=None):
    return prediction.answer.lower() == example.answer.lower()

# Create optimizer
optimizer = BootstrapFewShot(metric=metric)

# Training examples
trainset = [
    dspy.Example(
        context="Python was created by Guido van Rossum.",
        question="Who created Python?",
        answer="Guido van Rossum"
    ).with_inputs("context", "question"),
    # ... more examples
]

# Optimize
optimized_qa = optimizer.compile(qa_module, trainset=trainset)
```

---

## Prompt Versioning

### Version Control System

```python
class PromptVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.current_version = None
    
    def save_version(
        self,
        prompt: str,
        version: str,
        metadata: dict
    ):
        version_data = {
            "version": version,
            "prompt": prompt,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "created_by": os.getenv("USER")
        }
        
        path = f"{self.storage_path}/{version}.json"
        with open(path, "w") as f:
            json.dump(version_data, f, indent=2)
    
    def load_version(self, version: str) -> str:
        path = f"{self.storage_path}/{version}.json"
        with open(path, "r") as f:
            data = json.load(f)
        return data["prompt"]
    
    def diff(self, version_a: str, version_b: str) -> str:
        prompt_a = self.load_version(version_a)
        prompt_b = self.load_version(version_b)
        
        import difflib
        diff = difflib.unified_diff(
            prompt_a.splitlines(),
            prompt_b.splitlines(),
            fromfile=version_a,
            tofile=version_b,
            lineterm=""
        )
        return "\n".join(diff)
```

### Deployment Strategy

```python
class PromptDeployment:
    def __init__(self):
        self.active_version = None
        self.canary_version = None
        self.canary_percentage = 0
    
    def deploy_canary(self, version: str, percentage: int = 10):
        """Deploy new version to small percentage of traffic."""
        self.canary_version = version
        self.canary_percentage = percentage
    
    def promote_canary(self):
        """Promote canary to active."""
        self.active_version = self.canary_version
        self.canary_version = None
        self.canary_percentage = 0
    
    def rollback(self, version: str):
        """Roll back to a previous version."""
        self.canary_version = None
        self.canary_percentage = 0
        self.active_version = version
    
    def get_version_for_request(self) -> str:
        """Route request to appropriate version."""
        if self.canary_version and random.random() < self.canary_percentage / 100:
            return self.canary_version
        return self.active_version
```

---

## Interview Questions

### Q: How do you systematically improve prompt quality?

**Strong answer:**

"I follow a structured optimization loop:

**1. Define success criteria.** Before optimizing, I need to know what 'better' means. This could be accuracy, format compliance, tone, latency, or cost. I create an evaluation dataset with clear expected outputs.

**2. Establish baseline.** Run the current prompt on the evaluation set and measure all metrics. This is my baseline to beat.

**3. Generate variations.** I systematically modify one aspect at a time:
- Add examples (few-shot)
- Add chain-of-thought instruction
- Clarify output format
- Add constraints or guidelines
- Simplify language

**4. Test variations.** Run each variation on the same evaluation set. I look for statistically significant improvements.

**5. Analyze failures.** For cases where even the best prompt fails, I analyze why. Sometimes the issue is the evaluation, sometimes the prompt needs specific handling.

**6. Deploy with canary.** I do not flip 100% to a new prompt immediately. I run 10% traffic to validate in production.

The key is treating prompt engineering like software engineering: version control, testing, gradual rollout."

### Q: When would you use automated prompt optimization like DSPy?

**Strong answer:**

"DSPy is valuable when:

**Manual tuning hits limits.** If I have spent significant time hand-crafting prompts and quality has plateaued, automated optimization can find improvements I missed.

**I have good training data.** DSPy needs examples to optimize. Without labeled examples, it cannot learn what 'good' means.

**The task is stable.** If requirements change frequently, the optimization investment may not pay off.

**I want reproducibility.** DSPy produces deterministic optimized prompts that I can version control and deploy.

I would not use DSPy for simple tasks where basic prompting works, or for tasks without clear evaluation criteria. The overhead of setting up DSPy is not worth it if manual prompting gets 95% of the way there."

---

## References

- DSPy: https://dspy-docs.vercel.app/
- PromptBench: https://github.com/microsoft/promptbench
- Anthropic Prompt Engineering Guide: https://docs.anthropic.com/claude/docs/prompt-engineering

---

*Next: [Guardrails in Prompting](04-guardrails-and-safety.md)*

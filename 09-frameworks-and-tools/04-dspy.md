# DSPy

DSPy is a framework for programming language models through composable, optimizable modules rather than prompt engineering. It treats prompts as parameters to be learned.

## Table of Contents

- [DSPy Philosophy](#dspy-philosophy)
- [Core Concepts](#core-concepts)
- [Signatures](#signatures)
- [Modules](#modules)
- [Optimizers](#optimizers)
- [Evaluation](#evaluation)
- [When to Use DSPy](#when-to-use-dspy)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## DSPy Philosophy

### The Problem with Prompt Engineering

```
Traditional Approach:
"Write a good prompt" ──► Test ──► Tweak ──► Test ──► Deploy
                         ↑                      │
                         └──────────────────────┘
                              Manual iteration
```

### DSPy Approach

```
DSPy Approach:
Define task ──► Define metrics ──► Let optimizer find best prompts
                                         │
                                         ▼
                                  Optimized pipeline
```

### Key Insight

| Traditional | DSPy |
|-------------|------|
| Prompts are written | Prompts are compiled |
| Manual optimization | Automated optimization |
| Brittle to model changes | Adapts to model |
| Example engineering | Metric-driven |

---

## Core Concepts

### Setting Up

```python
import dspy

# Configure LLM
lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
dspy.configure(lm=lm)

# Or use different providers
# lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022")
# lm = dspy.LM("ollama_chat/llama3.1")
```

### Basic Prediction

```python
# Simple question answering
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

---

## Signatures

### What Signatures Are

Signatures define the input/output specification of a task:

```python
# Simple signature (string form)
"question -> answer"
"document -> summary"
"text, query -> relevant: bool"

# Class-based signature (more control)
class SummarizeSignature(dspy.Signature):
    """Summarize the document in 2-3 sentences."""
    
    document: str = dspy.InputField(desc="The document to summarize")
    summary: str = dspy.OutputField(desc="A 2-3 sentence summary")
```

### Signature Components

```python
class AnswerQuestion(dspy.Signature):
    """Answer the question based on the context."""
    
    # Input fields
    context: str = dspy.InputField(desc="Relevant background information")
    question: str = dspy.InputField(desc="The question to answer")
    
    # Output fields
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    answer: str = dspy.OutputField(desc="The final answer")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
```

---

## Modules

### Built-in Modules

```python
# Predict: Single LLM call
predict = dspy.Predict(SummarizeSignature)
result = predict(document="Long text...")

# ChainOfThought: Adds reasoning step
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 15% of 80?")
print(result.reasoning)  # Shows step-by-step
print(result.answer)     # "12"

# ReAct: Reasoning + Acting (tool use)
react = dspy.ReAct("question -> answer", tools=[search, calculate])
```

### Custom Modules

```python
class RAGModule(dspy.Module):
    def __init__(self, retriever, k=3):
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # Retrieve
        context = self.retriever(question, k=self.k)
        
        # Generate
        result = self.generate(
            context=context,
            question=question
        )
        
        return result


# Use the module
rag = RAGModule(retriever=my_retriever, k=5)
answer = rag(question="What is DSPy?")
```

### Composing Modules

```python
class MultiHopQA(dspy.Module):
    def __init__(self, retriever, max_hops=3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            # Generate search query
            query_result = self.generate_query(
                context="\n".join(context),
                question=question
            )
            
            # Retrieve
            new_context = self.retriever(query_result.search_query)
            context.append(new_context)
        
        # Final answer
        return self.generate_answer(
            context="\n".join(context),
            question=question
        )
```

---

## Optimizers

### What Optimizers Do

Optimizers find the best prompts for your modules by:
1. Trying different prompt variations
2. Evaluating on your metric
3. Selecting best-performing prompts

### Available Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `BootstrapFewShot` | Learns from examples | Small data |
| `BootstrapFewShotWithRandomSearch` | + random search | More exploration |
| `MIPRO` | Multi-stage optimization | Complex pipelines |
| `BootstrapFinetune` | Fine-tunes the model | When you can fine-tune |

### Using Optimizers

```python
from dspy.teleprompt import BootstrapFewShot

# Define your module
rag = RAGModule(retriever)

# Define training data
trainset = [
    dspy.Example(question="What is Python?", answer="A programming language"),
    dspy.Example(question="Who created Linux?", answer="Linus Torvalds"),
    # ... more examples
]

# Define metric
def metric(example, prediction, trace=None):
    # Return True if prediction is correct
    return example.answer.lower() in prediction.answer.lower()

# Optimize
optimizer = BootstrapFewShot(metric=metric)
optimized_rag = optimizer.compile(rag, trainset=trainset)

# Use optimized module
result = optimized_rag(question="What is GitHub?")
```

### MIPRO for Complex Pipelines

```python
from dspy.teleprompt import MIPRO

# MIPRO optimizes both instructions and few-shot examples
optimizer = MIPRO(
    metric=metric,
    prompt_model=dspy.LM("openai/gpt-4o"),  # Model for generating prompts
    task_model=dspy.LM("openai/gpt-4o-mini"),  # Model for task
    num_candidates=10,
    init_temperature=1.0
)

optimized_module = optimizer.compile(
    module,
    trainset=trainset,
    num_batches=10,
    max_bootstrapped_demos=3,
    max_labeled_demos=5
)
```

---

## Evaluation

### Creating Evaluators

```python
from dspy.evaluate import Evaluate

# Define metric
def answer_correctness(example, prediction, trace=None):
    # Exact match
    return example.answer == prediction.answer

def answer_similarity(example, prediction, trace=None):
    # Semantic similarity
    from sklearn.metrics.pairwise import cosine_similarity
    emb1 = embed(example.answer)
    emb2 = embed(prediction.answer)
    return cosine_similarity([emb1], [emb2])[0][0] > 0.8

# LLM-based metric
class SemanticMatch(dspy.Signature):
    """Judge if the prediction matches the expected answer semantically."""
    expected: str = dspy.InputField()
    prediction: str = dspy.InputField()
    matches: bool = dspy.OutputField()

semantic_judge = dspy.Predict(SemanticMatch)

def semantic_metric(example, prediction, trace=None):
    result = semantic_judge(
        expected=example.answer,
        prediction=prediction.answer
    )
    return result.matches
```

### Running Evaluation

```python
# Create evaluator
evaluator = Evaluate(
    devset=dev_examples,
    metric=answer_correctness,
    num_threads=4,
    display_progress=True
)

# Evaluate
score = evaluator(module)
print(f"Accuracy: {score}%")
```

---

## When to Use DSPy

### Use DSPy When

| Scenario | Why |
|----------|-----|
| Complex pipelines | Optimize multi-step chains |
| Quality focus | Metric-driven optimization |
| Model changes frequent | Prompts adapt automatically |
| Have training data | Optimizers need examples |
| Research/experimentation | Easy to iterate |

### Skip DSPy When

| Scenario | Why |
|----------|-----|
| Simple use case | Overhead not worth it |
| No training data | Optimizers need examples |
| Need full control | Abstraction hides details |
| Production simplicity | Additional complexity |

### DSPy vs Others

| Framework | Focus |
|-----------|-------|
| **DSPy** | Optimizing prompts |
| **LangChain** | Orchestrating chains |
| **LlamaIndex** | Document retrieval |
| **Direct API** | Maximum control |

---

## Interview Questions

### Q: How does DSPy differ from traditional prompt engineering?

**Strong answer:**

"DSPy treats prompts as parameters to be optimized rather than hand-crafted strings.

**Traditional approach:**
1. Write a prompt
2. Test manually
3. Tweak wording
4. Repeat until good enough
5. Hope it works when model updates

**DSPy approach:**
1. Define task signature (inputs/outputs)
2. Define evaluation metric
3. Provide training examples
4. Let optimizer find best prompts
5. Prompts can re-optimize for new models

**Key benefits:**
- Reproducible optimization
- Adapts to model changes
- Metric-driven rather than intuition-driven
- Composable modules

I use DSPy when I have training data and need consistent quality across pipeline changes. For simple one-off prompts, direct API is simpler."

### Q: Explain DSPy optimizers.

**Strong answer:**

"DSPy optimizers automatically find effective prompts by:

1. **Generating candidates:** Try different prompt variations, few-shot examples
2. **Evaluating:** Score each candidate on your metric
3. **Selecting:** Keep the best-performing configuration

**Main optimizers:**

**BootstrapFewShot:** Samples successful examples from training data to use as few-shot demonstrations. Good starting point.

**BootstrapFewShotWithRandomSearch:** Adds random exploration to find better example combinations.

**MIPRO:** Optimizes both instructions and examples. Uses a stronger model to generate prompt variations, tests on task model.

**Example workflow:**
```python
optimizer = BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(module, trainset=examples)
```

The optimizer runs the module on training examples, identifies which produce correct outputs, and uses those as demonstrations in the final prompt."

---

## References

- DSPy Docs: https://dspy-docs.vercel.app/
- DSPy GitHub: https://github.com/stanfordnlp/dspy

---

*Previous: [LlamaIndex](03-llamaindex.md)*

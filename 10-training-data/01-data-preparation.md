# Training Data Preparation

Quality training data is the foundation of successful fine-tuning. This chapter covers data collection, curation, and preparation.

## Table of Contents

- [Data Quality Principles](#data-quality-principles)
- [Data Collection Strategies](#data-collection-strategies)
- [Data Formatting](#data-formatting)
- [Data Cleaning](#data-cleaning)
- [Data Augmentation](#data-augmentation)
- [Validation and Splitting](#validation-and-splitting)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Data Quality Principles

### The 100-Example Rule

Start with 100 high-quality examples before collecting more:

```
100 Examples
    │
    ├── Train initial model
    │
    ├── Evaluate on held-out set
    │
    ├── Identify failure modes
    │
    ├── Collect targeted data for failures
    │
    └── Iterate
```

### Quality Over Quantity

| 500 Perfect Examples | > | 5000 Noisy Examples |
|----------------------|---|---------------------|

Quality indicators:
- Correct outputs
- Consistent format
- Representative of real distribution
- Diverse (not repetitive)
- Clear and unambiguous

### Common Data Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| Incorrect labels | Model learns wrong behavior | Expert review |
| Inconsistent format | Unstable outputs | Format standardization |
| Duplicates | Overfitting to repeated examples | Deduplication |
| Biased distribution | Poor generalization | Balanced sampling |
| Low diversity | Narrow capability | Active collection |

---

## Data Collection Strategies

### From Production Logs

```python
class ProductionDataCollector:
    def __init__(self, quality_threshold: float = 4.0):
        self.quality_threshold = quality_threshold
    
    async def collect_examples(self) -> list[dict]:
        examples = []
        
        # Get interactions with positive feedback
        positive_interactions = await self.db.query("""
            SELECT input, output, user_rating
            FROM interactions
            WHERE user_rating >= ?
            AND created_at > NOW() - INTERVAL 30 DAY
        """, [self.quality_threshold])
        
        for interaction in positive_interactions:
            examples.append({
                "messages": [
                    {"role": "user", "content": interaction["input"]},
                    {"role": "assistant", "content": interaction["output"]}
                ],
                "quality_score": interaction["user_rating"]
            })
        
        return examples
```

### Human Annotation

```python
class AnnotationPipeline:
    def __init__(self):
        self.annotators = []
        self.min_agreement = 0.8
    
    async def create_task(self, inputs: list[str]) -> AnnotationTask:
        return AnnotationTask(
            inputs=inputs,
            instructions=self.get_instructions(),
            examples=self.get_examples(),
            num_annotators_per_item=3
        )
    
    async def collect_annotations(self, task: AnnotationTask) -> list[dict]:
        annotations = await self.annotation_platform.run(task)
        
        # Filter by agreement
        agreed = []
        for item in annotations:
            if self.calculate_agreement(item) >= self.min_agreement:
                agreed.append({
                    "input": item["input"],
                    "output": self.resolve_annotation(item["annotations"])
                })
        
        return agreed
    
    def calculate_agreement(self, item: dict) -> float:
        annotations = item["annotations"]
        # For categorical: majority percentage
        # For text: semantic similarity average
        return self.compute_similarity(annotations)
```

### Synthetic Data Generation

```python
class SyntheticDataGenerator:
    """
    Use a stronger model to generate training data.
    """
    
    def __init__(self, teacher_model: str = "gpt-4o"):
        self.teacher = teacher_model
    
    async def generate_examples(
        self,
        task_description: str,
        num_examples: int,
        seed_examples: list[dict]
    ) -> list[dict]:
        prompt = f"""
Generate {num_examples} training examples for this task:
{task_description}

Here are some example input-output pairs:
{self.format_examples(seed_examples)}

Generate diverse examples covering different scenarios.
Return as JSON array with "input" and "output" fields.
"""
        
        response = await self.llm.generate(prompt, model=self.teacher)
        examples = json.loads(response)
        
        # Validate and filter
        validated = []
        for ex in examples:
            if await self.validate_example(ex, task_description):
                validated.append(ex)
        
        return validated
    
    async def validate_example(self, example: dict, task: str) -> bool:
        # Check if output is correct for input
        validation_prompt = f"""
Task: {task}
Input: {example['input']}
Proposed output: {example['output']}

Is this output correct and high quality? Return "yes" or "no" with reason.
"""
        result = await self.llm.generate(validation_prompt)
        return "yes" in result.lower()
```

---

## Data Formatting

### OpenAI Format

```python
def format_for_openai(examples: list[dict]) -> list[dict]:
    formatted = []
    
    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "system", "content": ex.get("system", "")},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]}
            ]
        })
    
    return formatted

# Save as JSONL
def save_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
```

### Multi-Turn Conversations

```python
def format_conversation(conversation: list[dict]) -> dict:
    messages = []
    
    for turn in conversation:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })
    
    return {"messages": messages}

# Example
conversation = {
    "messages": [
        {"role": "system", "content": "You are a helpful SQL assistant."},
        {"role": "user", "content": "Show me all users"},
        {"role": "assistant", "content": "SELECT * FROM users"},
        {"role": "user", "content": "Only active ones"},
        {"role": "assistant", "content": "SELECT * FROM users WHERE status = 'active'"}
    ]
}
```

### Tool Use Training

```python
def format_tool_call(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": example["tool_name"],
                        "arguments": json.dumps(example["tool_args"])
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": example["tool_result"]
            },
            {"role": "assistant", "content": example["final_response"]}
        ]
    }
```

---

## Data Cleaning

### Deduplication

```python
class DataDeduplicator:
    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
    
    async def deduplicate(self, examples: list[dict]) -> list[dict]:
        # Embed all examples
        embeddings = await self.embed_batch([
            e["input"] + e["output"] for e in examples
        ])
        
        # Find duplicates
        unique = []
        unique_embeddings = []
        
        for i, (example, embedding) in enumerate(zip(examples, embeddings)):
            is_duplicate = False
            
            for unique_emb in unique_embeddings:
                similarity = cosine_similarity(embedding, unique_emb)
                if similarity > self.threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(example)
                unique_embeddings.append(embedding)
        
        return unique
```

### Quality Filtering

```python
class QualityFilter:
    def __init__(self):
        self.min_length = 10
        self.max_length = 10000
    
    async def filter(self, examples: list[dict]) -> list[dict]:
        filtered = []
        
        for ex in examples:
            if not self.passes_basic_checks(ex):
                continue
            
            if not await self.passes_quality_check(ex):
                continue
            
            filtered.append(ex)
        
        return filtered
    
    def passes_basic_checks(self, example: dict) -> bool:
        # Length checks
        if len(example["input"]) < self.min_length:
            return False
        if len(example["output"]) < self.min_length:
            return False
        if len(example["input"]) + len(example["output"]) > self.max_length:
            return False
        
        # Completeness
        if not example["input"].strip() or not example["output"].strip():
            return False
        
        return True
    
    async def passes_quality_check(self, example: dict) -> bool:
        # Use LLM to check quality
        prompt = f"""
Is this a high-quality training example?

Input: {example['input']}
Output: {example['output']}

Rate quality 1-5 and explain briefly.
"""
        result = await self.llm.generate(prompt)
        score = self.extract_score(result)
        return score >= 4
```

### Format Standardization

```python
class FormatStandardizer:
    def standardize(self, examples: list[dict]) -> list[dict]:
        standardized = []
        
        for ex in examples:
            std = {
                "input": self.clean_text(ex["input"]),
                "output": self.standardize_output(ex["output"])
            }
            standardized.append(std)
        
        return standardized
    
    def clean_text(self, text: str) -> str:
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Remove invalid characters
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        
        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        
        return text
    
    def standardize_output(self, output: str) -> str:
        output = self.clean_text(output)
        
        # Task-specific formatting
        # e.g., ensure JSON is valid, SQL is properly formatted
        
        return output
```

---

## Data Augmentation

### Paraphrasing

```python
class Paraphraser:
    async def augment(self, examples: list[dict], factor: int = 2) -> list[dict]:
        augmented = examples.copy()
        
        for ex in examples:
            for _ in range(factor - 1):
                paraphrased_input = await self.paraphrase(ex["input"])
                augmented.append({
                    "input": paraphrased_input,
                    "output": ex["output"]
                })
        
        return augmented
    
    async def paraphrase(self, text: str) -> str:
        prompt = f"""
Paraphrase this text while preserving the exact meaning:
{text}

Return only the paraphrased version.
"""
        return await self.llm.generate(prompt)
```

### Difficulty Variations

```python
class DifficultyAugmenter:
    async def augment(self, examples: list[dict]) -> list[dict]:
        augmented = examples.copy()
        
        for ex in examples:
            # Simpler version
            simpler = await self.make_simpler(ex)
            if simpler:
                augmented.append(simpler)
            
            # Harder version
            harder = await self.make_harder(ex)
            if harder:
                augmented.append(harder)
        
        return augmented
    
    async def make_harder(self, example: dict) -> dict:
        prompt = f"""
Create a harder version of this example by adding complexity:

Input: {example['input']}
Output: {example['output']}

Make the input more complex while keeping the same type of task.
Return as JSON with "input" and "output".
"""
        result = await self.llm.generate(prompt)
        return json.loads(result)
```

---

## Validation and Splitting

### Train/Validation Split

```python
class DataSplitter:
    def split(
        self,
        examples: list[dict],
        val_ratio: float = 0.1,
        stratify_by: str = None
    ) -> tuple[list[dict], list[dict]]:
        if stratify_by:
            return self.stratified_split(examples, val_ratio, stratify_by)
        
        # Simple random split
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - val_ratio))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]
        
        return train, val
    
    def stratified_split(
        self,
        examples: list[dict],
        val_ratio: float,
        stratify_by: str
    ) -> tuple[list[dict], list[dict]]:
        # Group by stratification key
        groups = defaultdict(list)
        for ex in examples:
            key = ex.get(stratify_by, "default")
            groups[key].append(ex)
        
        train, val = [], []
        
        # Split each group proportionally
        for group_examples in groups.values():
            g_train, g_val = self.split(group_examples, val_ratio)
            train.extend(g_train)
            val.extend(g_val)
        
        return train, val
```

### Validation Metrics

```python
class TrainingDataValidator:
    def validate(self, train: list[dict], val: list[dict]) -> dict:
        metrics = {}
        
        # Size checks
        metrics["train_size"] = len(train)
        metrics["val_size"] = len(val)
        metrics["val_ratio"] = len(val) / (len(train) + len(val))
        
        # Diversity metrics
        metrics["train_diversity"] = self.measure_diversity(train)
        metrics["val_diversity"] = self.measure_diversity(val)
        
        # Distribution overlap
        metrics["distribution_overlap"] = self.measure_overlap(train, val)
        
        # Recommendations
        metrics["warnings"] = []
        
        if len(train) < 100:
            metrics["warnings"].append("Training set may be too small")
        
        if metrics["distribution_overlap"] < 0.5:
            metrics["warnings"].append("Validation set may not be representative")
        
        return metrics
```

---

## Interview Questions

### Q: How do you prepare training data for fine-tuning?

**Strong answer:**

"My preparation process:

**1. Collection:** Start from production data with positive signals (user ratings, successful completions) or create annotation tasks with clear guidelines.

**2. Quality filtering:** Remove duplicates, short/incomplete examples, incorrect outputs. I use both heuristics (length, format) and LLM-based quality scoring.

**3. Format standardization:** Ensure consistent format across all examples. This is critical since format inconsistency confuses the model.

**4. Diversity check:** Ensure examples cover the input distribution. If I have 500 examples of one category and 10 of another, the model will be biased.

**5. Validation split:** Hold out 10% for evaluation. Critical to never train on validation data.

**6. Iterative refinement:** After initial training, analyze failures on validation set. Collect more examples for failure modes.

Quality matters more than quantity. 500 perfect examples beat 5000 noisy ones."

### Q: How do you handle data imbalance in training data?

**Strong answer:**

"Several strategies:

**Upsampling minority classes:** Repeat minority examples to balance distribution. Risk: overfitting on repeated examples.

**Downsampling majority classes:** Reduce majority examples. Risk: losing valuable data.

**Weighted training:** If supported, assign higher weights to minority examples.

**Synthetic data generation:** Use a stronger model to generate more examples for underrepresented categories.

**Curriculum learning:** Start with balanced subset, gradually add more majority examples.

My preference: combine synthetic generation with slight upsampling. This maintains diversity while addressing imbalance.

I also monitor per-category performance during training to catch balancing issues early."

---

## References

- OpenAI Data Preparation: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
- Anthropic Fine-Tuning: https://docs.anthropic.com/

---

*Next: [Fine-Tuning Process](03-fine-tuning-process.md)*

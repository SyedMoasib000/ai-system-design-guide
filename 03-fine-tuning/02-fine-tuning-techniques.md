# Fine-Tuning Techniques

This chapter covers the technical aspects of fine-tuning LLMs, from full fine-tuning to parameter-efficient methods.

## Table of Contents

- [Fine-Tuning Approaches](#fine-tuning-approaches)
- [Full Fine-Tuning](#full-fine-tuning)
- [LoRA and QLoRA](#lora-and-qlora)
- [Training Configuration](#training-configuration)
- [Evaluation During Training](#evaluation-during-training)
- [Deployment](#deployment)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Fine-Tuning Approaches

### Comparison

| Method | Parameters Updated | Memory | Quality | Cost |
|--------|-------------------|--------|---------|------|
| Full Fine-Tuning | All | Very High | Highest | $$$ |
| LoRA | Adapter only (0.1-1%) | Low | Very Good | $ |
| QLoRA | Adapter on quantized | Very Low | Good | $ |
| Prefix Tuning | Prefix embeddings | Low | Good | $ |
| Prompt Tuning | Soft prompts | Minimal | Moderate | $ |

### When to Use Each

```
Need maximum quality + have resources? ──► Full Fine-Tuning

Need good quality + limited GPU? ──► LoRA

Running on consumer hardware? ──► QLoRA

Quick experimentation? ──► LoRA or Prompt Tuning
```

---

## Full Fine-Tuning

### When to Use

- Maximum quality required
- Sufficient compute resources (multiple A100/H100 GPUs)
- Large, high-quality training dataset (10K+ examples)
- Will deploy at scale (amortize training cost)

### Implementation

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

def full_fine_tune(
    base_model: str,
    train_dataset,
    val_dataset,
    output_dir: str
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        bf16=True,
        gradient_checkpointing=True
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    trainer.save_model(output_dir)
```

### Resource Requirements

| Model Size | GPU Memory (FP16) | Recommended Setup |
|------------|-------------------|-------------------|
| 7B | ~28 GB | 1x A100 80GB |
| 13B | ~52 GB | 2x A100 80GB |
| 70B | ~280 GB | 8x A100 80GB |

---

## LoRA and QLoRA

### How LoRA Works

```
Original weights: W (frozen)
LoRA adapters: A, B (trainable)

Forward pass: output = W @ x + (A @ B) @ x

A: (input_dim, rank)  - Low rank decomposition
B: (rank, output_dim) - Low rank decomposition
rank << min(input_dim, output_dim)
```

### LoRA Implementation

```python
from peft import LoraConfig, get_peft_model, TaskType

def lora_fine_tune(
    base_model: str,
    train_dataset,
    val_dataset,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 32
):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # Also consider: "gate_proj", "up_proj", "down_proj" for MLP
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0622
    
    # Training (same as full fine-tuning)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Can use larger batch with LoRA
        learning_rate=2e-4,  # Higher LR for LoRA
        # ... other args
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
```

### QLoRA Implementation

```python
from transformers import BitsAndBytesConfig

def qlora_fine_tune(
    base_model: str,
    train_dataset,
    output_dir: str
):
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Apply LoRA on quantized model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Train as usual
    # ...
```

### LoRA Configuration Guide

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| `rank (r)` | 8-64 | Higher = more capacity, more memory |
| `alpha` | 16-64 | Scaling factor, typically 2x rank |
| `dropout` | 0.05-0.1 | Regularization |
| `target_modules` | attention layers | Which weights get adapters |

---

## Training Configuration

### Hyperparameter Selection

```python
# Conservative starting point
SAFE_HYPERPARAMS = {
    "learning_rate": 2e-5,  # Full FT: 1e-5 to 5e-5, LoRA: 1e-4 to 3e-4
    "batch_size": 8,        # As large as memory allows
    "epochs": 3,            # Start with 3, monitor for overfitting
    "warmup_ratio": 0.1,    # 10% of steps for warmup
    "weight_decay": 0.01,   # Regularization
    "max_grad_norm": 1.0    # Gradient clipping
}

# For small datasets (<1000 examples)
SMALL_DATA_HYPERPARAMS = {
    "learning_rate": 1e-5,  # Lower to prevent overfitting
    "epochs": 1,            # Fewer epochs
    "weight_decay": 0.1     # Stronger regularization
}
```

### Learning Rate Schedules

```python
from transformers import get_scheduler

# Cosine with warmup (recommended)
scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Linear decay
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

## Evaluation During Training

### Metrics to Track

```python
class FineTuningEvaluator:
    def __init__(self, val_dataset, task_type: str):
        self.val_dataset = val_dataset
        self.task_type = task_type
    
    def evaluate(self, model, tokenizer) -> dict:
        metrics = {}
        
        # Loss on validation set
        metrics["val_loss"] = self.compute_loss(model, tokenizer)
        
        # Task-specific metrics
        if self.task_type == "classification":
            metrics["accuracy"] = self.compute_accuracy(model, tokenizer)
            metrics["f1"] = self.compute_f1(model, tokenizer)
        
        elif self.task_type == "generation":
            metrics["bleu"] = self.compute_bleu(model, tokenizer)
            metrics["exact_match"] = self.compute_exact_match(model, tokenizer)
        
        # Perplexity
        metrics["perplexity"] = torch.exp(torch.tensor(metrics["val_loss"]))
        
        return metrics
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
    ]
)
```

### Overfitting Detection

```python
class OverfitDetector:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_val_loss = float("inf")
        self.counter = 0
    
    def check(self, train_loss: float, val_loss: float) -> dict:
        result = {
            "overfit_warning": False,
            "should_stop": False
        }
        
        # Gap between train and val loss
        gap = val_loss - train_loss
        if gap > 0.5:
            result["overfit_warning"] = True
        
        # Val loss increasing
        if val_loss > self.best_val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                result["should_stop"] = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0
        
        return result
```

---

## Deployment

### Merging LoRA Weights

```python
from peft import PeftModel

def merge_and_save(base_model_path: str, lora_path: str, output_path: str):
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
```

### Serving Fine-Tuned Models

```python
# Option 1: Merged model (simpler, larger size)
model = AutoModelForCausalLM.from_pretrained("merged_model_path")

# Option 2: Base + LoRA (smaller, load adapter at runtime)
base_model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base_model, "lora_adapter_path")

# Option 3: Multiple adapters (switch by task)
base_model = AutoModelForCausalLM.from_pretrained("base_model")

adapters = {
    "task_a": "lora_task_a_path",
    "task_b": "lora_task_b_path"
}

def load_for_task(task: str):
    return PeftModel.from_pretrained(base_model, adapters[task])
```

---

## Interview Questions

### Q: Explain LoRA and when you would use it.

**Strong answer:**

"LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all model weights, it adds small trainable matrices to specific layers.

**How it works:**
For a weight matrix W, LoRA adds a low-rank decomposition A @ B where A and B have small inner dimension (rank). During forward pass: output = W @ x + (A @ B) @ x.

**Benefits:**
- Updates only ~0.1% of parameters
- Memory efficient (single GPU for 7B+ models)
- Fast training
- Small adapter files (~50MB vs 14GB full model)
- Can swap adapters for different tasks

**When I use LoRA:**
- Limited GPU memory
- Quick iteration needed
- Multiple tasks from same base model
- Quality requirements not extreme

**When I use full fine-tuning:**
- Maximum quality needed
- Large compute budget available
- Single production task

In most production scenarios, LoRA achieves 95% of full fine-tuning quality with 10% of the cost."

### Q: How do you prevent overfitting during fine-tuning?

**Strong answer:**

"Several strategies:

**Data side:**
- More diverse training data
- Data augmentation (paraphrasing)
- Ensure no train/val leakage

**Training side:**
- Lower learning rate
- Fewer epochs
- Higher weight decay
- Dropout (LoRA has lora_dropout)
- Early stopping based on validation loss

**Architecture side:**
- Lower LoRA rank (less capacity)
- Fewer target modules

**Monitoring:**
- Track train vs val loss gap
- Stop if val loss increases for 2-3 checkpoints
- Compare to base model on held-out set

The clearest overfitting signal is val loss increasing while train loss decreases. I checkpoint frequently and always restore the best checkpoint, not the last one."

---

## References

- LoRA Paper: https://arxiv.org/abs/2106.09685
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- PEFT Library: https://huggingface.co/docs/peft

---

*Next: [Evaluation After Fine-Tuning](04-evaluation.md)*

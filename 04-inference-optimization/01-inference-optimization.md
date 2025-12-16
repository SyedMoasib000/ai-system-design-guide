# Inference Optimization

Optimizing LLM inference is critical for production systems. This chapter covers quantization, batching, caching, and other techniques to improve latency, throughput, and cost.

## Table of Contents

- [Inference Bottlenecks](#inference-bottlenecks)
- [Quantization](#quantization)
- [KV Cache Optimization](#kv-cache-optimization)
- [Batching Strategies](#batching-strategies)
- [Speculative Decoding](#speculative-decoding)
- [Serving Frameworks](#serving-frameworks)
- [Hardware Selection](#hardware-selection)
- [Optimization Checklist](#optimization-checklist)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Inference Bottlenecks

### Understanding the Two Phases

LLM inference has distinct phases with different bottlenecks:

| Phase | Bottleneck | Characteristic |
|-------|------------|----------------|
| Prefill | Compute-bound | Process all input tokens in parallel |
| Decode | Memory-bound | Generate tokens one at a time |

### Prefill Phase

- Processes entire prompt at once
- Matrix multiplications dominate
- Scales with input length
- GPU compute utilization is high

### Decode Phase

- Generates one token at a time
- Loads KV cache from memory each step
- Memory bandwidth is the bottleneck
- GPU compute often underutilized

```
Memory Bandwidth Bottleneck:

For 70B model at FP16:
- KV cache load per token: ~2.6 MB
- At 100 tokens/sec: 260 MB/s just for KV cache
- Plus model weights access

H100 bandwidth: 3.35 TB/s
A100 bandwidth: 2.0 TB/s

Decode is often at 10-30% compute utilization
because waiting for memory.
```

---

## Quantization

### What Is Quantization

Reduce precision of model weights to use less memory and compute faster.

| Precision | Bits | Memory | Speed | Quality |
|-----------|------|--------|-------|---------|
| FP32 | 32 | 4x baseline | Slowest | Baseline |
| FP16 | 16 | 2x baseline | Fast | ~Same |
| BF16 | 16 | 2x baseline | Fast | ~Same |
| INT8 | 8 | 1x baseline | Faster | 99%+ |
| INT4 | 4 | 0.5x baseline | Fastest | 95-99% |

### Quantization Methods

**Post-Training Quantization (PTQ):**
Quantize after training, no retraining needed.

```python
# Using bitsandbytes for 4-bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"  # Normalized float 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Quantization-Aware Training (QAT):**
Train with quantization in mind for better quality.

### Popular Quantization Formats

| Format | Method | Quality | Speed | Notes |
|--------|--------|---------|-------|-------|
| GPTQ | PTQ, weight-only | Good | Fast | Popular for 4-bit |
| AWQ | PTQ, activation-aware | Better | Fast | Better quality than GPTQ |
| GGUF | CPU-optimized | Good | Medium | For llama.cpp |
| bitsandbytes | Dynamic | Good | Medium | Easy to use |
| FP8 | Native on H100 | Excellent | Very fast | Newest hardware |

### Quantization Impact

```python
# Memory savings for Llama 70B
memory_requirements = {
    "FP16": 140,   # GB
    "INT8": 70,    # GB
    "INT4": 35,    # GB
    "GPTQ 4-bit": 35,  # GB
    "AWQ 4-bit": 35,   # GB
}

# Quality impact (approximate)
quality_retention = {
    "FP16": 100,    # % of FP32 quality
    "INT8": 99.5,   # %
    "INT4": 97,     # %
    "GPTQ 4-bit": 96,  # %
    "AWQ 4-bit": 98,   # %
}
```

---

## KV Cache Optimization

### KV Cache Basics

During generation, Key and Value tensors from attention are cached to avoid recomputation.

```
KV Cache size per token:
= 2 (K and V) × layers × kv_heads × head_dim × bytes

Llama 70B (GQA, 8 KV heads):
= 2 × 80 × 8 × 128 × 2 (FP16)
= 327 KB per token

At 8K context: 2.6 GB per request
```

### Grouped Query Attention (GQA)

Share KV heads across multiple query heads:

```
Multi-Head Attention:  64 query heads, 64 KV heads
Grouped Query (8:1):   64 query heads, 8 KV heads

KV cache reduction: 8x
Quality impact: Minimal (<1%)
```

### Multi-Query Attention (MQA)

Extreme sharing: one KV head for all queries:

```
MQA: 64 query heads, 1 KV head

KV cache reduction: 64x
Quality impact: Noticeable (1-3%)
```

### PagedAttention (vLLM)

Virtual memory for KV cache:

```python
# Traditional: Pre-allocate max sequence length
cache = torch.zeros(batch_size, max_seq_len, kv_size)
# Wastes memory for short sequences

# PagedAttention: Allocate pages as needed
pages = []
for token in sequence:
    if len(current_page) == page_size:
        pages.append(allocate_page())
    current_page.append(token_kv)
```

**Benefits:**
- Near-zero memory waste
- Enables larger batch sizes
- Better GPU utilization
- 2-4x throughput improvement

### KV Cache Quantization

Quantize the cache itself:

```python
# FP16 KV cache (default)
cache_memory = 2.6  # GB at 8K context

# INT8 KV cache
cache_memory = 1.3  # GB (50% reduction)

# FP8 KV cache (H100)
cache_memory = 1.3  # GB, faster compute
```

Quality impact is usually minimal for INT8 KV cache.

---

## Batching Strategies

### Static Batching

Fixed batch processed together:

```python
# All requests must wait for longest
batch = collect_requests(count=32)
outputs = model.generate(batch, max_length=max(lengths))
```

**Problems:**
- Short requests wait for long ones
- Inefficient GPU utilization
- High latency variance

### Continuous Batching

Add/remove requests dynamically:

```python
class ContinuousBatcher:
    def __init__(self, max_batch_size: int = 32):
        self.active_requests = []
        self.waiting_queue = []
    
    def step(self):
        # Remove completed requests
        self.active_requests = [
            r for r in self.active_requests 
            if not r.is_complete()
        ]
        
        # Add waiting requests to fill batch
        while (len(self.active_requests) < self.max_batch_size 
               and self.waiting_queue):
            self.active_requests.append(self.waiting_queue.pop(0))
        
        # Run one decode step for all active
        if self.active_requests:
            self.decode_step(self.active_requests)
```

**Benefits:**
- Requests finish as soon as done
- Better GPU utilization
- More consistent latency

### Iteration-Level Batching

vLLM and similar systems batch at the iteration level:

```
Iteration 1: [Req A (token 1), Req B (token 1), Req C (token 1)]
Iteration 2: [Req A (token 2), Req B (token 2), Req C (done, removed)]
Iteration 3: [Req A (token 3), Req B (done), Req D (new, token 1)]
```

### In-Flight Batching (TensorRT-LLM)

Combine prefill and decode in same batch:

```
Batch: [
    Req A: decode step 50,
    Req B: decode step 12,
    Req C: prefill (new request)
]
```

Maximizes GPU utilization by mixing compute-heavy prefill with memory-heavy decode.

---

## Speculative Decoding

### The Idea

Use a small fast model to draft tokens, verify with large model in parallel.

```
Small model (draft):   token1, token2, token3, token4
Large model (verify):  ✓       ✓       ✗       -

Accept: token1, token2
Resample token3 from large model
Repeat
```

### How It Works

```python
def speculative_decode(
    prompt: str,
    draft_model,
    target_model,
    gamma: int = 4  # Draft tokens per step
) -> str:
    tokens = tokenize(prompt)
    
    while not done:
        # Draft: generate gamma tokens with small model
        draft_tokens = draft_model.generate(tokens, num_tokens=gamma)
        
        # Verify: run target model on all draft tokens
        target_logits = target_model.forward(tokens + draft_tokens)
        
        # Accept/reject each draft token
        accepted = 0
        for i, draft_token in enumerate(draft_tokens):
            if should_accept(draft_token, target_logits[i]):
                accepted += 1
            else:
                # Resample from target distribution
                tokens.append(sample(target_logits[i]))
                break
        
        if accepted == gamma:
            # All accepted, sample one more from target
            tokens.extend(draft_tokens)
            tokens.append(sample(target_logits[-1]))
    
    return detokenize(tokens)
```

### Speedup Factors

| Draft Model | Target Model | Acceptance Rate | Speedup |
|-------------|--------------|-----------------|---------|
| 7B | 70B | 70-80% | 2-3x |
| 1B | 7B | 60-70% | 1.5-2x |
| Same model (self-speculative) | - | 50-60% | 1.3-1.5x |

### When to Use

**Good fit:**
- Long generations
- Predictable content (code, structured output)
- Have a good draft model

**Poor fit:**
- Very short generations
- Highly creative/unpredictable content
- Latency-sensitive single requests

---

## Serving Frameworks

### vLLM

High-throughput serving with PagedAttention:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,  # Across 4 GPUs
    quantization="awq"
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256
)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

**Features:**
- PagedAttention for memory efficiency
- Continuous batching
- Tensor parallelism
- Quantization support

### TensorRT-LLM

NVIDIA-optimized serving:

```python
# Build optimized engine
trtllm-build --model_dir ./llama-70b \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16

# Deploy with Triton
# Supports in-flight batching, FP8, speculative decoding
```

**Features:**
- NVIDIA hardware optimization
- FP8 quantization (H100)
- In-flight batching
- Custom kernels

### Framework Comparison

| Feature | vLLM | TensorRT-LLM | Text Generation Inference |
|---------|------|--------------|---------------------------|
| Ease of use | High | Medium | High |
| Performance | Very good | Best (NVIDIA) | Good |
| Hardware | Any CUDA | NVIDIA optimized | Any CUDA |
| Quantization | GPTQ, AWQ | FP8, INT8, INT4 | GPTQ, bitsandbytes |
| Production ready | Yes | Yes | Yes |

---

## Hardware Selection

### GPU Comparison

| GPU | Memory | Bandwidth | FP16 TFLOPs | Best For |
|-----|--------|-----------|-------------|----------|
| A10G | 24 GB | 600 GB/s | 31 | Small models, inference |
| A100 40GB | 40 GB | 1.6 TB/s | 312 | Medium models |
| A100 80GB | 80 GB | 2.0 TB/s | 312 | Large models |
| H100 80GB | 80 GB | 3.35 TB/s | 989 | Maximum performance |
| L40S | 48 GB | 864 GB/s | 91 | Cost-effective inference |

### Sizing Guidelines

```python
def estimate_gpu_requirements(
    model_params_b: float,
    precision: str,
    batch_size: int,
    context_length: int
) -> dict:
    # Model memory
    bytes_per_param = {"fp16": 2, "int8": 1, "int4": 0.5}[precision]
    model_memory_gb = model_params_b * bytes_per_param
    
    # KV cache memory (approximate for common architectures)
    kv_per_token_mb = model_params_b * 0.00005  # Rough estimate
    kv_memory_gb = kv_per_token_mb * context_length * batch_size / 1000
    
    # Overhead
    overhead_gb = 2  # Activations, framework overhead
    
    total_gb = model_memory_gb + kv_memory_gb + overhead_gb
    
    return {
        "model_memory_gb": model_memory_gb,
        "kv_cache_gb": kv_memory_gb,
        "total_gb": total_gb,
        "recommended_gpu": recommend_gpu(total_gb)
    }

# Examples
estimate_gpu_requirements(70, "int4", 8, 4096)
# -> ~55 GB, recommend 1x H100 or 2x A100-40GB
```

### Multi-GPU Strategies

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| Tensor Parallel | Model too large for one GPU | Medium |
| Pipeline Parallel | Very large models | High |
| Data Parallel | Multiple replicas for throughput | Low |

---

## Optimization Checklist

### Quick Wins (Do First)

- [ ] Enable Flash Attention
- [ ] Use continuous batching
- [ ] Quantize to INT8 or INT4
- [ ] Enable KV cache
- [ ] Use GQA/MQA models when available

### Medium Effort

- [ ] Implement PagedAttention (use vLLM)
- [ ] Tune batch size for your workload
- [ ] Profile and identify bottlenecks
- [ ] Consider speculative decoding

### Advanced

- [ ] Custom CUDA kernels
- [ ] Hardware-specific optimizations (TensorRT-LLM)
- [ ] Model distillation for your use case
- [ ] Hybrid CPU/GPU strategies

---

## Interview Questions

### Q: Explain the difference between prefill and decode phases.

**Strong answer:**
LLM inference has two distinct phases:

**Prefill (Prompt Processing):**
- Processes all input tokens in parallel
- Compute-bound: limited by GPU FLOPs
- Populates the KV cache
- Latency scales with input length
- GPU utilization is high

**Decode (Token Generation):**
- Generates one token at a time
- Memory-bound: limited by memory bandwidth
- Reads entire KV cache each step
- Latency per token is roughly constant
- GPU compute often underutilized (10-30%)

**Implications for optimization:**
- Prefill: Optimize compute (better GPU, Flash Attention)
- Decode: Optimize memory (quantization, GQA, batching)
- Different optimization strategies for each phase

### Q: How does quantization affect inference?

**Strong answer:**
Quantization reduces model precision:

**Benefits:**
- Memory reduction: FP16 (2 bytes) → INT4 (0.5 bytes) = 4x less
- Speed improvement: Smaller memory footprint = faster loads
- Cost reduction: Fit larger models on fewer GPUs

**Tradeoffs:**
- Quality degradation: Typically 1-5% for INT4
- Not all operations quantize well
- May need calibration data

**Practical approach:**
1. Start with INT8: Usually <1% quality loss
2. Try AWQ/GPTQ for INT4: ~2-3% quality loss
3. Evaluate on your specific task
4. Use FP8 on H100 for best speed/quality

### Q: How would you optimize a 70B model for production serving?

**Strong answer:**
Layered optimization approach:

**1. Memory optimization:**
- AWQ 4-bit quantization: 140GB → 35GB
- GQA already in Llama 70B: 8x smaller KV cache
- PagedAttention: Near-zero memory waste

**2. Serving setup:**
- vLLM or TensorRT-LLM
- 2-4x H100 with tensor parallelism
- Continuous batching enabled

**3. Throughput optimization:**
- Target batch size 8-32
- Balance latency vs throughput based on SLO
- Consider speculative decoding for long outputs

**4. Cost optimization:**
- Right-size hardware to actual traffic
- Use spot instances for non-critical workloads
- Cache common responses

**Expected performance:**
- 30-50 tokens/sec per request
- 100-200 requests/min per replica
- P99 latency 2-5s for typical responses

---

## References

- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023)
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)
- vLLM: https://docs.vllm.ai/
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- GPTQ: https://github.com/IST-DASLab/gptq
- AWQ: https://github.com/mit-han-lab/llm-awq

---

*Next: [Serving Architecture](02-serving-architecture.md)*

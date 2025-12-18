# Serving Infrastructure

Deploying LLMs at scale requires a robust infrastructure layer that handles load balancing, model parallelism, and multi-tenant isolation. In 2025, the focus has shifted from "serving a model" to "orchestrating an inference fleet."

## Table of Contents

- [The Inference Gateway](#inference-gateway)
- [Model Parallelism (Tensor vs. Pipeline)](#parallelism)
- [Multi-GPU Orchestration](#multi-gpu)
- [Streaming and Long-Lived Connections](#streaming)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Inference Gateway

The gateway is the "Traffic Controller" for your AI workload.

| Component | Responsibility (Dec 2025) |
|-----------|---------------------------|
| **Auth & Rate Limiting** | Token-based quotas and tenant isolation. |
| **Model Router** | Directing requests to specific model versions (Canary/A-B). |
| **Context Tracker** | Ensuring a user's prompt cache is sent to the same GPU node (Sticky sessions). |
| **Output Filter** | Real-time safety and PII scrubbing on streaming responses. |

---

## Model Parallelism

For models that don't fit on a single GPU (e.g., Llama 4 405B requires ~800GB VRAM), we must split them.

### 1. Tensor Parallelism (TP)
Splits individual layers/tensors across multiple GPUs.
- **Latency**: Low (Fastest).
- **Communication**: High (Requires NVLink).
- **Standard**: Used for 90% of production serving within a single node (8x GPUs).

### 2. Pipeline Parallelism (PP)
Splits different layers (e.g., layers 1-40 on GPU 1, 41-80 on GPU 2).
- **Latency**: High (Micro-batching overhead).
- **Efficiency**: Lower util (Bubble time).
- **Standard**: Used only for massive models spanning multiple nodes.

---

## Multi-GPU Orchestration (Dec 2025)

In late 2025, Kubernetes operators (like **Kube-Ray** or **Gloo**) manage "GPU Pools."

- **Heterogeneous Clusters**: Mixing H100s for frontier models and L4s for small models in the same cluster.
- **Autoscaling**: Scaling based on **KV Cache utilization** rather than CPU or standard memory usage.
- **Cold Booting**: Using **Un-quantized Base Images** and loading weights from a high-speed Lustre/mount to reduce startup time from minutes to 15-20 seconds.

---

## Streaming and Long-Lived Connections

LLMs are almost always served via **Server-Sent Events (SSE)** or **WebSockets**.

**The 2025 Infrastructure Challenge**: Standard load balancers (Layer 4) struggle with long-lived AI connections.
- **The Fix**: Use **Layer 7 Load Balancers** (Envoy/Istio) that understand the "End of Sequence" token and can re-balance traffic *between* user turns rather than just at the connection level.

---

## Interview Questions

### Q: Why is Tensor Parallelism preferred over Pipeline Parallelism for low-latency serving?

**Strong answer:**
Tensor Parallelism (TP) performs the matrix multiplications of a single layer across multiple GPUs simultaneously. This means the latency of that layer is reduced by the number of GPUs. Pipeline Parallelism (PP), conversely, processes different layers sequentially. While GPU 2 is working on layers 40-80, GPU 1 is idle unless you have a deep pipeline of multiple requests (batching). For a single user's request, PP adds the latency of all GPUs, whereas TP divides the latency across all GPUs.

### Q: How do you handle "Noisy Neighbors" in a multi-tenant LLM cluster?

**Strong answer:**
We handle noisy neighbors through **Tiered Iteration-Level Scheduling**. Each tenant is assigned a "share" of the total GPU cycles. In the continuous batching loop, the scheduler ensures that a single tenant doesn't occupy 100% of the KV cache slots. If Tenant A is overwhelming the system, the scheduler will prioritize "Prefill" steps for Tenant B and C, or only process a subset of Tenant A's decode iterations per cycle. This is enforced at the Gateway via token-bucket rate limiting and at the serving engine via specific scheduling policies.

---

## References
- Narayanan et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Pipedream" (2019/2021)
- NVIDIA. "Megatron-LM: Training Multi-Billion Parameter Models on GPU Clusters" (2021)

---

*Next: [Cost Optimization Playbook](07-cost-optimization-playbook.md)*

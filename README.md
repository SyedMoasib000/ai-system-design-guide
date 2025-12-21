# 🧠 AI System Design Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Updated](https://img.shields.io/badge/Updated-December%202025-blue.svg)](#)
[![Stars](https://img.shields.io/github/stars/ombharatiya/ai-system-design-guide?style=social)](https://github.com/ombharatiya/ai-system-design-guide)

> **The missing manual for building production AI systems.** From LLM internals to multi-agent architectures, from RAG patterns to agentic security: everything you need to design, build, and scale AI at production grade.

---

## ✨ Why This Guide?

Traditional AI tutorials teach you to call an API. This guide teaches you to **architect systems** that handle millions of users, stay secure under adversarial conditions, and evolve with the rapidly changing AI landscape.

| What You Will Learn | What You Will NOT Find Here |
|-----|-----|
| Multi-tenant RAG isolation patterns | Basic "Hello World" LLM calls |
| KV Cache optimization for 10x throughput | Introductory Python tutorials |
| Agentic security (MCP, E2B sandboxing) | Explanations of REST APIs |
| Cost optimization playbooks (real $$$) | Generic cloud introductions |
| Staff-level interview Q&A | Outdated 2023 techniques |

---

## 🚀 Quick Start

**New to AI Systems?** Start here:
1. [LLM Internals](01-foundations/01-llm-internals.md) - How transformers actually work
2. [RAG Fundamentals](06-retrieval-systems/01-rag-fundamentals.md) - The most common production pattern
3. [Model Selection Guide](02-model-landscape/04-model-selection-guide.md) - Choosing GPT vs Claude vs Gemini

**Preparing for Interviews?**
- [Interview Question Bank](00-interview-prep/01-question-bank.md)
- [Answer Frameworks](00-interview-prep/02-answer-frameworks.md)
- [Whiteboard Exercises](00-interview-prep/04-whiteboard-exercises.md)

**Building Production Systems?**
- [Multi-Tenant RAG Isolation](12-security-and-access/04-multi-tenant-rag-isolation.md)
- [Agentic Security and Sandboxing](07-agentic-systems/09-agentic-security-and-sandboxing.md)
- [Cost Optimization Playbook](04-inference-optimization/07-cost-optimization-playbook.md)

---

## 📚 Guide Structure

```
ai-system-design-guide/
├── 00-interview-prep/           # Questions, frameworks, exercises
├── 01-foundations/              # LLM internals, tokenization, attention
├── 02-model-landscape/          # GPT-5, Claude 4, Gemini 3 comparisons
├── 03-training-and-adaptation/  # Fine-tuning, LoRA, DPO, distillation
├── 04-inference-optimization/   # KV cache, PagedAttention, vLLM
├── 05-prompting-and-context/    # CoT, DSPy, prompt injection defense
├── 06-retrieval-systems/        # RAG, chunking, GraphRAG, Agentic RAG
├── 07-agentic-systems/          # MCP, multi-agent, swarms, evaluation
├── 08-memory-and-state/         # L1-L3 memory tiers, Mem0, semantic cache
├── 09-frameworks-and-tools/     # LangGraph, DSPy, Semantic Kernel
├── 10-document-processing/      # Vision-LLM OCR, multimodal parsing
├── 11-infrastructure-and-mlops/ # GPU clusters, LLMOps, cost management
├── 12-security-and-access/      # RBAC, ABAC, zero-trust for AI
├── 13-reliability-and-safety/   # Guardrails, red-teaming, responsible AI
├── 14-evaluation-and-observability/ # RAGAS, LangSmith, drift detection
├── 15-ai-design-patterns/       # Pattern catalog and anti-patterns
├── 16-case-studies/             # Real-world enterprise architectures
└── GLOSSARY.md                  # All technical terms defined
```

---

## 🎯 What Makes This Different

### Real Numbers, Not Vibes
Every benchmark includes sources and verification dates. No hallucinated metrics.

### December 2025 Context
Updated for the latest models: GPT-5.2, Claude 4.5 Opus, Gemini 3, o3/o4, DeepSeek R1.

### Interview-Ready Depth
Each chapter includes sample questions and staff-level answer frameworks.

### Zero Repetition
Concepts appear once with cross-references. No bloat.

---

## 📖 Chapter Highlights

| Chapter | Must-Read Topics |
|---------|------------------|
| **07 Agentic Systems** | MCP (Model Context Protocol), Swarm patterns, Trajectory evaluation |
| **06 Retrieval** | GraphRAG, Agentic RAG, Matryoshka embeddings |
| **08 Memory** | L1-L3 cognitive tiers, Mem0, semantic caching |
| **09 Frameworks** | LangGraph state machines, DSPy prompt compilation |
| **12 Security** | Multi-tenant RAG isolation, PII filtering |

---

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md) first.

- Cite sources for all benchmark data
- Include "Last verified: [date]" for time-sensitive info
- Follow existing table and code formats
- Run markdown linting before submitting PRs

---

## 👤 Author

**Om Bharatiya**

Building AI systems that scale. Previously: distributed systems, cloud infrastructure, developer tools.

[![GitHub](https://img.shields.io/badge/GitHub-ombharatiya-181717?logo=github)](https://github.com/ombharatiya)
[![Twitter](https://img.shields.io/badge/Twitter-@ombharatiya-1DA1F2?logo=twitter)](https://x.com/ombharatiya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ombharatiya-0A66C2?logo=linkedin)](https://linkedin.com/in/ombharatiya)

**If this guide helped you, consider giving it a ⭐!**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Last updated: December 2025</i>
</p>

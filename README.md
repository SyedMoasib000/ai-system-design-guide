# 🧠 AI System Design Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Updated](https://img.shields.io/badge/Updated-December%202025-blue.svg)](#)
[![Stars](https://img.shields.io/github/stars/ombharatiya/ai-system-design-guide?style=social)](https://github.com/ombharatiya/ai-system-design-guide)

> **The living reference for production AI systems.**  
> Updated continuously as the field evolves. From LLM internals to multi-agent architectures: everything you need to design, build, and scale AI at production grade.

---

## 📖 Why This Exists

**The AI field moves too fast for traditional books.**

By the time a printed book on LLMs hits the shelves, its model comparisons are outdated, its pricing tables are wrong, and its "best practices" have been superseded by new techniques. This guide solves that problem.

**This is a living document.** When OpenAI releases a new model, when a new prompting technique emerges, when costs change: this guide updates. You get the depth of a technical book with the freshness of a blog post.

### What You Get Here That No Single Book Offers

| This Guide | Traditional AI Books |
|------------|---------------------|
| December 2025 model comparisons (GPT-5, Claude 4.5, Gemini 3) | Stuck on GPT-4 or earlier |
| MCP (Model Context Protocol), the new standard for agents | Does not exist in print |
| Real API pricing with verification dates | Static numbers that are already wrong |
| Production patterns from actual deployments | Academic examples that do not scale |
| Interview questions with staff-level answer frameworks | Generic "explain transformers" questions |
| Cross-referenced, zero repetition | Same concepts explained three different ways |

---

## 🎯 Who This Is For

This guide is written for **senior engineers** making real architectural decisions:

- **Tech Leads** evaluating whether to build RAG or fine-tune
- **Staff Engineers** designing multi-tenant AI platforms
- **Principal Engineers** setting AI strategy across organizations
- **Senior ICs** preparing for AI-focused system design interviews

**Prerequisites:** You should be comfortable with distributed systems concepts (microservices, event-driven architectures, cloud infrastructure). This guide bridges that knowledge to AI-specific patterns.

**Not for:** Beginners looking for "How to call the OpenAI API" tutorials. Those are abundant elsewhere.

---

## 🔥 Why This Matters for Interviews

AI system design interviews are now common at senior levels. The questions are specific and require depth:

> "Design a multi-tenant RAG system that prevents cross-customer data leakage."

> "You have 10M documents and need sub-second retrieval. Walk me through the architecture."

> "Your agent is taking 15 steps to complete a 3-step task. How do you debug and fix this?"

Generic "transformers and attention" knowledge will not cut it. You need:
- **Concrete patterns** (not vague "use RAG")
- **Real tradeoffs** (latency vs cost vs accuracy)
- **Production failure modes** (and how to prevent them)
- **Current best practices** (not 2023 techniques)

This guide gives you all of that, organized for rapid reference before interviews.

### Interview Preparation Path
1. [Question Bank](00-interview-prep/01-question-bank.md): 50+ real interview questions by topic
2. [Answer Frameworks](00-interview-prep/02-answer-frameworks.md): Staff-level response structures
3. [Common Pitfalls](00-interview-prep/03-common-pitfalls.md): What trips up experienced engineers
4. [Whiteboard Exercises](00-interview-prep/04-whiteboard-exercises.md): Practice problems with solutions

---

## 🚀 Quick Start Paths

### Path 1: "I need to understand this field fast" (2 hours)
1. [LLM Internals](01-foundations/01-llm-internals.md): How transformers actually work
2. [RAG Fundamentals](06-retrieval-systems/01-rag-fundamentals.md): The most common production pattern
3. [Model Selection Guide](02-model-landscape/04-model-selection-guide.md): Choosing GPT vs Claude vs Gemini vs Open Source

### Path 2: "I am building a production system" (Deep Dive)
Follow chapters 01-17 in order. Each builds on previous concepts with explicit cross-references.

### Path 3: "I have an interview next week" (Focused Prep)
Start with [Chapter 00: Interview Prep](00-interview-prep/), then reference specific chapters for depth on any topic that comes up.

---

## 📚 What Is Covered

```
ai-system-design-guide/
├── 00-interview-prep/           # Questions, frameworks, whiteboard exercises
├── 01-foundations/              # Transformers, attention, tokenization, embeddings
├── 02-model-landscape/          # GPT-5, Claude 4.5, Gemini 3, o3, DeepSeek R1
├── 03-training-and-adaptation/  # Fine-tuning, LoRA, DPO, distillation, synthetic data
├── 04-inference-optimization/   # KV cache, PagedAttention, vLLM, cost optimization
├── 05-prompting-and-context/    # CoT, DSPy, prompt injection defense, structured output
├── 06-retrieval-systems/        # RAG, chunking, GraphRAG, Agentic RAG, reranking
├── 07-agentic-systems/          # MCP, reasoning loops, multi-agent, swarms, evaluation
├── 08-memory-and-state/         # L1-L3 memory tiers, Mem0, semantic caching
├── 09-frameworks-and-tools/     # LangGraph, DSPy, LlamaIndex, Semantic Kernel
├── 10-document-processing/      # Vision-LLM OCR, multimodal parsing, layout analysis
├── 11-infrastructure-and-mlops/ # GPU clusters, LLMOps, CI/CD for AI
├── 12-security-and-access/      # RBAC, ABAC, multi-tenant isolation, zero-trust
├── 13-reliability-and-safety/   # Guardrails, red-teaming, responsible AI
├── 14-evaluation-and-observability/ # RAGAS, LangSmith, LLM-as-Judge, drift detection
├── 15-ai-design-patterns/       # Pattern catalog and anti-patterns
├── 16-case-studies/             # Real-world enterprise architectures
├── GLOSSARY.md                  # Every technical term defined precisely
└── PATTERNS.md                  # Quick reference for all AI design patterns
```

---

## 🔄 The Living Book Philosophy

This guide follows three principles:

### 1. Accuracy Over Speed
Every benchmark, every price, every capability claim includes a source or verification note. When information is time-sensitive, it is flagged: `⚠️ Verify current values`.

### 2. Depth Over Breadth
Rather than shallow coverage of 100 topics, this guide provides interview-ready depth on the topics that actually matter for production systems.

### 3. Continuous Updates
AI moves fast. This guide tracks:
- New model releases and their real-world performance
- Emerging patterns (MCP, Agentic RAG, Flow Engineering)
- Changing best practices and deprecations
- Updated pricing and rate limits

**Watch this repository** to get notified when significant updates are pushed.

---

## 📊 Chapter Highlights

| Chapter | Key Topics You Will Not Find Elsewhere |
|---------|----------------------------------------|
| **06 Retrieval** | GraphRAG with community summarization, Agentic RAG, Matryoshka embeddings |
| **07 Agents** | MCP (Model Context Protocol), Swarm patterns, Trajectory-based evaluation |
| **08 Memory** | L1-L3 cognitive architecture, Mem0 integration, semantic caching economics |
| **09 Frameworks** | LangGraph state machines, DSPy prompt compilation, Framework selection matrix |
| **12 Security** | Multi-tenant RAG isolation, context leakage prevention, PII filtering pipelines |

---

## 🤝 Contributing

This guide improves with community input. Contributions welcome:

- **Corrections**: Found outdated information? Open an issue or PR.
- **Additions**: Have production experience to share? Contribute a case study.
- **Questions**: Something unclear? Open a discussion.

Please follow the [Contributing Guide](CONTRIBUTING.md) for formatting standards.

---

## 👤 Author

**Om Bharatiya**

Building AI systems that scale. Background in distributed systems, cloud infrastructure, and developer tools.

[![GitHub](https://img.shields.io/badge/GitHub-ombharatiya-181717?logo=github)](https://github.com/ombharatiya)
[![Twitter](https://img.shields.io/badge/Twitter-@ombharatiya-1DA1F2?logo=twitter)](https://x.com/ombharatiya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ombharatiya-0A66C2?logo=linkedin)](https://linkedin.com/in/ombharatiya)

---

## ⭐ Support This Project

If this guide has helped you:
- **Star this repository** to help others discover it
- **Share it** with colleagues preparing for AI roles
- **Contribute** corrections, updates, or case studies

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Last updated: December 2025</b><br/>
  <i>This is a living document. Star and watch for updates.</i>
</p>

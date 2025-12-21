# 🧠 AI System Design Guide

<p align="center">
  <a href="https://github.com/ombharatiya"><img src="https://img.shields.io/badge/GitHub-ombharatiya-181717?logo=github" alt="GitHub"></a>
  <a href="https://x.com/ombharatiya"><img src="https://img.shields.io/badge/Twitter-@ombharatiya-1DA1F2?logo=twitter" alt="Twitter"></a>
  <a href="https://linkedin.com/in/ombharatiya"><img src="https://img.shields.io/badge/LinkedIn-ombharatiya-0A66C2?logo=linkedin" alt="LinkedIn"></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Updated-December%202025-blue.svg" alt="Last Updated"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/ombharatiya/ai-system-design-guide"><img src="https://img.shields.io/github/stars/ombharatiya/ai-system-design-guide?style=social" alt="Stars"></a>
</p>

> **The living reference for production AI systems.** Continuously updated. Interview-ready depth.

---

## 📚 Quick Navigation

| I want to... | Start here |
|--------------|------------|
| **Prepare for interviews** | [Question Bank](00-interview-prep/01-question-bank.md) → [Answer Frameworks](00-interview-prep/02-answer-frameworks.md) |
| **Learn AI systems fast** | [LLM Internals](01-foundations/01-llm-internals.md) → [RAG Fundamentals](06-retrieval-systems/01-rag-fundamentals.md) |
| **Build production RAG** | [Chunking](06-retrieval-systems/02-chunking-strategies.md) → [Vector DBs](06-retrieval-systems/04-vector-databases-comparison.md) → [Reranking](06-retrieval-systems/06-reranking-strategies.md) |
| **Design multi-tenant AI** | [Isolation Patterns](12-security-and-access/04-multi-tenant-rag-isolation.md) → [Case Study](16-case-studies/08-multi-tenant-saas.md) |
| **Build agents** | [Agent Fundamentals](07-agentic-systems/01-agent-fundamentals.md) → [MCP](07-agentic-systems/03-tool-use-and-mcp.md) → [LangGraph](09-frameworks-and-tools/02-langgraph-orchestration.md) |

---

## 🎯 Why This Guide

**Traditional books are outdated before they ship.** This is a living document: when new models release, when patterns evolve, this updates.

| This Guide | Printed Books |
|------------|---------------|
| December 2025 models (GPT-5, Claude 4.5, Gemini 3) | Stuck on GPT-4 |
| MCP, Agentic RAG, Flow Engineering | Does not exist |
| Real pricing with verification dates | Already wrong |
| Staff-level interview Q&A | Generic questions |

---

## 📖 Guide Structure

```
├── 00-interview-prep/           # Questions, frameworks, exercises
├── 01-foundations/              # Transformers, attention, embeddings
├── 02-model-landscape/          # GPT-5, Claude 4.5, Gemini 3, o3, DeepSeek
├── 03-training-and-adaptation/  # Fine-tuning, LoRA, DPO, distillation
├── 04-inference-optimization/   # KV cache, PagedAttention, vLLM
├── 05-prompting-and-context/    # CoT, DSPy, prompt injection defense
├── 06-retrieval-systems/        # RAG, chunking, GraphRAG, Agentic RAG
├── 07-agentic-systems/          # MCP, multi-agent, swarms, evaluation
├── 08-memory-and-state/         # L1-L3 memory tiers, Mem0, caching
├── 09-frameworks-and-tools/     # LangGraph, DSPy, LlamaIndex
├── 10-document-processing/      # Vision-LLM OCR, multimodal parsing
├── 11-infrastructure-and-mlops/ # GPU clusters, LLMOps, cost management
├── 12-security-and-access/      # RBAC, ABAC, multi-tenant isolation
├── 13-reliability-and-safety/   # Guardrails, red-teaming
├── 14-evaluation-and-observability/ # RAGAS, LangSmith, drift detection
├── 15-ai-design-patterns/       # Pattern catalog, anti-patterns
├── 16-case-studies/             # Real-world architectures with diagrams
└── GLOSSARY.md                  # Every term defined
```

---

## 🔥 Featured Case Studies

Real interview problems with complete solutions:

| Case Study | Problem | Key Patterns |
|------------|---------|--------------|
| [Real-Time Search](16-case-studies/06-real-time-search.md) | 5-minute data freshness at scale | Streaming + Hybrid Search |
| [Coding Agent](16-case-studies/07-autonomous-coding-agent.md) | Autonomous multi-file changes | Sandboxing + Self-Correction |
| [Multi-Tenant SaaS](16-case-studies/08-multi-tenant-saas.md) | Coca-Cola and Pepsi on same infra | Defense-in-Depth Isolation |
| [Customer Support](16-case-studies/09-customer-support-automation.md) | 60% auto-resolution rate | Tiered Routing + Escalation |
| [Document Intelligence](16-case-studies/10-document-intelligence.md) | 50K contracts/month extraction | Vision-LLM + Parallel Extractors |

---

## 🎓 For Interview Prep

AI system design interviews ask questions like:

> "Design a multi-tenant RAG system where competitors cannot see each other's data."

> "Your agent takes 15 steps for a 3-step task. How do you debug it?"

This guide gives you **concrete patterns**, **real tradeoffs**, and **production failure modes**: the depth interviewers expect at senior levels.

➡️ Start with [Interview Prep](00-interview-prep/)

---

## 🔄 Living Book

This guide tracks:
- New model releases and real-world performance
- Emerging patterns (MCP, Agentic RAG, Flow Engineering)
- Updated pricing and rate limits
- Deprecations and best practice changes

**⭐ Star and Watch** to get notified when updates are pushed.

---

## 🤝 Contributing

Found outdated info? Have production experience to share? PRs welcome.
See [Contributing Guide](CONTRIBUTING.md).

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

<p align="center">
  <b>Built by <a href="https://github.com/ombharatiya">Om Bharatiya</a></b><br/>
  <a href="https://github.com/ombharatiya"><img src="https://img.shields.io/badge/GitHub-Follow-181717?logo=github" alt="GitHub"></a>
  <a href="https://x.com/ombharatiya"><img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=twitter" alt="Twitter"></a>
  <a href="https://linkedin.com/in/ombharatiya"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin" alt="LinkedIn"></a>
</p>

<p align="center"><i>Last updated: December 2025</i></p>

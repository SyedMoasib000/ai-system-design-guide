# Semantic Kernel (Dec 2025)

**Semantic Kernel (SK)** is Microsoft's engine for enterprise-grade AI orchestration. In late 2025, it is the primary bridge for organizations committed to the **Azure/Microsoft ecosystem** and **C#/.NET** architectures.

## Table of Contents

- [Enterprise DNA](#dna)
- [Plugins and Planners](#plugins)
- [Memory and Connectors](#memory)
- [Multi-Language Support (C# vs. Python)](#multi-language)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Enterprise DNA

While LangChain is favored by startups, Semantic Kernel is favored by **Banks and Fortune 500s**.
- **Dependency Injection**: SK follows standard enterprise design patterns.
- **Strong Typing**: First-class support for C# types makes it highly reliable in large-scale mission-critical systems.
- **Security**: Deep integration with Azure Active Directory (Microsoft Entra ID) and Managed Identities.

---

## Plugins and Planners

1. **Kernel Functions**: The basic unit of logic (Native code or LLM prompts).
2. **Plugins**: A collection of functions (e.g., a "GitHub Plugin" or an "SQL Plugin").
3. **Planners (2025 Tech)**: SK's planners have evolved from simple ReAct to **Hierarchical Planners** that can coordinate long-running business processes across multiple days.

---

## Memory and Connectors

Semantic Kernel uses **Connectors** to abstract away the underlying infrastructure.
- **Universal Connectors**: One interface for OpenAI, Mistral, and local Onyx models.
- **Vector Store Abstraction**: Seamlessly switch between Azure AI Search, Pinecone, and Qdrant without changing the core business logic.

---

## Multi-Language Support

In 2025, SK is the **only** major framework that treats C# and Python as equals.
- **The Pattern**: Develop and prototype in Python; deploy the core orchestration in C# for performance and type-safety.
- **Logic Sharing**: Shared prompt templates (.yaml) that work across both languages.

---

## Interview Questions

### Q: Why would a Staff Engineer choose Semantic Kernel over LangChain?

**Strong answer:**
**Architectural Alignment**. If an organization is already built on the .NET/Azure stack, Semantic Kernel fits into their existing CI/CD, monitoring (App Insights), and security (Entra ID) pipelines. LangChain often feels like an "external" piece of tech. Furthermore, SK's **Strong Typing** and **Dependency Injection** patterns prevent the "spaghetti code" that often plagues large LangChain projects. For an enterprise handling sensitive financial data, the **Native Azure integration** for security and auditing is the deciding factor.

### Q: What is the "Function Calling" abstraction in Semantic Kernel?

**Strong answer:**
SK uses a **Plugin-based model**. Every function (native C# or LLM-based) is registered with the Kernel. When the LLM decides it needs a tool, the Kernel looks up the function in the Plugin registry, validates the parameters, and executes it. In 2025, this supports **Automatic Intent Detection**: the Kernel can proactively suggest which Plugin a user might need before they even ask, based on the current context window.

---

## References
- Microsoft Learn. "Semantic Kernel Documentation" (2025)
- Azure Architecture Center. "AI Design Patterns with Semantic Kernel" (2025)
- Build 2025. "The Future of Copilots with SK" (2025 Conference Recap)

---

*Next: [AutoGen and CrewAI](07-autogen-crewai.md)*

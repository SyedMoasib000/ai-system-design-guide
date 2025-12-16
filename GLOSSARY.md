# AI System Design Glossary

Quick reference for key terms used throughout this guide.

---

## A

**Agentic System** - LLM application that autonomously plans and executes multi-step tasks using tools.

**Attention Mechanism** - Neural network component that allows models to focus on relevant parts of input. Self-attention compares each token to all others.

**ABAC (Attribute-Based Access Control)** - Access control based on attributes of user, resource, and environment rather than fixed roles.

---

## B

**Batching** - Processing multiple requests together to improve GPU utilization. Continuous batching adds new requests while others generate.

**BM25** - Traditional keyword-based ranking algorithm. Often combined with vector search for hybrid retrieval.

---

## C

**Chain-of-Thought (CoT)** - Prompting technique that elicits step-by-step reasoning before final answer.

**Chunking** - Splitting documents into smaller pieces for embedding and retrieval. Strategies include fixed-size, semantic, and hierarchical.

**Context Window** - Maximum number of tokens an LLM can process in a single request. Ranges from 4K to 2M+ tokens.

**Cosine Similarity** - Measure of similarity between two vectors. Standard metric for comparing embeddings.

---

## D

**DPO (Direct Preference Optimization)** - Fine-tuning method that optimizes directly on preference data without a separate reward model.

**DSPy** - Framework for programming LLMs through optimizable modules rather than manual prompts.

---

## E

**Embedding** - Dense vector representation of text. Used for semantic search and similarity comparison.

**Ensemble** - Combining multiple model outputs to improve reliability. Includes voting, debate, and mixture-of-agents.

---

## F

**Few-Shot Prompting** - Including examples in the prompt to guide model behavior.

**Fine-Tuning** - Training a pre-trained model on task-specific data to improve performance.

**Function Calling** - LLM capability to output structured tool invocations rather than plain text.

---

## G

**Guardrails** - Input/output validation to prevent harmful or off-topic responses.

**Grounding** - Connecting LLM responses to factual sources to reduce hallucination.

---

## H

**Hallucination** - Model generating plausible but factually incorrect information.

**HNSW (Hierarchical Navigable Small World)** - Graph-based algorithm for approximate nearest neighbor search in vector databases.

**Human-in-the-Loop (HITL)** - Patterns for human oversight, approval, or correction of AI outputs.

---

## I

**In-Context Learning** - Model adapting to task based on examples in the prompt without weight updates.

**Inference** - Running a trained model to generate predictions/outputs.

---

## J

**JSON Mode** - LLM output mode that guarantees valid JSON structure.

---

## K

**KV Cache** - Cached key-value pairs from attention computation. Enables efficient autoregressive generation.

---

## L

**LangChain** - Framework for building LLM applications with chains, agents, and integrations.

**LlamaIndex** - Data framework focused on document processing and retrieval for LLM applications.

**LoRA (Low-Rank Adaptation)** - Parameter-efficient fine-tuning that trains small adapter matrices instead of full model weights.

**LLM-as-Judge** - Using an LLM to evaluate outputs from another LLM.

---

## M

**MCP (Model Context Protocol)** - Anthropic's protocol for standardized tool/resource integration with LLMs.

**Mixture of Agents (MoA)** - Ensemble pattern where multiple agents contribute to a synthesized response.

**Multi-Tenancy** - Serving multiple customers from shared infrastructure with data isolation.

---

## O

**OCR (Optical Character Recognition)** - Extracting text from images or scanned documents.

---

## P

**Prompt Injection** - Attack where malicious input manipulates LLM behavior.

**Prefix Caching** - Reusing KV cache for common prompt prefixes across requests.

---

## Q

**QLoRA** - LoRA combined with 4-bit quantization for memory-efficient fine-tuning.

**Quantization** - Reducing model precision (e.g., FP16 to INT4) to decrease memory and improve speed.

---

## R

**RAG (Retrieval-Augmented Generation)** - Pattern that retrieves relevant documents to provide context for LLM generation.

**RBAC (Role-Based Access Control)** - Access control based on user roles with predefined permissions.

**ReAct** - Agent pattern alternating between Reasoning and Acting steps.

**Reranking** - Second-stage scoring to improve retrieval precision. Cross-encoders provide higher accuracy than bi-encoders.

**RLHF (Reinforcement Learning from Human Feedback)** - Training method using human preferences to align model behavior.

---

## S

**Self-Consistency** - Sampling multiple reasoning paths and selecting most common answer.

**Semantic Search** - Finding documents by meaning rather than keywords, using embeddings.

**Speculative Decoding** - Using small draft model to propose tokens, verified by large model.

**Structured Output** - Constraining LLM output to specific formats (JSON, XML, etc.).

**System Prompt** - Instructions that set context and behavior for an LLM conversation.

---

## T

**Temperature** - Parameter controlling randomness of LLM outputs. Lower = more deterministic.

**Token** - Basic unit of text processing. Roughly 0.75 words or 4 characters in English.

**Tool Use** - LLM capability to invoke external functions/APIs.

**Transformer** - Neural network architecture based on self-attention. Foundation of modern LLMs.

---

## V

**Vector Database** - Database optimized for storing and searching high-dimensional vectors (embeddings).

---

## Z

**Zero-Shot** - Prompting without examples, relying on model's pre-existing knowledge.

---

*See also: [PATTERNS.md](PATTERNS.md) for design pattern quick reference*

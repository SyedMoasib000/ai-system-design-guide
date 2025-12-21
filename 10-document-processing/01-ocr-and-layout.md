# OCR and Layout Analysis (Dec 2025)

In late 2025, traditional OCR (Tesseract, specialized engines) has been largely superseded by **Native Multimodal LLMs** (Gemini 2.0, GPT-4o, Claude Sonnet 4.5). We no longer "read characters"; we "understand layouts."

## Table of Contents

- [The Shift: Traditional OCR vs. Vision-LLMs](#shift)
- [Vision-LLM Layout Extraction](#layout-extraction)
- [Reading Order and Logical Structure](#reading-order)
- [Handling Low-Quality Scans and Handwriting](#quality)
- [Cost and Latency Tradeoffs](#tradeoffs)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Shift: Traditional OCR vs. Vision-LLMs

| Feature | Traditional OCR (Tesseract/AWS Textract) | Vision-LLMs (Gemini/GPT-4o) |
|---------|-------------------------------------------|-----------------------------|
| **Primary Mechanism** | Character recognition | Visual token understanding |
| **Logic** | Point-and-line analysis | Semantic context |
| **Reading Order** | Simple top-to-bottom | Multi-column, complex layout aware |
| **Handwriting** | Poor | Excellent (Human-level) |
| **Output** | Text blocks + Bounding boxes | Structured Markdown/JSON |

---

## Vision-LLM Layout Extraction

The 2025 standard workflow is **Screenshot-to-Markdown**.
1. **Rasterize**: Convert PDF pages to images.
2. **Visual Prompting**: Ask the vision model to "Transcribe the following page into GitHub-flavored Markdown, preserving tables and headers."
3. **Structured Recovery**: Use the model's spatial awareness to rebuild the logical hierarchy.

---

## Reading Order and Logical Structure

> [!IMPORTANT]
> A common failure in naive RAG is breaking a paragraph across a column. 
> Vision-LLMs solve this by "Seeing" the column gutter and correctly sequencing the text, unlike rule-based parsers that might read straight across both columns.

---

## Handling Low-Quality Scans

Late 2025 models are robust to:
- **Skew/Rotation**: Automatically corrected in the visual attention layer.
- **Bleed-through**: The model uses semantic context to "ignore" text from the back of the page.
- **Handwritten Annotations**: Can be extracted into a separate `annotations` JSON field.

---

## Cost and Latency Tradeoffs

| Model Tier | Use Case | Latency | Cost (1K pages) |
|------------|----------|---------|-----------------|
| **Gemini 2.0 Flash** | High-volume batch | 1-2s / page | $1-3 |
| **GPT-4o (Native)** | High-precision / Legal | 3-5s / page | $10-20 |
| **Local (Llama 3.2 Vision)** | PII-sensitive / On-prem | <1s / page | Infrastructure only |

---

## Interview Questions

### Q: Why would you still use AWS Textract or Azure AI Search (OCR) in 2025?

**Strong answer:**
**Strict Spatial Metadata and Compliance**. If my application needs exact pixel-level bounding boxes for every single word (e.g., for a legal redaction tool), a specialized OCR engine is often more precise and cheaper. Furthermore, OCR engines are **Deterministic**: they do not \"Hallucinate\" words that do not exist. For high-stakes document processing where 100% character accuracy is required over \"Layout understanding,\" traditional engines still hold a spot in the hybrid pipeline.

### Q: How do you handle a 500-page PDF with Vision LLMs efficiently?

**Strong answer:**
We use a **Parallel Map-Reduce** pattern. 
1. **Map**: We spin up 50 parallel workers (using AWS Lambda or Modal) to process 10 pages each. Each worker calls a fast Vision model (like Gemini 2.0 Flash) to get the Markdown.
2. **Consolidate**: A central agent reviews the Markdown snippets to ensure header continuity.
3. **Cache**: We store the resulting Markdown in a vector DB.
This reduces the processing time from 30 minutes (sequential) to under 20 seconds.

---

## References
- Google DeepMind. "Gemini 2.0: Understanding Multi-column Documents" (2025)
- OpenAI. "Vision Models for Document Understanding" (2025)
- Tesseract v6. "The Integration of Hybrid Transformer OCR" (2025)

---

*Next: [Multimodal Parsing and Markdown Conversion](02-multimodal-parsing.md)*

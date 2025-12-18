# Structured Generation

Structured Generation is the process of forcing an LLM to produce output in a machine-readable format (JSON, YAML, CSV) with 100% reliability. In 2025, this has evolved from "prompt-based requests" to "engine-level constraints."

## Table of Contents

- [The JSON Mode Revolution](#json-mode)
- [Function Calling & Tool Use](#function-calling)
- [Constrained Decoding (CFG & Regex)](#constrained-decoding)
- [Multi-Stage Extraction Pattern](#multi-stage)
- [Validation & Formatting Errors](#validation)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The JSON Mode Revolution

Historically, getting JSON was a struggle of "only return JSON, no other text."
**The 2025 Standard**: Use native `response_format: { type: "json_schema" }` (OpenAI/Gemini) or similar.

- **Benefit**: 100% syntactical validity. The model literally cannot output a string that is not a valid JSON.
- **Behind the scenes**: The serving engine masks the vocabulary at each step, ensuring only valid JSON characters (e.g., `{`, `"`, `:`, `[`) can be picked next.

---

## Function Calling & Tool Use

Function calling is structured generation where the LLM "picks" a function and populates its arguments.

```json
// Example Tool Call
{
  "name": "get_stock_price",
  "arguments": { "symbol": "AAPL", "interval": "1d" }
}
```

**2025 Nuance**: We now use **Parallel Function Calling**. A model can decide to call 5 different tools simultaneously (e.g., check account balance, check credit score, check loan rates) and aggregate the results.

---

## Constrained Decoding (CFG & Regex)

For self-hosted models (Llama-cpp, vLLM via Outlines), we use **Context-Free Grammars (CFG)** or **Regex**.

```python
# Outlines Pattern (2025)
model = outlines.models.transformers("meta-llama/Llama-4-8B")
generator = outlines.generate.regex(model, r"(\d{3})-\d{3}-\d{4}")
# Result: The model can ONLY output telephone numbers.
```

---

## Multi-Stage Extraction Pattern

For complex data extraction (e.g., 50 fields from a medical record), don't do it in one pass.
- **Stage 1 (Text-to-Text)**: Extract a "messy" but complete set of facts in natural language.
- **Stage 2 (Text-to-JSON)**: Use a smaller, cheaper model to convert those natural language facts into a strict JSON schema.
- **Benefit**: Reduces "hallucination under pressure"—large models struggle when forced to reason AND follow strict syntax simultaneously.

---

## Validation & Formatting Errors

Even with "JSON mode," the **Logic** inside the JSON might be wrong (e.g., a field is missing or a date is in the wrong format).

**The 2025 Recovery Pattern**:
1. Validate output against **Pydantic/Zod**.
2. If it fails, send the **Traceback** back to the model:
   "Error: Field 'age' must be an integer, got 'twenty'. Fix and re-generate."
3. Most models fix the error on the first retry.

---

## Interview Questions

### Q: Why is "JSON Mode" more reliable than prompt-based JSON requests?

**Strong answer:**
Prompt-based requests rely on the model's *willingness* to follow instructions; "JSON Mode" (or Constrained Decoding) relies on the serving engine's *inability* to do anything else. By applying a "Logit Bias" or a "Grammar Mask" at the inference level, the engine restricts the choice of the next token to only those that would be valid according to the schema. This eliminates the "preamble" (e.g., "Sure, here is your JSON...") and ensures that you never get a malformed string due to high temperature or randomness.

### Q: What is the risk of asking an LLM for too many structured fields at once?

**Strong answer:**
There is a trade-off between **Schema Complexity** and **Information Integrity**. As the schema grows (e.g., 20+ hierarchical fields), the model's attention is consumed by maintaining the JSON structure (brackets, keys, quotes) rather than verifying the accuracy of the data. This often leads to "Omission Hallucinations" where the model skips fields or fills them with placeholder data. In 2025, we mitigate this by using a "Chain-of-Density" extraction or splitting the extraction into multiple parallel sub-tasks.

---

## References
- OpenAI. "Structured Outputs Documentation" (August 2024 update)
- Outlines Project. "Context-Free Grammar Guided Generation" (2024)
- Willard et al. "Efficient Guided Generation for LLMs" (2023)

---

*Next: [Prompt Optimization (DSPy)](07-prompt-optimization-dspy.md)*

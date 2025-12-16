# Multimodal Document Understanding

Modern document processing leverages vision models and multimodal LLMs to understand documents beyond text extraction.

## Table of Contents

- [Vision-Based Document Understanding](#vision-based-document-understanding)
- [Multimodal LLMs for Documents](#multimodal-llms-for-documents)
- [Layout Understanding](#layout-understanding)
- [Table Extraction](#table-extraction)
- [Form Processing](#form-processing)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Vision-Based Document Understanding

### Why Vision Matters

| Text-Only Extraction | Vision-Based |
|----------------------|--------------|
| Loses layout context | Preserves spatial relationships |
| Tables become garbled | Tables understood visually |
| Misses diagrams | Can describe diagrams |
| Reading order issues | Natural reading order |

### Document-to-Image Processing

```python
from pdf2image import convert_from_path
from PIL import Image
import base64
import io

class DocumentImageProcessor:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi
    
    def pdf_to_images(self, path: str) -> list[Image.Image]:
        return convert_from_path(path, dpi=self.dpi)
    
    def image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
```

---

## Multimodal LLMs for Documents

### GPT-4o Vision

```python
from openai import OpenAI

client = OpenAI()

def analyze_document_page(image_base64: str, query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high"
                }}
            ]
        }],
        max_tokens=4096
    )
    return response.choices[0].message.content
```

### Claude Vision

```python
import anthropic

client = anthropic.Anthropic()

def analyze_with_claude(image_base64: str, query: str) -> str:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64
                }},
                {"type": "text", "text": query}
            ]
        }]
    )
    return message.content[0].text
```

---

## Layout Understanding

```python
class LayoutDetector:
    def detect_layout(self, image: Image.Image) -> list[dict]:
        # Use layout detection model
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        elements = []
        for score, label, box in zip(
            outputs["scores"], outputs["labels"], outputs["boxes"]
        ):
            if score > 0.5:
                elements.append({
                    "type": self.model.config.id2label[label.item()],
                    "confidence": score.item(),
                    "bbox": box.tolist()
                })
        
        return elements
```

---

## Table Extraction

### Vision-Based Table Extraction

```python
class VisionTableExtractor:
    async def extract_table(self, image_base64: str) -> dict:
        prompt = """
Analyze the table in this image.
1. Identify all column headers
2. Extract all data rows
3. Preserve the structure exactly

Return as JSON:
{
    "headers": ["col1", "col2", ...],
    "rows": [["val1", "val2", ...], ...]
}
"""
        result = await self.vision.analyze(image_base64, prompt)
        return json.loads(result)
```

---

## Form Processing

```python
class FormProcessor:
    async def extract_form_fields(self, image_base64: str) -> list[dict]:
        prompt = """
Extract all form fields from this document.
For each field, identify:
1. Field label/name
2. Field value (what was filled in)
3. Field type (text, checkbox, date, signature)
4. Whether it is filled or empty

Return as JSON array.
"""
        result = await self.vision.analyze(image_base64, prompt)
        return json.loads(result)
```

---

## Interview Questions

### Q: When would you use vision models vs traditional text extraction?

**Strong answer:**

"Vision preferred for:
- Forms with boxes, checkboxes
- Tables with complex spanning cells
- Documents with mixed content (text + diagrams)
- Scanned documents with poor OCR

Traditional preferred for:
- Text-heavy documents
- Known, simple formats
- High volume, cost-sensitive
- Digital PDFs with good text layer

I use a hybrid approach: text extraction first (cheap), vision for complex pages or validation."

---

## References

- GPT-4 Vision: https://platform.openai.com/docs/guides/vision
- Document AI: https://cloud.google.com/document-ai

---

*Previous: [Fundamentals](01-fundamentals.md)*

# Document Processing Fundamentals

Processing documents for LLM applications requires handling diverse formats, extracting structure, and preparing content for retrieval or generation.

## Table of Contents

- [Document Processing Challenges](#document-processing-challenges)
- [Format Handling](#format-handling)
- [Text Extraction](#text-extraction)
- [Structure Preservation](#structure-preservation)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Quality Assurance](#quality-assurance)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Document Processing Challenges

### Common Issues

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Format variety** | PDF, DOCX, HTML, images | Multi-format parsers |
| **Layout complexity** | Tables, columns, headers | Layout-aware extraction |
| **Poor OCR** | Garbled text from scans | Document AI, preprocessing |
| **Lost structure** | Headers, lists become flat text | Structure preservation |
| **Large files** | Memory issues, slow processing | Streaming, chunking |

### Document Types

| Type | Extraction Method | Complexity |
|------|-------------------|------------|
| Plain text | Direct read | Low |
| Markdown | Parse structure | Low |
| HTML | Parse DOM, clean | Medium |
| PDF (digital) | PDF library | Medium |
| PDF (scanned) | OCR | High |
| DOCX | XML parsing | Medium |
| Spreadsheets | Cell extraction | Medium |
| Images with text | OCR + vision | High |

---

## Format Handling

### PDF Processing

```python
from pypdf import PdfReader
import pdfplumber

# PyPDF for simple text extraction
def extract_with_pypdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# pdfplumber for tables and layout
def extract_with_pdfplumber(path: str) -> dict:
    with pdfplumber.open(path) as pdf:
        result = {"text": [], "tables": []}
        
        for page in pdf.pages:
            result["text"].append(page.extract_text())
            tables = page.extract_tables()
            for table in tables:
                result["tables"].append(table)
        
        return result
```

### DOCX Processing

```python
from docx import Document

def extract_docx_structured(path: str) -> list[dict]:
    doc = Document(path)
    elements = []
    
    for para in doc.paragraphs:
        element = {
            "type": "paragraph",
            "text": para.text,
            "style": para.style.name if para.style else None
        }
        
        if para.style and "Heading" in para.style.name:
            element["type"] = "heading"
            element["level"] = int(para.style.name.split()[-1]) if para.style.name[-1].isdigit() else 1
        
        elements.append(element)
    
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text for cell in row.cells]
            rows.append(cells)
        elements.append({"type": "table", "rows": rows})
    
    return elements
```

### HTML Processing

```python
from bs4 import BeautifulSoup
import html2text

def html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.body_width = 0
    return converter.handle(html)

def extract_html_structured(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    
    result = {
        "title": soup.title.string if soup.title else None,
        "headings": [],
        "paragraphs": []
    }
    
    for i in range(1, 7):
        for heading in soup.find_all(f"h{i}"):
            result["headings"].append({
                "level": i,
                "text": heading.get_text(strip=True)
            })
    
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            result["paragraphs"].append(text)
    
    return result
```

---

## Text Extraction

### OCR for Scanned Documents

```python
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def ocr_pdf(path: str, dpi: int = 300) -> str:
    images = convert_from_path(path, dpi=dpi)
    
    text = ""
    for i, image in enumerate(images):
        processed = preprocess_for_ocr(image)
        page_text = pytesseract.image_to_string(processed, lang="eng")
        text += f"\n--- Page {i+1} ---\n{page_text}"
    
    return text

def preprocess_for_ocr(image: Image) -> Image:
    import cv2
    import numpy as np
    
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary)
```

---

## Structure Preservation

### Hierarchical Extraction

```python
class DocumentStructure:
    def __init__(self):
        self.title = None
        self.sections = []
    
    def add_section(self, heading: str, level: int, content: str):
        self.sections.append({
            "heading": heading,
            "level": level,
            "content": content
        })
    
    def to_chunks_with_context(self, chunk_size: int = 500) -> list[dict]:
        chunks = []
        
        for section in self.sections:
            section_text = section["content"]
            section_chunks = self.split_text(section_text, chunk_size)
            
            for i, chunk_text in enumerate(section_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section": section["heading"],
                        "section_level": section["level"],
                        "chunk_index": i
                    }
                })
        
        return chunks
```

---

## Preprocessing Pipeline

```python
class DocumentProcessor:
    def __init__(self):
        self.extractors = {
            ".pdf": self.extract_pdf,
            ".docx": self.extract_docx,
            ".html": self.extract_html,
            ".txt": self.extract_text
        }
    
    async def process(self, file_path: str) -> ProcessedDocument:
        ext = Path(file_path).suffix.lower()
        extractor = self.extractors.get(ext)
        
        if not extractor:
            raise UnsupportedFormatError(ext)
        
        raw_content = await extractor(file_path)
        cleaned = self.clean_content(raw_content)
        structured = self.extract_structure(cleaned)
        chunks = self.chunk_content(structured)
        
        return ProcessedDocument(
            source=file_path,
            chunks=chunks
        )
    
    def clean_content(self, content: str) -> str:
        content = " ".join(content.split())
        content = content.encode("utf-8", errors="ignore").decode("utf-8")
        return content
```

---

## Quality Assurance

```python
class ExtractionValidator:
    def validate(self, original: str, extracted: str) -> ValidationResult:
        issues = []
        
        if len(extracted) < len(original) * 0.5:
            issues.append("Significant content loss detected")
        
        garbled_ratio = self.detect_garbled_text(extracted)
        if garbled_ratio > 0.1:
            issues.append(f"Garbled text detected: {garbled_ratio:.1%}")
        
        return ValidationResult(valid=len(issues) == 0, issues=issues)
```

---

## Interview Questions

### Q: How do you handle different document formats in a RAG pipeline?

**Strong answer:**

"I build a format-aware document processing pipeline:

**Format detection:** Identify file type by extension and magic bytes.

**Specialized extractors:** PDF (pdfplumber for digital, OCR for scanned), DOCX (python-docx), HTML (BeautifulSoup).

**Structure preservation:** Extract headings, tables, lists as metadata for better chunking and retrieval.

**Quality validation:** Check for extraction errors before indexing.

The key is that extraction quality directly impacts RAG quality."

---

## References

- pdfplumber: https://github.com/jsvine/pdfplumber
- Tesseract: https://github.com/tesseract-ocr/tesseract

---

*Next: [Multimodal Understanding](02-multimodal-understanding.md)*

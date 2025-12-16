# Chunking Strategies

Chunking is how you split documents into retrievable units. It is one of the most impactful decisions in RAG system design. This chapter covers chunking strategies, tradeoffs, and best practices.

## Table of Contents

- [Why Chunking Matters](#why-chunking-matters)
- [Chunking Parameters](#chunking-parameters)
- [Chunking Strategies](#chunking-strategies)
- [Content-Aware Chunking](#content-aware-chunking)
- [Parent-Child Chunking](#parent-child-chunking)
- [Chunking for Different Content Types](#chunking-for-different-content-types)
- [Chunking Pipeline Implementation](#chunking-pipeline-implementation)
- [Evaluation and Optimization](#evaluation-and-optimization)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Chunking Matters

### The Fundamental Tradeoff

| Chunk Size | Retrieval Precision | Context Quality | Trade-off |
|------------|--------------------|-----------------| ----------|
| Small (100 tokens) | High | May lack context | Precise but incomplete |
| Medium (500 tokens) | Balanced | Usually sufficient | Good default |
| Large (1000+ tokens) | Low | Rich context | May include noise |

**The core tension:**
- Smaller chunks = more precise matching, but may miss context
- Larger chunks = more context, but may include irrelevant information

### Impact on RAG Quality

Poor chunking is the #1 cause of RAG failure in production systems.

**Failure examples:**
1. **Split mid-sentence:** "The capital of France is" | "Paris, which was founded..."
2. **Broken tables:** Header in one chunk, data in another
3. **Lost context:** Answer requires information from multiple chunks
4. **Noise inclusion:** Large chunk includes unrelated paragraphs

---

## Chunking Parameters

### Chunk Size

How large each chunk should be (in tokens or characters).

| Size | Tokens | Use Case |
|------|--------|----------|
| Small | 100-200 | Precise Q&A, fine-grained retrieval |
| Medium | 400-600 | General purpose, most common |
| Large | 800-1500 | Document summaries, broad context |

**Recommended starting point:** 500 tokens

**Considerations:**
- Embedding model max tokens (often 512)
- LLM context budget for retrieved chunks
- Content type (technical vs. narrative)

### Chunk Overlap

How much content is shared between adjacent chunks.

```
Document: "A B C D E F G H I J"

No overlap (chunks of 4):
  Chunk 1: "A B C D"
  Chunk 2: "E F G H"
  
With 2-token overlap:
  Chunk 1: "A B C D"
  Chunk 2: "C D E F"
  Chunk 3: "E F G H"
```

**Why overlap?**
- Prevents losing context at chunk boundaries
- Important sentences may span chunk boundaries
- Typical overlap: 10-20% of chunk size

**Recommendation:** 50-100 token overlap for 500 token chunks

---

## Chunking Strategies

### Strategy 1: Fixed-Size Chunking

Split every N tokens/characters regardless of content.

```python
def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = tokenize(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = detokenize(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks
```

**Pros:**
- Simple, predictable
- Consistent chunk sizes
- Easy to implement

**Cons:**
- Ignores document structure
- May split sentences, paragraphs, code blocks
- Not optimal for any specific content type

**Use when:** Quick prototyping, homogeneous content

### Strategy 2: Recursive Character Splitting

Split by hierarchy of separators, respecting document structure.

```python
def recursive_split(
    text: str,
    chunk_size: int,
    separators: list[str] = ["\n\n", "\n", ". ", " "]
) -> list[str]:
    chunks = []
    
    # Try each separator in order
    for separator in separators:
        if separator in text:
            splits = text.split(separator)
            
            current_chunk = ""
            for split in splits:
                if len(current_chunk) + len(split) < chunk_size:
                    current_chunk += split + separator
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = split + separator
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
    
    # Fallback: character split
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

**Separator hierarchy:**
1. `\n\n` - Paragraph boundaries
2. `\n` - Line breaks
3. `. ` - Sentence boundaries
4. ` ` - Word boundaries

**Pros:**
- Respects some document structure
- Better than fixed-size
- Works on most text

**Cons:**
- Does not understand semantic units
- May still split related content
- Separator-dependent

**Use when:** General-purpose documents, no special structure

### Strategy 3: Semantic Chunking

Split based on semantic similarity between sentences.

```python
def semantic_chunk(text: str, threshold: float = 0.5) -> list[str]:
    sentences = split_sentences(text)
    embeddings = embed_batch(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate similarity to current chunk
        similarity = cosine_similarity(
            embeddings[i],
            mean(embeddings[j] for j in current_chunk_indices)
        )
        
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(" ".join(current_chunk))
    return chunks
```

**Pros:**
- Keeps semantically related content together
- Adapts to content
- Good for varied documents

**Cons:**
- Expensive (requires embeddings)
- Variable chunk sizes
- Threshold tuning required

**Use when:** Mixed content, varying topics within documents

### Strategy 4: Markdown/Structure-Based

Parse document structure and chunk by headers, sections.

```python
def markdown_chunk(markdown_text: str, max_size: int = 1000) -> list[str]:
    sections = parse_markdown_sections(markdown_text)
    chunks = []
    
    for section in sections:
        if len(section.content) <= max_size:
            chunks.append({
                "text": section.content,
                "metadata": {"header": section.header, "level": section.level}
            })
        else:
            # Sub-chunk large sections
            sub_chunks = recursive_split(section.content, max_size)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "metadata": {"header": section.header, "level": section.level}
                })
    
    return chunks
```

**Pros:**
- Respects document hierarchy
- Natural section boundaries
- Preserves headers as metadata

**Cons:**
- Requires structured documents
- Variable chunk sizes
- Format-specific

**Use when:** Documentation, wikis, structured content

### Strategy Comparison

| Strategy | Best For | Chunk Size Variance | Implementation Complexity |
|----------|----------|---------------------|---------------------------|
| Fixed-size | Simple content | None | Low |
| Recursive | General text | Low-Medium | Low |
| Semantic | Varied topics | High | High |
| Structure-based | Markdown, HTML | High | Medium |

---

## Content-Aware Chunking

### Code Chunking

Code requires special handling to preserve function/class boundaries:

```python
def chunk_code(code: str, language: str) -> list[str]:
    tree = parse_ast(code, language)
    chunks = []
    
    for node in tree.children:
        if node.type in ["function", "class", "method"]:
            # Include docstring + signature + body
            chunk = extract_definition(node)
            
            if len(chunk) > MAX_CHUNK_SIZE:
                # Large function: split by logical blocks
                sub_chunks = split_function_body(node)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)
        elif node.type == "import":
            # Group imports
            continue  # Handle separately
    
    return chunks
```

**Key principles for code:**
- Never split mid-function
- Include docstrings with their functions
- Preserve import context
- Consider class-method relationships

### Table Chunking

Tables must stay together or be handled specially:

```python
def chunk_with_tables(document: str) -> list[str]:
    chunks = []
    current_pos = 0
    
    for table in find_tables(document):
        # Chunk text before table
        pre_text = document[current_pos:table.start]
        chunks.extend(recursive_split(pre_text))
        
        # Keep table together if not too large
        if table.size < MAX_TABLE_SIZE:
            chunks.append({
                "text": table.text,
                "type": "table",
                "metadata": table.caption
            })
        else:
            # Split large tables by rows, keeping header
            header = table.header_row
            for row_group in batch(table.data_rows, 10):
                chunks.append({
                    "text": format_table([header] + row_group),
                    "type": "table_fragment"
                })
        
        current_pos = table.end
    
    # Chunk remaining text
    chunks.extend(recursive_split(document[current_pos:]))
    return chunks
```

### PDF Chunking

PDFs require layout awareness:

```python
def chunk_pdf(pdf_path: str) -> list[str]:
    # Use layout-aware extraction
    pages = extract_with_layout(pdf_path)  # e.g., PyMuPDF, Unstructured
    
    chunks = []
    for page in pages:
        for element in page.elements:
            if element.type == "paragraph":
                chunks.append({
                    "text": element.text,
                    "page": page.number,
                    "bbox": element.bbox
                })
            elif element.type == "table":
                chunks.append({
                    "text": element.to_markdown(),
                    "type": "table",
                    "page": page.number
                })
            elif element.type == "figure":
                chunks.append({
                    "text": element.caption,
                    "type": "figure",
                    "page": page.number
                })
    
    # Merge small adjacent paragraphs
    return merge_small_chunks(chunks, min_size=200)
```

---

## Parent-Child Chunking

A pattern that retrieves small chunks but returns larger context:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parent Document (Full Section)               │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Child 1   │  │   Child 2   │  │   Child 3   │            │
│   │ (embedded)  │  │ (embedded)  │  │ (embedded)  │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Search:  Match on Child 2
Return:  Full Parent (provides context)
```

### Implementation

```python
class ParentChildChunker:
    def __init__(self, parent_size=2000, child_size=400):
        self.parent_size = parent_size
        self.child_size = child_size
    
    def chunk(self, document: str) -> tuple[list[str], list[str], dict]:
        # Create parent chunks
        parents = recursive_split(document, self.parent_size)
        
        # Create child chunks with parent reference
        children = []
        child_to_parent = {}
        
        for parent_id, parent in enumerate(parents):
            parent_children = recursive_split(parent, self.child_size)
            for child in parent_children:
                child_id = len(children)
                children.append(child)
                child_to_parent[child_id] = parent_id
        
        return parents, children, child_to_parent
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Search children
        child_results = vector_db.search(
            embed(query),
            collection="children",
            top_k=top_k * 2  # Get more to dedupe parents
        )
        
        # Map to parents and dedupe
        parent_ids = set()
        parents = []
        for child in child_results:
            parent_id = self.child_to_parent[child.id]
            if parent_id not in parent_ids:
                parent_ids.add(parent_id)
                parents.append(self.parents[parent_id])
            if len(parents) >= top_k:
                break
        
        return parents
```

### Benefits

| Aspect | Standard Chunking | Parent-Child |
|--------|-------------------|--------------|
| Retrieval precision | Good | Very good (small children) |
| Context quality | May be limited | Rich (full parent) |
| Storage | 1x | ~1.5x (both levels) |
| Complexity | Simple | Moderate |

**Use when:**
- Documents have natural parent sections
- Need both precise matching and rich context
- Context around the match is important

---

## Chunking for Different Content Types

### Recommended Strategies by Content Type

| Content Type | Strategy | Chunk Size | Special Handling |
|--------------|----------|------------|------------------|
| Documentation | Structure-based | 500-800 | Preserve headers |
| Legal contracts | Semantic + clause detection | 400-600 | Keep clauses together |
| Code | AST-based | Function size | Never split functions |
| Research papers | Section-based | 600-1000 | Handle citations |
| Chat logs | Message-based | Variable | Keep exchanges together |
| Product catalogs | Item-based | Per item | Structured fields |
| FAQ | Q&A pair | Per pair | Keep Q with A |

### Configuration Examples

```python
CHUNKING_CONFIG = {
    "documentation": {
        "strategy": "markdown_structure",
        "chunk_size": 600,
        "overlap": 50,
        "preserve_headers": True,
        "min_chunk_size": 100
    },
    "code": {
        "strategy": "ast_based",
        "languages": ["python", "javascript", "java"],
        "include_docstrings": True,
        "max_function_size": 2000
    },
    "legal": {
        "strategy": "semantic",
        "chunk_size": 500,
        "similarity_threshold": 0.6,
        "detect_clauses": True
    },
    "default": {
        "strategy": "recursive",
        "chunk_size": 500,
        "overlap": 50,
        "separators": ["\n\n", "\n", ". ", " "]
    }
}
```

---

## Chunking Pipeline Implementation

### Production Pipeline

```python
class ChunkingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.strategies = {
            "recursive": RecursiveChunker,
            "semantic": SemanticChunker,
            "markdown_structure": MarkdownChunker,
            "ast_based": CodeChunker
        }
    
    def process(self, document: Document) -> list[Chunk]:
        # 1. Detect content type
        content_type = self.detect_type(document)
        config = self.config.get(content_type, self.config["default"])
        
        # 2. Extract text
        text = self.extract_text(document)
        
        # 3. Preprocess
        text = self.preprocess(text)
        
        # 4. Apply strategy
        strategy = self.strategies[config["strategy"]](config)
        raw_chunks = strategy.chunk(text)
        
        # 5. Post-process
        chunks = self.postprocess(raw_chunks, document)
        
        # 6. Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": document.source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_type": content_type,
                **self.extract_metadata(chunk)
            }
        
        return chunks
    
    def preprocess(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters if needed
        text = self.clean_special_chars(text)
        return text
    
    def postprocess(self, chunks: list[str], document: Document) -> list[Chunk]:
        processed = []
        for chunk_text in chunks:
            # Skip empty or too small
            if len(chunk_text.strip()) < 50:
                continue
            
            # Truncate if too large
            if len(chunk_text) > self.config["max_chunk_size"]:
                chunk_text = chunk_text[:self.config["max_chunk_size"]]
            
            processed.append(Chunk(text=chunk_text))
        
        return processed
```

---

## Evaluation and Optimization

### Metrics for Chunking Quality

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| Boundary coherence | Do chunks start/end at natural points? | Manual review or LLM judge |
| Context completeness | Do chunks contain enough context? | Retrieval success rate |
| Chunk size variance | How consistent are chunk sizes? | Standard deviation |
| Retrieval precision | Does good chunking improve retrieval? | A/B test different strategies |

### A/B Testing Chunking Strategies

```python
def evaluate_chunking(strategy_a, strategy_b, test_set):
    results = {"a": [], "b": []}
    
    for query, relevant_docs in test_set:
        # Index with strategy A
        chunks_a = strategy_a.chunk(corpus)
        index_a = build_index(chunks_a)
        retrieved_a = search(index_a, query, k=5)
        results["a"].append(compute_recall(retrieved_a, relevant_docs))
        
        # Index with strategy B
        chunks_b = strategy_b.chunk(corpus)
        index_b = build_index(chunks_b)
        retrieved_b = search(index_b, query, k=5)
        results["b"].append(compute_recall(retrieved_b, relevant_docs))
    
    print(f"Strategy A recall: {mean(results['a']):.3f}")
    print(f"Strategy B recall: {mean(results['b']):.3f}")
    
    # Statistical significance
    p_value = ttest(results["a"], results["b"])
    print(f"P-value: {p_value:.4f}")
```

---

## Interview Questions

### Q: How would you approach chunking for a legal document RAG system?

**Strong answer:**
Legal documents have specific structure that chunking must respect:

**Strategy:**
1. Parse document structure (sections, clauses, definitions)
2. Keep individual clauses together (never split mid-clause)
3. Preserve references and definitions in context
4. Use parent-child pattern: clause for retrieval, section for context

**Implementation considerations:**
- Detect clause markers ("Section 4.2", "Whereas")
- Handle cross-references (definitions section should be available)
- Preserve numbered lists and sub-clauses
- Include surrounding context for interpretation

**Chunk size:** 400-600 tokens to fit typical clause length
**Overlap:** Minimal within clauses, overlap at section boundaries
**Metadata:** Clause number, section, document structure position

### Q: What is wrong with fixed-size chunking and when would you still use it?

**Strong answer:**
**Problems with fixed-size chunking:**
1. Ignores document structure (splits mid-sentence, mid-paragraph)
2. May break tables, code blocks, lists
3. Separates related content
4. No semantic awareness

**When I would still use it:**
- Quick prototyping where speed matters more than quality
- Homogeneous content with no clear structure
- When other strategies are computationally expensive
- As a baseline for comparing other strategies

For production, I would always evaluate it against structure-aware alternatives.

### Q: Explain the parent-child chunking pattern.

**Strong answer:**
Parent-child chunking uses two levels of chunks:

**Children:** Small chunks (200-400 tokens) used for embedding and retrieval. Small size enables precise semantic matching.

**Parents:** Larger chunks (1000-2000 tokens) containing the children. Not embedded, but returned when a child matches.

**How it works:**
1. Chunk documents into parents (sections, paragraphs)
2. Further chunk parents into children
3. Embed and index only children
4. When a child matches query, return its parent

**Benefits:**
- Precise retrieval: small chunks match specific queries
- Rich context: parent provides surrounding information
- Best of both worlds

**Storage cost:** ~1.5x (store both levels)
**Use case:** Any scenario where the context around a match is important

---

## References

- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- LlamaIndex Node Parsers: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
- Unstructured: https://unstructured.io/
- "Chunk Wisely, RAG More" (various blog posts on chunking best practices)

---

*Previous: [RAG Fundamentals](01-rag-fundamentals.md) | Next: [Vector Databases](03-vector-databases.md)*

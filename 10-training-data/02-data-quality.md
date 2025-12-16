# Data Quality for LLM Applications

Data quality is the foundation of effective LLM applications. This chapter covers quality assessment, improvement, and maintenance.

## Table of Contents

- [Why Data Quality Matters](#why-data-quality-matters)
- [Quality Dimensions](#quality-dimensions)
- [Assessment Methods](#assessment-methods)
- [Data Cleaning](#data-cleaning)
- [Quality Monitoring](#quality-monitoring)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Data Quality Matters

### Impact of Poor Data

| Quality Issue | Impact on RAG | Impact on Fine-Tuning |
|---------------|---------------|----------------------|
| Duplicates | Redundant retrieval | Overfitting |
| Outdated info | Wrong answers | Stale knowledge |
| Inconsistent format | Poor chunking | Unstable outputs |
| Incorrect labels | N/A | Wrong behavior |
| Missing data | Incomplete context | Gaps in capability |

### Quality vs Quantity

```
                    │
    High Quality    │    High Quality
    Low Quantity    │    High Quantity
         ◉          │          ◉
      (Usable)      │      (Optimal)
────────────────────┼────────────────────
                    │
    Low Quality     │    Low Quality
    Low Quantity    │    High Quantity
         ◉          │          ◉
      (Unusable)    │     (Dangerous)
                    │
```

Low quality + high quantity is particularly dangerous: it looks sufficient but produces unreliable results.

---

## Quality Dimensions

### For Knowledge Bases (RAG)

| Dimension | Description | Measurement |
|-----------|-------------|-------------|
| **Accuracy** | Information is correct | Expert review, fact-checking |
| **Completeness** | All relevant info included | Coverage analysis |
| **Freshness** | Information is current | Timestamp analysis |
| **Consistency** | No contradictions | Contradiction detection |
| **Relevance** | Info is on-topic | Topic classification |
| **Clarity** | Well-written, unambiguous | Readability scores |

### For Training Data

| Dimension | Description | Measurement |
|-----------|-------------|-------------|
| **Label accuracy** | Correct input-output pairs | Inter-annotator agreement |
| **Diversity** | Covers input distribution | Embedding clustering |
| **Balance** | Proportional class representation | Class distribution |
| **Consistency** | Similar inputs have similar outputs | Duplicate analysis |
| **Representative** | Matches production distribution | Distribution comparison |

---

## Assessment Methods

### Automated Quality Checks

```python
class DataQualityAssessor:
    def __init__(self):
        self.checks = [
            CompletenessCheck(),
            DuplicateCheck(),
            FormatCheck(),
            FreshnessCheck()
        ]
    
    async def assess(self, documents: list[dict]) -> QualityReport:
        results = {}
        
        for check in self.checks:
            results[check.name] = await check.run(documents)
        
        return QualityReport(
            overall_score=self.calculate_overall_score(results),
            dimension_scores=results,
            issues=self.extract_issues(results),
            recommendations=self.generate_recommendations(results)
        )


class CompletenessCheck:
    REQUIRED_FIELDS = ["content", "source", "date", "title"]
    
    async def run(self, documents: list[dict]) -> CheckResult:
        missing_fields = []
        
        for doc in documents:
            for field in self.REQUIRED_FIELDS:
                if not doc.get(field):
                    missing_fields.append({
                        "doc_id": doc.get("id"),
                        "missing_field": field
                    })
        
        return CheckResult(
            score=1 - (len(missing_fields) / (len(documents) * len(self.REQUIRED_FIELDS))),
            issues=missing_fields
        )


class DuplicateCheck:
    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
    
    async def run(self, documents: list[dict]) -> CheckResult:
        # Embed all documents
        embeddings = await embed_batch([d["content"] for d in documents])
        
        # Find near-duplicates
        duplicates = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim > self.threshold:
                    duplicates.append({
                        "doc1": documents[i]["id"],
                        "doc2": documents[j]["id"],
                        "similarity": sim
                    })
        
        return CheckResult(
            score=1 - (len(duplicates) / len(documents)),
            issues=duplicates
        )
```

### LLM-Based Quality Scoring

```python
class LLMQualityScorer:
    async def score_document(self, document: dict) -> dict:
        prompt = f"""
Evaluate this document for quality.

Document:
Title: {document['title']}
Content: {document['content'][:2000]}

Rate each dimension 1-5:
1. Accuracy: Is the information likely to be correct?
2. Clarity: Is it well-written and easy to understand?
3. Completeness: Does it cover the topic thoroughly?
4. Relevance: Is the content focused and on-topic?

Return as JSON with scores and brief justification for each.
"""
        
        result = await self.llm.generate(prompt)
        return json.loads(result)
    
    async def score_dataset(self, documents: list[dict], sample_size: int = 100) -> dict:
        # Sample for efficiency
        sample = random.sample(documents, min(sample_size, len(documents)))
        
        scores = await asyncio.gather(*[
            self.score_document(doc) for doc in sample
        ])
        
        return {
            "avg_accuracy": np.mean([s["accuracy"] for s in scores]),
            "avg_clarity": np.mean([s["clarity"] for s in scores]),
            "avg_completeness": np.mean([s["completeness"] for s in scores]),
            "avg_relevance": np.mean([s["relevance"] for s in scores]),
            "low_quality_docs": [s for s in scores if s["accuracy"] < 3]
        }
```

---

## Data Cleaning

### Deduplication

```python
class Deduplicator:
    def __init__(
        self,
        exact_threshold: float = 1.0,
        near_threshold: float = 0.90
    ):
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
    
    async def deduplicate(self, documents: list[dict]) -> list[dict]:
        # Phase 1: Exact duplicate removal (hash-based)
        seen_hashes = set()
        after_exact = []
        
        for doc in documents:
            content_hash = hash(doc["content"])
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                after_exact.append(doc)
        
        # Phase 2: Near-duplicate removal (embedding-based)
        embeddings = await embed_batch([d["content"] for d in after_exact])
        
        keep = []
        keep_embeddings = []
        
        for doc, emb in zip(after_exact, embeddings):
            is_duplicate = False
            for kept_emb in keep_embeddings:
                if cosine_similarity(emb, kept_emb) > self.near_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(doc)
                keep_embeddings.append(emb)
        
        return keep
```

### Content Normalization

```python
class ContentNormalizer:
    def normalize(self, text: str) -> str:
        # Whitespace normalization
        text = " ".join(text.split())
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Quote normalization
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # Remove control characters
        text = "".join(c for c in text if unicodedata.category(c) != "Cc")
        
        return text
```

### Freshness Management

```python
class FreshnessManager:
    def __init__(self, stale_threshold_days: int = 365):
        self.stale_threshold = timedelta(days=stale_threshold_days)
    
    def assess_freshness(self, documents: list[dict]) -> dict:
        now = datetime.now()
        
        fresh = []
        stale = []
        unknown = []
        
        for doc in documents:
            if not doc.get("date"):
                unknown.append(doc)
            elif now - doc["date"] > self.stale_threshold:
                stale.append(doc)
            else:
                fresh.append(doc)
        
        return {
            "fresh": fresh,
            "stale": stale,
            "unknown_date": unknown,
            "freshness_score": len(fresh) / len(documents) if documents else 0
        }
    
    def prioritize_update(self, stale: list[dict]) -> list[dict]:
        # Prioritize by: importance, access frequency, staleness
        return sorted(stale, key=lambda d: (
            -d.get("access_count", 0),
            -(datetime.now() - d["date"]).days
        ))
```

---

## Quality Monitoring

### Continuous Monitoring

```python
class DataQualityMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    async def monitor(self, data_source: str):
        # Run periodic checks
        documents = await self.fetch_documents(data_source)
        quality = await self.assessor.assess(documents)
        
        # Record metrics
        self.metrics[data_source] = {
            "timestamp": datetime.now(),
            "overall_score": quality.overall_score,
            "dimension_scores": quality.dimension_scores
        }
        
        # Check for degradation
        if self.has_degraded(data_source, quality):
            await self.alert(AlertType.QUALITY_DEGRADATION, {
                "source": data_source,
                "current_score": quality.overall_score,
                "previous_score": self.get_previous_score(data_source)
            })
    
    def has_degraded(self, source: str, current: QualityReport) -> bool:
        previous = self.get_previous_score(source)
        if previous is None:
            return False
        return current.overall_score < previous * 0.9  # 10% drop
```

### Quality Dashboard Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Duplicate rate | % near-duplicate documents | > 5% |
| Staleness rate | % documents > 1 year old | > 20% |
| Completeness | % documents with all fields | < 95% |
| Format errors | % with parsing issues | > 1% |
| Quality score | LLM-assessed average | < 3.5/5 |

---

## Interview Questions

### Q: How do you ensure data quality for a RAG system?

**Strong answer:**

"Data quality for RAG requires assessment across multiple dimensions:

**Accuracy:** Is the information correct? I use a combination of automated fact-checking against trusted sources and sampled expert review.

**Freshness:** Is it current? I track document dates and flag stale content for update or removal. For rapidly changing domains, I set aggressive staleness thresholds.

**Completeness:** Does it cover the needed topics? I analyze query logs against document coverage to identify gaps.

**Consistency:** Are there contradictions? I use embedding-based comparison to find documents that discuss the same topic differently.

**Implementation:**
1. Automated quality checks in the ingestion pipeline
2. LLM-based quality scoring for sampled documents
3. Freshness monitoring with automated alerts
4. Deduplication before indexing
5. Metadata enrichment for better filtering

I monitor retrieval quality in production: if users consistently need to rephrase or get poor answers, it often traces back to data quality issues."

### Q: How would you handle training data with label noise?

**Strong answer:**

"Label noise damages fine-tuning quality. My approach:

**Detection:**
1. Inter-annotator agreement analysis if multiple annotators
2. LLM-based label verification on samples
3. Confidence scoring on predictions to find hard/mislabeled examples

**Mitigation:**
1. Confident learning: train, identify high-loss examples, review and correct
2. Multiple annotator consensus: require 2/3 agreement
3. Exclude uncertain examples rather than include wrong ones

**Prevention:**
1. Clear annotation guidelines with examples
2. Annotator training and calibration
3. Regular quality audits of annotations

The key principle: fewer correct examples beat more noisy examples. I would rather have 500 verified examples than 2000 with 20% noise."

---

## References

- Data-Centric AI: https://dcai.csail.mit.edu/
- Cleanlab: https://github.com/cleanlab/cleanlab

---

*Next: [Data Augmentation](03-data-augmentation.md)*

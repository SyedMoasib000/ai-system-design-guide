# Case Study: Financial Analysis with Ensemble Verification

This case study covers designing a high-reliability AI system for generating equity research reports where accuracy is critical.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Requirements Analysis](#requirements-analysis)
- [Architecture Design](#architecture-design)
- [Ensemble Pipeline](#ensemble-pipeline)
- [Fact Verification](#fact-verification)
- [Quality Gates](#quality-gates)
- [Results and Metrics](#results-and-metrics)
- [Interview Walkthrough](#interview-walkthrough)

---

## Problem Statement

**Company:** Investment firm generating equity research reports

**Challenge:**
- Reports influence multi-million dollar investment decisions
- Zero tolerance for hallucinated financial data
- Regulatory scrutiny on AI-generated analysis
- Current manual process: 8 hours per report, $500 cost

**Goal:**
- Reduce report generation time to < 30 minutes
- Maintain accuracy at 99.5%+
- Clear audit trail for compliance
- Cost target: < $50 per report

---

## Requirements Analysis

### Accuracy Requirements

| Data Type | Tolerance | Verification Method |
|-----------|-----------|---------------------|
| Financial metrics (EPS, PE) | 0% error | Source verification |
| Percentage changes | ±0.1% | Cross-validation |
| Date references | 100% accuracy | Source extraction |
| Company names | 100% accuracy | Entity matching |
| Analyst quotes | Verbatim or flagged | Quote extraction |

### Compliance Requirements

- All claims must cite source documents
- No forward-looking statements without disclaimers
- Clear AI-generated disclosure
- Full audit trail of generation process
- Human review for publication

---

## Architecture Design

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│               FINANCIAL ANALYSIS PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Data Extraction (Self-Consistency k=5)                │
│  └── Extract key metrics from filings with majority vote        │
│                                                                  │
│  Stage 2: Analysis Generation (Mixture of Agents)               │
│  ├── Model A: Quantitative analysis focus                       │
│  ├── Model B: Qualitative/narrative focus                       │
│  ├── Model C: Risk factor analysis                              │
│  └── Aggregator: Synthesize into coherent report                │
│                                                                  │
│  Stage 3: Fact Verification (Multi-Agent Debate)                │
│  └── 3 models debate each factual claim, flag disagreements     │
│                                                                  │
│  Stage 4: Final Review (Panel of Judges)                        │
│  └── Quality score determines auto-publish vs human review      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   10-K/Q    │     │  Earnings   │     │  Analyst    │
│   Filings   │     │  Calls      │     │  Reports    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │     Data      │
                   │   Ingestion   │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │   Extraction  │
                   │  (k=5 SC)     │
                   └───────┬───────┘
                           │
                           ▼
              ┌────────────┴────────────┐
              │    Structured Data      │
              │    (verified metrics)   │
              └────────────┬────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │    MoA        │
                   │  Generation   │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │    Debate     │
                   │  Verification │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │    Panel      │
                   │    Review     │
                   └───────┬───────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
        ┌─────────────┐         ┌─────────────┐
        │ Auto-Publish│         │Human Review │
        │ (high conf) │         │ (low conf)  │
        └─────────────┘         └─────────────┘
```

---

## Ensemble Pipeline

### Stage 1: Data Extraction with Self-Consistency

```python
class FinancialDataExtractor:
    """
    Extract financial metrics with self-consistency for reliability.
    Critical: Financial data must be 100% accurate.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        self.model = GPT4o()
    
    async def extract_metrics(self, document: str, metrics: list[str]) -> dict:
        extraction_prompt = f"""
Extract the following metrics from this financial document.
For each metric, provide the exact value and the source quote.

Metrics to extract: {json.dumps(metrics)}

Document:
{document}

Return as JSON:
{{
    "metric_name": {{
        "value": "exact value",
        "source_quote": "exact quote from document",
        "confidence": "high/medium/low"
    }}
}}
"""
        
        # Generate k extractions
        extractions = await asyncio.gather(*[
            self.model.generate(extraction_prompt, temperature=0.3)
            for _ in range(self.k)
        ])
        
        # Parse and vote on each metric
        results = {}
        for metric in metrics:
            values = []
            for extraction in extractions:
                parsed = json.loads(extraction)
                if metric in parsed:
                    values.append(parsed[metric]["value"])
            
            # Majority vote
            if values:
                value_counts = Counter(values)
                majority_value, count = value_counts.most_common(1)[0]
                
                results[metric] = {
                    "value": majority_value,
                    "agreement": count / len(values),
                    "needs_verification": count < self.k  # Not unanimous
                }
        
        return results
```

### Stage 2: Analysis Generation with Mixture of Agents

```python
class MixtureOfAnalysts:
    """
    Multiple specialized models contribute to the analysis.
    Each has a different focus area.
    """
    
    def __init__(self):
        self.analysts = {
            "quantitative": QuantitativeAnalyst(),
            "qualitative": QualitativeAnalyst(),
            "risk": RiskAnalyst()
        }
        self.aggregator = AggregatorModel()
    
    async def generate_analysis(
        self,
        company_data: dict,
        financial_metrics: dict
    ) -> str:
        # Get perspectives from each analyst
        perspectives = {}
        for name, analyst in self.analysts.items():
            perspectives[name] = await analyst.analyze(company_data, financial_metrics)
        
        # Aggregate into coherent report
        aggregation_prompt = f"""
You have received analysis from three specialized financial analysts.
Synthesize their perspectives into a coherent equity research report.

Quantitative Analysis:
{perspectives['quantitative']}

Qualitative Analysis:
{perspectives['qualitative']}

Risk Analysis:
{perspectives['risk']}

Create a unified report that:
1. Leads with the investment thesis
2. Supports with quantitative evidence
3. Provides balanced qualitative context
4. Clearly states risk factors
5. Ends with a recommendation

Every factual claim must be attributable to the source data.
"""
        
        report = await self.aggregator.generate(aggregation_prompt)
        return report
```

### Stage 3: Fact Verification with Multi-Agent Debate

```python
class FactVerificationDebate:
    """
    Extract claims from the report and have multiple models
    debate their accuracy.
    """
    
    def __init__(self, debaters: list, rounds: int = 2):
        self.debaters = debaters
        self.rounds = rounds
        self.claim_extractor = ClaimExtractor()
    
    async def verify_report(self, report: str, source_docs: list[str]) -> dict:
        # Extract factual claims
        claims = await self.claim_extractor.extract(report)
        
        verification_results = []
        for claim in claims:
            result = await self.debate_claim(claim, source_docs)
            verification_results.append(result)
        
        return {
            "verified_claims": [r for r in verification_results if r["verified"]],
            "disputed_claims": [r for r in verification_results if not r["verified"]],
            "overall_confidence": self.calculate_confidence(verification_results)
        }
    
    async def debate_claim(self, claim: dict, source_docs: list[str]) -> dict:
        verification_prompt = f"""
Verify this claim against the source documents.

Claim: {claim['text']}

Source documents:
{self.format_sources(source_docs)}

Is this claim:
1. Supported: Explicitly stated in sources
2. Inferred: Reasonably derived from sources
3. Unsupported: Not found in sources
4. Contradicted: Conflicts with sources

Provide your verdict with evidence.
"""
        
        # Each debater verifies independently
        verdicts = await asyncio.gather(*[
            debater.generate(verification_prompt)
            for debater in self.debaters
        ])
        
        # Check consensus
        parsed_verdicts = [self.parse_verdict(v) for v in verdicts]
        consensus = self.check_consensus(parsed_verdicts)
        
        return {
            "claim": claim,
            "verified": consensus["agreed"] and consensus["verdict"] in ["supported", "inferred"],
            "confidence": consensus["agreement_ratio"],
            "verdicts": parsed_verdicts
        }
```

---

## Quality Gates

### Automated Quality Checks

```python
class QualityGate:
    def __init__(self):
        self.thresholds = {
            "claim_verification_rate": 0.95,  # 95% claims verified
            "data_accuracy": 0.99,            # 99% metrics accurate
            "panel_score": 4.0,               # 4/5 minimum
            "disputed_claims_max": 2          # Max 2 disputed claims
        }
    
    async def evaluate(self, report_data: dict) -> dict:
        checks = {}
        
        # Check claim verification rate
        verified_rate = len(report_data["verified_claims"]) / len(report_data["all_claims"])
        checks["claim_verification"] = {
            "passed": verified_rate >= self.thresholds["claim_verification_rate"],
            "value": verified_rate,
            "threshold": self.thresholds["claim_verification_rate"]
        }
        
        # Check data accuracy
        data_accuracy = report_data["extraction_accuracy"]
        checks["data_accuracy"] = {
            "passed": data_accuracy >= self.thresholds["data_accuracy"],
            "value": data_accuracy,
            "threshold": self.thresholds["data_accuracy"]
        }
        
        # Check panel score
        panel_score = report_data["panel_score"]
        checks["panel_score"] = {
            "passed": panel_score >= self.thresholds["panel_score"],
            "value": panel_score,
            "threshold": self.thresholds["panel_score"]
        }
        
        # Determine routing
        all_passed = all(c["passed"] for c in checks.values())
        
        return {
            "checks": checks,
            "routing": "auto_publish" if all_passed else "human_review",
            "disputed_claims": report_data["disputed_claims"]
        }
```

### Human Review Interface

```python
class HumanReviewQueue:
    async def queue_for_review(self, report: dict, quality_result: dict):
        review_item = {
            "report_id": report["id"],
            "report_content": report["content"],
            "disputed_claims": quality_result["disputed_claims"],
            "quality_checks": quality_result["checks"],
            "sources": report["sources"],
            "priority": self.calculate_priority(quality_result),
            "queued_at": datetime.now()
        }
        
        await self.review_queue.enqueue(review_item)
        
        # Notify reviewers
        await self.notify_reviewers(review_item)
```

---

## Results and Metrics

### Performance Comparison

| Metric | Manual Process | AI Pipeline | Improvement |
|--------|---------------|-------------|-------------|
| Time per report | 8 hours | 25 minutes | 19x faster |
| Cost per report | $500 | $42 | 92% reduction |
| Factual error rate | 2.1% | 0.4% | 81% reduction |
| Human review load | 100% | 28% | 72% reduction |

### Quality Metrics

| Quality Dimension | Target | Achieved |
|-------------------|--------|----------|
| Data extraction accuracy | 99% | 99.3% |
| Claim verification rate | 95% | 96.8% |
| Panel quality score | 4.0/5.0 | 4.2/5.0 |
| Regulatory compliance | 100% | 100% |

### Cost Breakdown

| Component | Cost | Percentage |
|-----------|------|------------|
| Data extraction (k=5) | $8 | 19% |
| MoA generation | $15 | 36% |
| Debate verification | $12 | 29% |
| Panel review | $5 | 12% |
| Infrastructure | $2 | 5% |
| **Total** | **$42** | 100% |

---

## Interview Walkthrough

**Interviewer:** "Design an AI system for generating financial research reports with very high accuracy requirements."

**Strong response:**

1. **Clarify accuracy requirements** (1 min)
   - "What's the acceptable error rate for financial data?"
   - "What's the regulatory compliance requirement?"
   - "Is latency or accuracy the priority?"

2. **Acknowledge the core challenge** (1 min)
   - "The key challenge is that hallucinations are unacceptable for financial data. A single wrong number could mislead investment decisions. I need ensemble methods for reliability."

3. **High-level architecture** (3 min)
   - "I would use a multi-stage pipeline with different ensemble techniques at each stage:"
   - "Data extraction: Self-consistency with k=5 for unanimous agreement on numbers"
   - "Analysis: Mixture of Agents for diverse perspectives"
   - "Verification: Multi-agent debate to catch hallucinations"
   - "Quality gate: Panel of judges to score before publishing"

4. **Deep dive on fact verification** (3 min)
   - "For fact verification, I extract every factual claim from the report"
   - "Three diverse models debate whether each claim is supported by sources"
   - "If they disagree, the claim is flagged for human review"
   - "This catches subtle errors that single-model verification misses"

5. **Cost-quality tradeoff** (2 min)
   - "This pipeline is 10-20x more expensive than single-model generation"
   - "But for financial reports, the cost of errors (legal, reputational) far exceeds the cost of verification"
   - "I would implement confidence-based routing: auto-publish high-confidence reports, human-review low-confidence ones"

6. **Monitoring** (1 min)
   - "I would track extraction accuracy, claim verification rate, and panel scores continuously"
   - "Drift detection would alert if accuracy drops"
   - "Full audit trail for compliance"

---

## Key Learnings

1. **Self-consistency alone is insufficient** for numerical data extraction. Unanimous agreement (k/k votes) should be required.

2. **Multi-agent debate most effective** for catching subtle reasoning errors and hallucinations.

3. **Source attribution is critical** for both accuracy and compliance. Every claim must link to source documents.

4. **Confidence-based routing** is essential for cost management. Not every report needs full ensemble verification.

5. **Human-in-the-loop is still necessary** for disputed claims and edge cases. Design for graceful escalation.

---

## References

- Verga et al. "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models" (2024)
- Du et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023)
- SEC AI Disclosure Requirements: https://www.sec.gov/

---

*Next: [Code Assistant Case Study](03-code-assistant.md)*

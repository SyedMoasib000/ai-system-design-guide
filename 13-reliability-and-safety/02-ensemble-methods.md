# Ensemble Methods for LLM Reliability

Ensemble methods are critical for production reliability. This chapter covers multi-model coordination patterns that improve accuracy and reduce hallucinations.

## Table of Contents

- [Why Ensembles Matter](#why-ensembles-matter)
- [Evaluation Ensembles](#evaluation-ensembles)
- [Generation Ensembles](#generation-ensembles)
- [Multi-Agent Patterns](#multi-agent-patterns)
- [Ensemble vs Arbitration](#ensemble-vs-arbitration)
- [Cost-Accuracy Tradeoffs](#cost-accuracy-tradeoffs)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Why Ensembles Matter

Single-model outputs are unreliable for high-stakes applications:
- Models hallucinate facts
- Reasoning can be flawed
- Outputs vary with temperature
- Single-judge evaluations are biased

Ensembles improve reliability through redundancy and diversity.

### Ensemble Methods Taxonomy

| Category | Purpose | Methods |
|----------|---------|---------|
| Evaluation | Reduce judge bias | Panel of Judges, Pairwise Comparison |
| Generation | Improve output quality | Self-Consistency, Best-of-N |
| Verification | Reduce hallucinations | Multi-Agent Debate, Fact Checking |
| Synthesis | Combine perspectives | Mixture of Agents |

---

## Evaluation Ensembles

### Panel of LLM Judges (PoLL)

Multiple diverse models score the same output:

```python
class PanelOfJudges:
    """
    Production implementation of PoLL pattern.
    Key insight: Diversity of judges matters more than individual judge quality.
    """
    def __init__(self, judges: list, aggregation: str = "mean"):
        # Use diverse model families, not just different sizes
        # Good: [Claude, GPT-4, Gemini, Llama-70B]
        # Bad: [GPT-4, GPT-4-turbo, GPT-3.5] - same family bias
        self.judges = judges
        self.aggregation = aggregation
    
    async def evaluate(self, question: str, answer: str, rubric: str) -> dict:
        # Parallel evaluation for latency
        judgments = await asyncio.gather(*[
            judge.score(question, answer, rubric) 
            for judge in self.judges
        ])
        
        scores = [j["score"] for j in judgments]
        
        # Track inter-judge agreement for confidence
        agreement = 1 - (np.std(scores) / max(np.mean(scores), 0.01))
        
        if self.aggregation == "mean":
            final_score = np.mean(scores)
        elif self.aggregation == "median":  # More robust to outliers
            final_score = np.median(scores)
        elif self.aggregation == "trimmed_mean":  # Drop highest and lowest
            final_score = np.mean(sorted(scores)[1:-1])
        
        return {
            "score": final_score,
            "confidence": agreement,
            "individual_scores": scores,
            "needs_review": agreement < 0.7  # Flag for human review
        }
```

**When to use:** High-stakes evaluations, benchmark creation, when single-judge bias is unacceptable.

### Pairwise Comparison with Positional Debiasing

Models prefer the first option 60-70% of the time. Always run both orderings:

```python
async def pairwise_compare_debiased(model, response_a: str, response_b: str, criteria: str) -> dict:
    """
    Critical: Models have significant positional bias.
    Always run both orderings and aggregate.
    """
    # Run both orderings in parallel
    result_ab, result_ba = await asyncio.gather(
        model.compare(first=response_a, second=response_b, criteria=criteria),
        model.compare(first=response_b, second=response_a, criteria=criteria)
    )
    
    # If A wins in both positions -> Strong signal for A
    if result_ab["winner"] == "first" and result_ba["winner"] == "second":
        return {"winner": "A", "confidence": "high"}
    
    # If B wins in both positions -> Strong signal for B
    elif result_ab["winner"] == "second" and result_ba["winner"] == "first":
        return {"winner": "B", "confidence": "high"}
    
    # Winner depends on position -> Positional bias detected
    else:
        return {
            "winner": "tie",
            "confidence": "low",
            "note": "Positional bias detected"
        }
```

---

## Generation Ensembles

### Self-Consistency (Majority Voting)

Generate multiple reasoning paths, vote on the final answer:

```python
class SelfConsistencyDecoder:
    """
    Key parameters:
    - k (sample count): 5-10 for most tasks, 15-20 for hard math
    - temperature: 0.5-0.8 for reasoning tasks
    
    Too low temperature = not enough diversity
    Too high temperature = too much noise
    """
    
    def __init__(self, model, k: int = 7, temperature: float = 0.7):
        self.model = model
        self.k = k
        self.temperature = temperature
    
    async def generate_with_consistency(self, prompt: str) -> dict:
        # Generate k reasoning paths in parallel
        responses = await asyncio.gather(*[
            self.model.generate(prompt, temperature=self.temperature)
            for _ in range(self.k)
        ])
        
        # Extract final answers (task-specific)
        answers = [self.extract_answer(r) for r in responses]
        
        # Majority voting
        answer_counts = Counter(answers)
        majority_answer, majority_count = answer_counts.most_common(1)[0]
        
        # Confidence = proportion of votes for winner
        confidence = majority_count / self.k
        
        # Get best reasoning path that led to majority answer
        best_reasoning = self.select_best_reasoning(
            responses, answers, majority_answer
        )
        
        return {
            "answer": majority_answer,
            "confidence": confidence,
            "num_paths": self.k,
            "reasoning": best_reasoning,
            "vote_distribution": dict(answer_counts)
        }
    
    def extract_answer(self, response: str) -> str:
        # Task-specific answer extraction
        # For math: extract the final number
        # For code: extract the function
        # Implement based on your task
        pass
```

**Best for:** Math, logic, coding with verifiable answers. Accuracy gain: 5-15%.

### Best-of-N with Reward Model

Generate N candidates, score with reward model, return best:

```python
class BestOfNSampler:
    """
    Key considerations:
    1. N selection: N=4-8 for interactive, N=16-64 for batch
    2. Reward model ensemble prevents reward hacking
    3. Monitor sample diversity - if too similar, BoN is wasted compute
    """
    
    def __init__(self, generator, reward_models: list, n: int = 8):
        self.generator = generator
        self.reward_models = reward_models  # Ensemble for robustness
        self.n = n
    
    async def generate_best(self, prompt: str) -> dict:
        # Generate N candidates in parallel
        candidates = await asyncio.gather(*[
            self.generator.generate(prompt, temperature=0.8)
            for _ in range(self.n)
        ])
        
        # Score with reward model ensemble
        scored_candidates = []
        for candidate in candidates:
            rm_scores = await asyncio.gather(*[
                rm.score(prompt, candidate) for rm in self.reward_models
            ])
            
            # Conservative aggregation prevents reward hacking
            # Use 25th percentile instead of mean
            conservative_score = np.percentile(rm_scores, 25)
            
            scored_candidates.append({
                "response": candidate,
                "score": conservative_score,
                "rm_agreement": 1 - np.std(rm_scores) / np.mean(rm_scores)
            })
        
        # Select best by conservative score
        best = max(scored_candidates, key=lambda x: x["score"])
        
        # Compute diversity metric
        diversity = self.compute_diversity(candidates)
        
        return {
            "response": best["response"],
            "score": best["score"],
            "n_sampled": self.n,
            "diversity_score": diversity,
            "low_diversity_warning": diversity < 0.3
        }
    
    def compute_diversity(self, candidates: list) -> float:
        # Embed candidates and compute average pairwise distance
        embeddings = [embed(c) for c in candidates]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
        return 1 - np.mean(similarities)  # Higher = more diverse
```

**Best for:** Open-ended generation, creative tasks. Accuracy gain: 10-30%.

---

## Multi-Agent Patterns

### Multi-Agent Debate

Multiple models critique each other iteratively:

```python
class MultiAgentDebate:
    """
    Pattern: Multiple models debate to reduce hallucinations.
    
    Most effective when:
    1. Models have different biases (diverse model families)
    2. 2-3 rounds is optimal (more = diminishing returns)
    3. Explicit "devil's advocate" prompting improves results
    """
    
    def __init__(self, debaters: list, rounds: int = 2):
        self.debaters = debaters
        self.rounds = rounds
    
    async def debate(self, question: str) -> dict:
        # Round 0: Initial positions
        positions = await asyncio.gather(*[
            debater.generate(f"Answer this question with reasoning: {question}")
            for debater in self.debaters
        ])
        
        debate_history = [{"round": 0, "positions": positions}]
        
        # Debate rounds
        for round_num in range(1, self.rounds + 1):
            new_positions = []
            
            for i, debater in enumerate(self.debaters):
                other_positions = [p for j, p in enumerate(positions) if j != i]
                
                critique_prompt = f"""
Question: {question}

Your previous answer: {positions[i]}

Other perspectives:
{self.format_positions(other_positions)}

Consider the other perspectives. If they raise valid points, update your answer.
If you still disagree, explain why with specific reasoning.
Provide your final answer.
"""
                new_position = await debater.generate(critique_prompt)
                new_positions.append(new_position)
            
            positions = new_positions
            debate_history.append({"round": round_num, "positions": positions})
        
        # Final synthesis
        final_answer = await self.synthesize(question, debate_history)
        
        return {
            "answer": final_answer,
            "rounds": self.rounds,
            "consensus_reached": self.check_consensus(positions),
            "debate_history": debate_history
        }
```

**Best for:** Fact verification, reducing hallucinations in complex answers.

### Mixture of Agents (MoA)

Layered architecture where multiple models feed into aggregators:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIXTURE OF AGENTS (MoA)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1 (Proposers):                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Claude  │  │  GPT-4  │  │ Gemini  │  │ Llama   │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
│       └────────────┴─────┬──────┴────────────┘                  │
│                          │                                       │
│  Layer 2 (Aggregator):   ▼                                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │  "Given these perspectives: [R1, R2, R3, R4]    │           │
│  │   Synthesize the best answer..."                │           │
│  └────────────────────────┬─────────────────────────┘           │
│                           │                                      │
│                           ▼                                      │
│                    [Final Output]                                │
└─────────────────────────────────────────────────────────────────┘
```

```python
class MixtureOfAgents:
    def __init__(self, proposers: list, aggregator):
        self.proposers = proposers
        self.aggregator = aggregator
    
    async def generate(self, prompt: str) -> str:
        # Layer 1: Get diverse proposals
        proposals = await asyncio.gather(*[
            proposer.generate(prompt) for proposer in self.proposers
        ])
        
        # Layer 2: Aggregate
        aggregation_prompt = f"""
Given the following question and multiple expert responses, 
synthesize the best possible answer.

Question: {prompt}

Expert responses:
{self.format_proposals(proposals)}

Synthesize the best answer, combining the strongest elements from each response.
"""
        
        final_answer = await self.aggregator.generate(aggregation_prompt)
        return final_answer
```

**Best for:** Complex synthesis, report generation, multi-domain problems.

---

## Ensemble vs Arbitration

### Conceptual Distinction

| Aspect | Ensemble Learning | Model Arbitration |
|--------|------------------|-------------------|
| **Goal** | Combine ALL outputs | SELECT single best output |
| **Mechanism** | Aggregation (voting, averaging) | Selection (scoring, ranking) |
| **Relationship** | Collaborative | Competitive |
| **Final Output** | Composite from all models | Output of single winner |
| **When to Use** | Want robustness, reduced variance | Want best quality |

### Decision Framework

```
Is there a single "correct" answer format?
├── Yes (classification, math)
│   └── Use Ensemble (voting/averaging)
│
└── No (creative writing, open QA)
    └── Use Arbitration (best-of-N)
        └── Do you have reliable scoring?
            ├── Yes → Reward model selection
            └── No → LLM-as-judge or human
```

---

## Cost-Accuracy Tradeoffs

### Ensemble Cost Matrix

| Method | Cost Multiplier | Latency | Accuracy Gain | When to Use |
|--------|-----------------|---------|---------------|-------------|
| Single Model | 1x | 1x | Baseline | Low-stakes, high-volume |
| Self-Consistency k=3 | 3x | 1x (parallel) | +5-8% | Reasoning, latency-sensitive |
| Self-Consistency k=10 | 10x | 1x (parallel) | +10-15% | Math, accuracy-critical |
| Best-of-N (N=8) | 8x + scoring | 1x (parallel) | +15-25% | Creative generation |
| Panel of Judges (3) | 3x eval | 1x (parallel) | Bias reduction | Evaluation tasks |
| Multi-Agent Debate | 6x | 3x | Hallucination ↓ | Fact-critical |
| Mixture of Agents | 5-8x | 2x | Better synthesis | Complex reports |

### When NOT to Use Ensembles

| Situation | Why Not | Alternative |
|-----------|---------|-------------|
| Simple factual lookup | No diversity benefit | Single RAG call |
| Latency < 500ms required | Ensemble adds latency | Single model + caching |
| Cost is primary constraint | Ensembles multiply cost | Model distillation |
| Models highly correlated | No diversity = no benefit | Get diverse models first |

---

## Interview Questions

### Q: When would you use Self-Consistency vs Best-of-N?

**Strong answer:**

"These serve different purposes:

**Self-Consistency** is for tasks with extractable, verifiable answers:
- Math problems: Extract final number, majority vote
- Classification: Vote on labels
- Short-form QA: Vote on answer

The key is you can compare answers for equality. Temperature 0.5-0.8 provides diversity while maintaining coherence. I use k=5-10 for most tasks.

**Best-of-N** is for open-ended generation where there is no single right answer:
- Creative writing
- Explanations
- Code that could be written many ways

Here I need a reward model or judge to score candidates since I cannot just compare for equality. N=8-16 typically. The challenge is avoiding reward hacking, so I use reward model ensembles with conservative aggregation.

I would not use Self-Consistency for creative writing (no extractable answer) or Best-of-N for math (just use voting, simpler)."

### Q: How do you prevent reward hacking in Best-of-N?

**Strong answer:**

"Reward hacking is when the model exploits weaknesses in the reward model rather than genuinely improving quality.

**My mitigations:**

1. **Reward model ensemble**: Use 3+ diverse reward models. A sample that hacks one RM is unlikely to hack all of them.

2. **Conservative aggregation**: Instead of using the mean score, use the 25th percentile or minimum. This selects samples that score well across all RMs, not just one.

3. **Diversity monitoring**: Track sample diversity. If diversity drops too low, the model may be exploiting a narrow reward hack. I adjust temperature or use different prompts.

4. **Human calibration**: Periodically validate that RM-selected samples actually match human preferences.

5. **Multiple dimensions**: Score on multiple criteria (quality, safety, relevance) and require good scores on all, not just composite.

The key insight is that any single reward signal can be gamed. Ensembles make gaming much harder."

---

## References

- Verga et al. "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models" (2024)
- Wang et al. "Self-Consistency Improves Chain of Thought Reasoning" (2023)
- Du et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023)

---

*Next: [Reliability Patterns Extended](02-reliability-patterns.md)*

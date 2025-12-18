# Tree-of-Thought (ToT)

Tree-of-Thought (ToT) is an advanced prompting architecture where a model explores multiple reasoning paths, evaluates them, and "backtracks" if a path leads to a dead end. In 2025, this is the blueprint for autonomous research agents.

## Table of Contents

- [The Tree vs. The Chain](#tree-vs-chain)
- [The ToT Loop: Propose, Evaluate, Search](#tot-loop)
- [Self-Correction & Backtracking](#self-correction)
- [ToT in 2025: MCTS and Search-as-Service](#mcts)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## The Tree vs. The Chain

While **Chain-of-Thought** is linear (one path), **Tree-of-Thought** allows for branching.

| Feature | Chain-of-Thought | Tree-of-Thought |
|---------|------------------|-----------------|
| **Topology** | Linear (1 path) | Branching (Multiple paths) |
| **Logic** | Sequential | Parallel + Evaluative |
| **Self-Correction**| Low (Commitment bias) | High (Backtracking) |
| **Use Case** | Math, Simple Logic | Puzzle Solving, Coding Architecture, Strategic Planning |

---

## The ToT Loop: Propose, Evaluate, Search

A ToT system consists of three modules:
1. **Thought Proposer**: Generates 3-5 potential "next steps" for a problem.
2. **State Evaluator**: Grades each step (e.g., "Good", "Maybe", "Impossible").
3. **Search Algorithm**: (BFS or DFS) to decide which branch to explore next.

```python
# The ToT logic (Simplified):
For each branch:
   Score = Evaluate(branch)
   If Score < Threshold:
      Prune branch (Backtrack)
   Else:
      Continue exploring
```

---

## Self-Correction & Backtracking

ToT is specifically designed to overcome **Hallucination Cascades**. 
In a linear chain, if the model makes a mistake in Step 1, every subsequent step is likely wrong. In ToT, the "Evaluator" (which can be a different model or a rule-based check) catches the error at Step 1 and forces the model to try a different starting point.

---

## ToT in 2025: MCTS and Search-as-Service

In late 2025, ToT has evolved into **Monte Carlo Tree Search (MCTS)** for LLMs.
- **Search-time Compute Scaling**: Instead of one large prompt, we use 100 small prompts to "search" for the best answer.
- **RAD-T (Reasoning-as-Data-Tree)**: Many teams now use specialized "Searcher" models (like Gemini 3 Ultra) that are natively trained to manage these branches.

---

## Interview Questions

### Q: When is ToT significantly better than simple CoT?

**Strong answer:**
ToT is superior when the problem has a "large search space" and requires "global consistency." For example, in a complex software refactor, a single Chain-of-Thought might start well but hit a constraint conflict 10 steps later. With ToT, the model can propose 3 different refactoring patterns, evaluate the impact of each on the codebase, and discard patterns that lead to circular dependencies before it writes any code.

### Q: What is the main drawback of Tree-of-Thought in a consumer-facing app?

**Strong answer:**
The primary drawback is **Exponential Cost and Latency**. Exploring 3 branches to a depth of 5 can require 15-20 individual LLM calls. In a consumer app, this could result in a 30-second delay and a $0.50 cost for a single query. In 2025, we mitigate this by using a "Hybrid Model": Use ToT for high-stakes offline tasks (like generating golden datasets or security audits) and distill those results into a fast, linear model for real-time interaction.

---

## References
- Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
- Silver et al. "Mastering the Game of Go without Human Knowledge" (MCTS inspiration)

---

*Next: [Context Engineering](05-context-engineering.md)*

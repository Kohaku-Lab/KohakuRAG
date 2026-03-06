# Deduplication and Reranking Features

This document describes the new optional deduplication and reranking features for multi-query retrieval.

## Overview

When using multi-query retrieval (with `planner_max_queries > 1`), the same document nodes can appear in results from different queries. The new features allow you to:

1. **Deduplicate** results by node_id
2. **Rerank** results using frequency and total score metrics
3. **Tree dedup** — remove child nodes whose ancestor is also retrieved
4. **Truncate** final results to a specified limit

## Configuration Parameters

### In Config Files

Add these parameters to your config (e.g., `configs/text_only/answer.py`):

```python
# Deduplication
deduplicate_retrieval = True  # Remove duplicate nodes by node_id

# Reranking strategy
rerank_strategy = "frequency"  # Options: None, "frequency", "score", "combined"

# Tree dedup (optional) — remove children when their ancestor is also retrieved
tree_dedup = True  # Useful when retrieval targets multiple node levels

# Final truncation (optional)
top_k_final = 20  # Truncate to top-20 after dedup+rerank (None = no truncation)
```

### In RAGPipeline

```python
from kohakurag import RAGPipeline

pipeline = RAGPipeline(
    store=store,
    embedder=embedder,
    planner=planner,
    top_k=16,  # Results per query
    deduplicate_retrieval=True,
    rerank_strategy="frequency",
    tree_dedup=True,  # Remove children when ancestor is also retrieved
    top_k_final=20
)
```

## How It Works

### Without Deduplication/Reranking

**Configuration:**
```python
top_k = 16
planner_max_queries = 3
deduplicate_retrieval = False
rerank_strategy = None
top_k_final = None
```

**Result:**
- 3 queries × 16 results = **48 documents**
- Duplicates possible (same node can appear multiple times)
- Order: concatenated by query order

### With Deduplication Only

**Configuration:**
```python
deduplicate_retrieval = True
rerank_strategy = None
top_k_final = None
```

**Result:**
- Up to **48 unique documents** (duplicates removed)
- Order: first occurrence preserved

### With Deduplication and Reranking

**Configuration:**
```python
deduplicate_retrieval = True
rerank_strategy = "frequency"  # or "score" or "combined"
top_k_final = None
```

**Result:**
- **Unique documents** reranked by strategy
- Order: best-ranked first

### With Deduplication, Reranking, and Truncation

**Configuration:**
```python
deduplicate_retrieval = True
rerank_strategy = "frequency"
top_k_final = 20
```

**Result:**
- **20 documents** (best-ranked after dedup)
- This is the recommended configuration for production use

## Reranking Strategies

### 1. Frequency Strategy (`"frequency"`)

Ranks documents by:
1. **Primary:** How many queries returned this document (frequency)
2. **Secondary:** Sum of similarity scores across all queries (total_score)

**Use case:** Prioritize documents that are relevant to multiple query variations.

**Example:**
- Document A: appeared in 3 queries, total_score = 2.4
- Document B: appeared in 2 queries, total_score = 2.8
- **Ranking:** A > B (higher frequency wins)

### 2. Score Strategy (`"score"`)

Ranks documents by total similarity score (sum of scores across queries).

**Use case:** Prioritize documents with highest cumulative relevance.

**Example:**
- Document A: total_score = 2.4
- Document B: total_score = 2.8
- **Ranking:** B > A (higher total score wins)

### 3. Combined Strategy (`"combined"`)

Ranks by weighted combination:
```
combined_score = 0.4 × (frequency / max_frequency) + 0.6 × (total_score / max_total_score)
```

**Use case:** Balance between frequency and score.

## Tree Dedup (Parent Dedup)

When retrieval targets multiple node levels (e.g., both SENTENCE and PARAGRAPH), different queries may return a parent and its child:

```
Query 1 → "doc:sec1:p2"       (paragraph)
Query 2 → "doc:sec1:p2:s3"    (sentence within that paragraph)
```

Standard dedup only removes exact `node_id` duplicates, so both survive. With `tree_dedup=True`, the child is removed because its ancestor is already present — the parent's text already contains the child's text.

**How it works:**
- After dedup+rerank, collect all `node_id`s in the result set
- For each node, check if any other node's ID is a prefix (ancestor)
- Remove nodes that have an ancestor present (keep the broader parent)

**Example:**
```
Before tree dedup: [doc:sec1:p2, doc:sec1:p2:s3, doc:sec2:p1, doc:sec2:p1:s1]
After tree dedup:  [doc:sec1:p2, doc:sec2:p1]
```

**When to use:** Enable when your retrieval searches multiple node levels (SENTENCE + PARAGRAPH) and you want to avoid wasting context window on redundant content.

## Recommended Configurations

### For Maximum Recall (Research, Exploration)

```python
top_k = 16
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = None
top_k_final = None
```

**Result:** Up to 48 unique documents, maintaining diversity.

### For Balanced Performance (Production QA)

```python
top_k = 16
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = "frequency"
top_k_final = 20
```

**Result:** Top-20 documents ranked by multi-query relevance.

### For Precision (Focused Answers)

```python
top_k = 8
planner_max_queries = 3
deduplicate_retrieval = True
rerank_strategy = "combined"
top_k_final = 10
```

**Result:** Top-10 best documents with balanced ranking.

### For Speed (Low Latency)

```python
top_k = 5
planner_max_queries = 1  # Disable multi-query
deduplicate_retrieval = False
rerank_strategy = None
top_k_final = None
```

**Result:** 5 documents, single query (fastest).

## Technical Details

### Frequency Calculation

When a document appears in multiple query results:
- **Frequency:** Count of how many queries returned it
- **Total Score:** Sum of similarity scores across all queries
- **Max Score:** Highest similarity score among all queries

### Deduplication Logic

1. Documents with the same `node_id` are considered duplicates
2. When deduplicating without reranking: keeps first occurrence
3. When reranking: aggregates all occurrences, uses frequency + total_score

### Processing Order

```
1. Query Planning (planner generates N queries)
      ↓
2. Embedding (embed all queries)
      ↓
3. Retrieval (search top_k for each query)
      ↓
4. Deduplication (if enabled, remove duplicates)
      ↓
5. Reranking (if enabled, sort by strategy)
      ↓
6. Tree Dedup (if enabled, remove children with ancestor present)
      ↓
7. Truncation (if top_k_final set, keep top N)
      ↓
8. Context Expansion (add parent/child nodes)
      ↓
9. Return RetrievalResult
```

## Example: Full Configuration

**Config file:** `configs/production/answer.py`

```python
"""Production RAG configuration with dedup and reranking."""

db = "artifacts/production.db"
table_prefix = "prod"
questions = "data/questions.csv"
output = "artifacts/answers.csv"

# LLM settings
model = "openai/gpt-4o"
planner_model = "openai/gpt-4o-mini"

# Retrieval settings
top_k = 16                      # Documents per query
planner_max_queries = 3         # Generate 3 query variations

# Dedup and rerank settings
deduplicate_retrieval = True    # Remove duplicates
rerank_strategy = "frequency"   # Rank by multi-query relevance
top_k_final = 20               # Final truncation to 20 docs

# Metadata
metadata = "data/metadata.csv"
max_retries = 3
max_concurrent = 10
```

## Performance Impact

| Configuration | Docs Retrieved | Vector Searches | Context Expansion |
|--------------|----------------|-----------------|-------------------|
| Basic (no dedup) | 3 × 16 = 48 | 3 | 48 |
| With dedup | ≤48 unique | 3 | ≤48 |
| With rerank | ≤48 unique | 3 | ≤48 |
| With top_k_final=20 | 20 | 3 | 20 |

**Recommendation:** Use `top_k_final` to limit context expansion overhead.

## Testing

Run the test script to verify the implementation:

```bash
python test_dedup_rerank.py
```

This will test all configuration combinations and show the number of documents retrieved.

## Migration Guide

### Updating Existing Configs

If you have existing configs, add these lines:

```python
# Add after planner_max_queries
deduplicate_retrieval = False  # Keep original behavior
rerank_strategy = None         # Keep original behavior
tree_dedup = False             # Keep original behavior
top_k_final = None            # Keep original behavior
```

### Updating Existing Scripts

If you create `RAGPipeline` directly in code:

```python
# Old code (still works)
pipeline = RAGPipeline(
    store=store,
    embedder=embedder,
    planner=planner,
    top_k=16
)

# New code with optional features
pipeline = RAGPipeline(
    store=store,
    embedder=embedder,
    planner=planner,
    top_k=16,
    deduplicate_retrieval=True,   # Optional
    rerank_strategy="frequency",  # Optional
    tree_dedup=True,              # Optional
    top_k_final=20               # Optional
)
```

## Related: Ensemble Aggregation

While dedup/rerank operates at the **retrieval** level (combining results from multi-query), there's also **ensemble aggregation** at the **inference** level (combining results from multiple runs). See `scripts/wattbot_aggregate.py` and `docs/usage.md` for:

- **Aggregation modes**: `independent`, `ref_priority`, `answer_priority`, `union`, `intersection`
- **ignore_blank**: Filter out `is_blank` answers before voting (if non-blank exist)

These are complementary features:
- **Dedup/rerank**: Improves retrieval quality within a single run
- **Ensemble aggregation**: Improves robustness across multiple runs

## FAQ

**Q: Should I always enable deduplication?**

A: Generally yes, unless you specifically want to see how many queries returned each document (e.g., for debugging).

**Q: Which reranking strategy is best?**

A: `"frequency"` is recommended for most cases as it prioritizes documents relevant to multiple query variations. Use `"score"` if you trust similarity scores more than multi-query consensus.

**Q: What's the difference between `top_k` and `top_k_final`?**

A: `top_k` controls how many results each query returns. `top_k_final` truncates the final deduplicated+reranked results. Example: `top_k=16, max_queries=3, top_k_final=20` retrieves 48 docs, then keeps top-20.

**Q: What's the difference between `tree_dedup` and `no_overlap`?**

A: Both remove parent-child redundancy, but at different pipeline stages:
- `tree_dedup` operates on **retrieval matches** (before context expansion and truncation). It reduces the match count, freeing slots for other unique content.
- `no_overlap` operates on **context snippets** (after context expansion). It removes overlap introduced by the expansion step itself.

They are complementary — `tree_dedup` handles overlap from multi-level retrieval, while `no_overlap` handles overlap from context expansion.

**Q: Does reranking automatically deduplicate?**

A: Yes! Reranking inherently requires aggregating duplicate nodes to calculate frequency and total_score. However, setting `deduplicate_retrieval=True` is still recommended for clarity.

**Q: Can I use reranking without deduplication?**

A: Yes, but the results will be automatically deduplicated as part of reranking (since we need to aggregate scores). Set `deduplicate_retrieval=False` if you want to preserve query order before reranking.

**Q: What's the difference between dedup/rerank and ignore_blank?**

A: Different levels of the pipeline:
- **Dedup/rerank**: Operates on retrieval results (document nodes)
- **ignore_blank**: Operates on aggregation (final answer values)

Use dedup/rerank for better retrieval within a run; use ignore_blank for robust ensemble voting across runs.

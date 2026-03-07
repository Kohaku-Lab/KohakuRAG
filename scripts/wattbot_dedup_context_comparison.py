"""Compare context length under different dedup settings (retrieval-only).

Runs retrieval for each training question under three dedup modes:
  1. No dedup      – raw concatenation of multi-query results
  2. Node-ID dedup – deduplicate by node_id (standard)
  3. Tree dedup    – also remove children when ancestor is present

Reports match counts and total context character/token lengths per question,
plus aggregate statistics. No LLM calls are made.

Usage:
    .venv/Scripts/python scripts/wattbot_dedup_context_comparison.py
"""

import asyncio
import csv
import os
import statistics
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kohakurag import NodeKind
from kohakurag.datastore import KVaultNodeStore, matches_to_snippets
from kohakurag.embeddings import JinaV4EmbeddingModel
from kohakurag.types import RetrievalMatch

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
QUESTIONS_PATH = "data/train_QA.csv"

# Retrieval settings (match competition defaults)
TOP_K = 16
PLANNER_MAX_QUERIES = 4
RERANK_STRATEGY = "combined"
TOP_K_FINAL = 32

# Embedding
EMBEDDING_DIM = 512
EMBEDDING_TASK = "retrieval"


# ============================================================================
# SIMPLE QUERY PLANNER (no LLM needed)
# ============================================================================


class DiverseQueryPlanner:
    """Generate exactly *max_queries* diverse query variants without LLM.

    Strategies used (in order until we have enough):
      1. Original question (punctuation stripped)
      2. Question-prefix removal  ("What is the X?" → "X")
      3. Stop-word removal        (keywords only)
      4. Reversed keyword order
      5. Rotated keywords         (second-half + first-half)
      6. Keyword sub-spans
    """

    _STOP = frozenset(
        "the a an is are was were of in to for on at by with from and or "
        "that this it its be has have had do does did what which how who "
        "when where much many can could would should will shall".split()
    )
    _Q_PREFIXES = [
        "What is ",
        "What are ",
        "What was ",
        "What were ",
        "How much ",
        "How many ",
        "Which ",
        "When did ",
        "When ",
        "Where ",
        "Who ",
        "How does ",
        "How do ",
        "How did ",
    ]

    def __init__(self, max_queries: int = 4):
        self._n = max_queries

    async def plan(self, question: str) -> Sequence[str]:
        q = question.strip().rstrip("?.,!").strip()
        seen: set[str] = set()
        out: list[str] = []

        def _add(t: str) -> None:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

        # 1. Original (cleaned)
        _add(q)

        # 2. Remove question prefix
        ql = q.lower()
        for pfx in self._Q_PREFIXES:
            if ql.startswith(pfx.lower()):
                _add(q[len(pfx) :].strip())
                break

        # Keywords (stop-word removal)
        kw = [w for w in q.split() if w.lower() not in self._STOP and len(w) > 1]

        # 3. Keywords only
        if kw:
            _add(" ".join(kw))

        # 4. Reversed keywords
        if len(kw) > 1:
            _add(" ".join(reversed(kw)))

        # 5. Rotated keywords (second half first)
        if len(out) < self._n and len(kw) > 2:
            mid = len(kw) // 2
            _add(" ".join(kw[mid:] + kw[:mid]))

        # 6. Keyword sub-spans (first half, second half)
        if len(out) < self._n and len(kw) > 2:
            _add(" ".join(kw[: len(kw) // 2 + 1]))
        if len(out) < self._n and len(kw) > 2:
            _add(" ".join(kw[len(kw) // 2 :]))

        # Pad: if still short, duplicate original (ensures 4 retrievals)
        while len(out) < self._n:
            out.append(out[0])

        return out[: self._n]


# ============================================================================
# CORE LOGIC
# ============================================================================


@dataclass
class DedupStats:
    """Stats for one question under one dedup setting."""

    num_matches: int
    num_snippets: int
    total_chars: int


async def run_retrieval_raw(
    embedder: JinaV4EmbeddingModel,
    store: KVaultNodeStore,
    planner: DiverseQueryPlanner,
    question: str,
) -> list[RetrievalMatch]:
    """Run multi-query retrieval and return raw (un-deduped) matches."""
    queries = list(await planner.plan(question))
    query_vectors = await embedder.embed(queries)

    all_matches: list[RetrievalMatch] = []
    for vector in query_vectors:
        matches = await store.search(
            vector,
            k=TOP_K,
            kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH},
        )
        all_matches.extend(matches)
    return all_matches


def dedup_by_node_id(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Standard node-ID deduplication."""
    seen: set[str] = set()
    result = []
    for m in matches:
        if m.node.node_id not in seen:
            seen.add(m.node.node_id)
            result.append(m)
    return result


def rerank_combined(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Combined reranking (frequency + score, simplified)."""
    from collections import defaultdict

    freq: dict[str, int] = defaultdict(int)
    total_score: dict[str, float] = defaultdict(float)
    first_match: dict[str, RetrievalMatch] = {}

    for m in matches:
        nid = m.node.node_id
        freq[nid] += 1
        total_score[nid] += m.score
        if nid not in first_match:
            first_match[nid] = m

    unique = list(first_match.values())

    if not unique:
        return unique

    max_f = max(freq.values())
    min_f = min(freq.values())
    max_s = max(total_score.values())
    min_s = min(total_score.values())

    range_f = max_f - min_f if max_f != min_f else 1.0
    range_s = max_s - min_s if max_s != min_s else 1.0

    unique.sort(
        key=lambda m: (
            0.5 * (freq[m.node.node_id] - min_f) / range_f
            + 0.5 * (total_score[m.node.node_id] - min_s) / range_s
        ),
        reverse=True,
    )
    return unique


def tree_dedup(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Remove matches whose ancestor is also present."""
    if not matches:
        return matches

    all_ids = {m.node.node_id for m in matches}
    ids_with_ancestor: set[str] = set()

    for node_id in all_ids:
        for other_id in all_ids:
            if other_id != node_id and node_id.startswith(other_id + ":"):
                ids_with_ancestor.add(node_id)
                break

    if not ids_with_ancestor:
        return matches

    return [m for m in matches if m.node.node_id not in ids_with_ancestor]


async def compute_stats(
    matches: list[RetrievalMatch],
    store: KVaultNodeStore,
) -> DedupStats:
    """Expand matches to snippets and compute stats."""
    snippets = await matches_to_snippets(
        matches, store, parent_depth=1, child_depth=0, no_overlap=False
    )
    total_chars = sum(len(s.text) for s in snippets)
    return DedupStats(
        num_matches=len(matches),
        num_snippets=len(snippets),
        total_chars=total_chars,
    )


async def main() -> None:
    print("=" * 75)
    print("Context Length Comparison: No Dedup vs Node-ID Dedup vs Tree Dedup")
    print("=" * 75)

    # Load embedder
    print(f"\nLoading Jina v4 embedder (dim={EMBEDDING_DIM})...")
    embedder = JinaV4EmbeddingModel(truncate_dim=EMBEDDING_DIM, task=EMBEDDING_TASK)

    # Load store
    print(f"Opening database: {DB_PATH}")
    store = KVaultNodeStore(Path(DB_PATH), table_prefix=TABLE_PREFIX, dimensions=None)

    # Load questions
    print(f"Loading questions: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        questions = [(row["id"], row["question"]) for row in reader]
    print(f"Loaded {len(questions)} questions\n")

    planner = DiverseQueryPlanner(max_queries=PLANNER_MAX_QUERIES)

    # Per-question results
    results_no_dedup: list[DedupStats] = []
    results_node_dedup: list[DedupStats] = []
    results_tree_dedup: list[DedupStats] = []

    header = f"{'ID':<6} {'Question':<55} {'Mode':<12} {'Matches':>8} {'Snippets':>9} {'Chars':>8}"
    print(header)
    print("-" * len(header))

    for qid, question in questions:
        q_short = question[:52] + "..." if len(question) > 55 else question

        # 1. Raw retrieval (shared across all modes)
        raw_matches = await run_retrieval_raw(embedder, store, planner, question)

        # Mode 1: No dedup – sort by score, keep duplicates
        m1 = sorted(list(raw_matches), key=lambda m: m.score, reverse=True)
        if TOP_K_FINAL:
            m1 = m1[:TOP_K_FINAL]
        s1 = await compute_stats(m1, store)
        results_no_dedup.append(s1)

        # Mode 2: Node-ID dedup + sort by score + truncate
        m2 = dedup_by_node_id(list(raw_matches))
        m2 = sorted(m2, key=lambda m: m.score, reverse=True)
        if TOP_K_FINAL:
            m2 = m2[:TOP_K_FINAL]
        s2 = await compute_stats(m2, store)
        results_node_dedup.append(s2)

        # Mode 3: Node-ID dedup + sort by score + tree dedup + truncate
        m3 = dedup_by_node_id(list(raw_matches))
        m3 = sorted(m3, key=lambda m: m.score, reverse=True)
        m3 = tree_dedup(m3)
        if TOP_K_FINAL:
            m3 = m3[:TOP_K_FINAL]
        s3 = await compute_stats(m3, store)
        results_tree_dedup.append(s3)

        # Print per-question results
        print(
            f"{qid:<6} {q_short:<55} {'no_dedup':<12} {s1.num_matches:>8} {s1.num_snippets:>9} {s1.total_chars:>8}"
        )
        print(
            f"{'':6} {'':55} {'node_id':<12} {s2.num_matches:>8} {s2.num_snippets:>9} {s2.total_chars:>8}"
        )
        print(
            f"{'':6} {'':55} {'tree':<12} {s3.num_matches:>8} {s3.num_snippets:>9} {s3.total_chars:>8}"
        )
        print()

    # Aggregate statistics
    def summarize(label: str, stats_list: list[DedupStats]) -> None:
        matches = [s.num_matches for s in stats_list]
        snippets = [s.num_snippets for s in stats_list]
        chars = [s.total_chars for s in stats_list]
        print(
            f"  {label:<14} | "
            f"matches: {statistics.mean(matches):6.1f} +/- {statistics.stdev(matches):5.1f} | "
            f"snippets: {statistics.mean(snippets):6.1f} +/- {statistics.stdev(snippets):5.1f} | "
            f"chars: {statistics.mean(chars):8.0f} +/- {statistics.stdev(chars):6.0f}"
        )

    print("\n" + "=" * 75)
    print("AGGREGATE STATISTICS (mean +/- std)")
    print("=" * 75)
    summarize("No dedup", results_no_dedup)
    summarize("Node-ID dedup", results_node_dedup)
    summarize("Tree dedup", results_tree_dedup)

    # Reduction percentages
    avg_no = statistics.mean([s.total_chars for s in results_no_dedup])
    avg_node = statistics.mean([s.total_chars for s in results_node_dedup])
    avg_tree = statistics.mean([s.total_chars for s in results_tree_dedup])

    print(f"\n  Context reduction (chars):")
    print(
        f"    Node-ID dedup vs No dedup:   {(1 - avg_node / avg_no) * 100:5.1f}% reduction"
    )
    print(
        f"    Tree dedup vs No dedup:      {(1 - avg_tree / avg_no) * 100:5.1f}% reduction"
    )
    print(
        f"    Tree dedup vs Node-ID dedup: {(1 - avg_tree / avg_node) * 100:5.1f}% reduction"
    )

    # Match count reduction
    avg_m_no = statistics.mean([s.num_matches for s in results_no_dedup])
    avg_m_node = statistics.mean([s.num_matches for s in results_node_dedup])
    avg_m_tree = statistics.mean([s.num_matches for s in results_tree_dedup])

    print(f"\n  Match reduction:")
    print(
        f"    Node-ID dedup vs No dedup:   {(1 - avg_m_node / avg_m_no) * 100:5.1f}% reduction"
    )
    print(
        f"    Tree dedup vs No dedup:      {(1 - avg_m_tree / avg_m_no) * 100:5.1f}% reduction"
    )
    print(
        f"    Tree dedup vs Node-ID dedup: {(1 - avg_m_tree / avg_m_node) * 100:5.1f}% reduction"
    )


if __name__ == "__main__":
    asyncio.run(main())

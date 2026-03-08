"""Dedup Evaluation Matrix: 2x3 grid of expansion x dedup modes.

Runs a 2x3 matrix of configurations:
  Rows: parent_depth=1/child_depth=1, parent_depth=1/child_depth=0
  Cols: no-dedup, node-id dedup, tree dedup

Each cell runs N independent inferences (ensemble), aggregates with
answer_priority voting + ignore_blank, then validates against train_QA.csv.

Reports: context length (from retrieval stats) and accuracy (value, ref, final).

Usage:
    kogine run workflows/sweeps/dedup_eval_matrix.py
    .venv/Scripts/python workflows/sweeps/dedup_eval_matrix.py
"""

import ast
import csv
import json
from pathlib import Path

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# CONFIGURATION (all configurable via KohakuEngine config injection)
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/dedup_matrix")
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# LLM — uses OpenAI client; set OPENAI_API_KEY and OPENAI_BASE_URL env vars
# to route through OpenRouter (e.g. OPENAI_BASE_URL=https://openrouter.ai/api/v1)
MODEL = "openai/gpt-oss-120b"
ENSEMBLE_RUNS = 5
MAX_CONCURRENT = 64

# What to run
RUN_CONTEXT_MEASUREMENT = False
RUN_LLM_EVAL = True

# Matrix dimensions
EXPANSION_MODES = [
    {"parent_depth": 1, "child_depth": 1, "label": "p1c1"},
    {"parent_depth": 1, "child_depth": 0, "label": "p1c0"},
]

DEDUP_MODES = [
    {"snippet_dedup": "none", "label": "no_dedup"},
    {"snippet_dedup": "node_id", "label": "node_id"},
    {"snippet_dedup": "tree", "label": "tree"},
]

# ============================================================================
# BASE CONFIG (module-level capture_globals)
# ============================================================================

with capture_globals() as ctx:
    db = DB
    table_prefix = TABLE_PREFIX
    questions = QUESTIONS
    metadata = METADATA

    # LLM settings — use "openai" provider
    llm_provider = "openai"
    model = MODEL
    planner_model = None

    # Retrieval settings
    top_k = 16
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = 32

    # Context expansion & dedup (defaults, overridden per cell)
    parent_depth = 1
    child_depth = 0
    snippet_dedup = "none"

    # JinaV4 embedding settings
    embedding_model = "jinav4"
    embedding_dim = 512
    embedding_task = "retrieval"

    # Paragraph search mode
    paragraph_search_mode = "averaged"

    # Other
    with_images = True
    top_k_images = 4
    send_images_to_llm = False
    use_reordered_prompt = True
    max_retries = 3
    max_concurrent = MAX_CONCURRENT
    single_run_debug = False
    question_id = None
    bm25_top_k = 0


# ============================================================================
# CONFIG FACTORY
# ============================================================================


def create_answer_config(expansion: dict, dedup: dict, output_path: str) -> Config:
    """Create config for one matrix cell inference run."""
    config = Config.from_context(ctx)
    config.globals_dict["parent_depth"] = expansion["parent_depth"]
    config.globals_dict["child_depth"] = expansion["child_depth"]
    config.globals_dict["snippet_dedup"] = dedup["snippet_dedup"]
    config.globals_dict["output"] = output_path
    return config


# ============================================================================
# INFERENCE
# ============================================================================


def run_inference(cell_dir: Path, expansion: dict, dedup: dict, num_runs: int):
    """Run N independent inferences for one matrix cell."""
    raw_dir = cell_dir / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(num_runs):
        pred_path = raw_dir / f"run{run_idx}_preds.csv"

        if pred_path.exists():
            # Check if it has more than just a header
            with open(pred_path, encoding="utf-8-sig") as f:
                lines = sum(1 for _ in f)
            if lines > 1:
                print(f"    [run {run_idx}] Skipping (already exists, {lines-1} rows)")
                continue
            # Header-only file = incomplete run, redo it
            pred_path.unlink()

        print(f"    [run {run_idx}] Running inference...")
        config = create_answer_config(expansion, dedup, str(pred_path))
        answer_script = Script("scripts/wattbot_answer.py", config=config)
        answer_script.run()


# ============================================================================
# AGGREGATION & VALIDATION
# ============================================================================


def aggregate_runs(cell_dir: Path, num_runs: int) -> Path:
    """Aggregate N runs using answer_priority voting with ignore_blank."""
    raw_dir = cell_dir / "raw_runs"
    agg_path = cell_dir / "aggregated.csv"

    if agg_path.exists():
        print(f"    Aggregated file already exists, skipping")
        return agg_path

    # Collect run files
    run_files = [str(raw_dir / f"run{i}_preds.csv") for i in range(num_runs)]
    existing = [f for f in run_files if Path(f).exists()]

    if not existing:
        print(f"    WARNING: No run files found!")
        return agg_path

    config = Config(
        globals_dict={
            "inputs": existing,
            "output": str(agg_path),
            "ref_mode": "answer_priority",
            "tiebreak": "first",
            "ignore_blank": True,
        }
    )
    agg_script = Script("scripts/wattbot_aggregate.py", config=config)
    agg_script.run()

    return agg_path


# ============================================================================
# INLINE VALIDATION (no imports from other scripts)
# ============================================================================


def _load_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load CSV and index by question ID."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader if row.get("id")}


def _is_blank(value: str | None) -> bool:
    if value is None:
        return True
    s = value.strip()
    return not s or s.lower() == "is_blank"


def _parse_numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_range(value: str) -> tuple[float, float] | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(isinstance(x, (int, float)) for x in parsed)
    ):
        return (float(parsed[0]), float(parsed[1]))
    return None


def _match_numeric(target: float, candidate: float) -> float:
    tolerance = max(abs(target) * 0.001, 1e-9)
    return 1.0 if abs(target - candidate) <= tolerance else 0.0


def _score_answer_value(true_val: str, pred_val: str) -> float:
    if _is_blank(true_val):
        return 1.0 if _is_blank(pred_val) else 0.0
    if _is_blank(pred_val):
        return 0.0

    # Range matching
    tr = _parse_range(true_val)
    pr = _parse_range(pred_val)
    if tr is not None:
        if pr is None:
            return 0.0
        return _match_numeric(tr[0], pr[0]) * _match_numeric(tr[1], pr[1])

    # Numeric matching (0.1% tolerance)
    tn = _parse_numeric(true_val)
    pn = _parse_numeric(pred_val)
    if tn is not None and pn is not None:
        return _match_numeric(tn, pn)
    if tn is not None or pn is not None:
        return 0.0  # Type mismatch

    # Categorical (case-insensitive)
    t_norm = " ".join(true_val.strip().lower().split())
    p_norm = " ".join(pred_val.strip().lower().split())
    return 1.0 if t_norm == p_norm else 0.0


def _parse_ref_ids(value: str) -> set[str]:
    if _is_blank(value):
        return set()
    try:
        data = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        data = value
    if isinstance(data, (list, tuple, set)):
        tokens = (str(x) for x in data)
    else:
        tokens = (t.strip() for t in str(data).strip().strip("[]").split(","))
    return {t.lower() for t in tokens if t}


def _score_ref_ids(true_val: str, pred_val: str) -> float:
    truth = _parse_ref_ids(true_val)
    pred = _parse_ref_ids(pred_val)
    if not truth and not pred:
        return 1.0
    union = truth | pred
    if not union:
        return 0.0
    return len(truth & pred) / len(union)


def _score_na(true_row: dict, pred_row: dict) -> float:
    if not _is_blank(true_row.get("answer_value")):
        return 1.0  # Not an NA question
    return (
        1.0
        if all(_is_blank(pred_row.get(f)) for f in ("answer_value", "ref_id"))
        else 0.0
    )


def validate(pred_path: Path) -> dict:
    """Validate predictions and return scores."""
    zero = {"final_score": 0.0, "value_score": 0.0, "ref_score": 0.0, "na_score": 0.0}
    if not pred_path.exists():
        return zero

    truth_rows = _load_csv_rows(Path(QUESTIONS))
    pred_rows = _load_csv_rows(pred_path)

    if not truth_rows:
        return zero

    value_scores = []
    ref_scores = []
    na_scores = []

    for qid, t_row in truth_rows.items():
        p_row = pred_rows.get(qid, {})
        value_scores.append(
            _score_answer_value(
                t_row.get("answer_value", ""), p_row.get("answer_value", "")
            )
        )
        ref_scores.append(
            _score_ref_ids(t_row.get("ref_id", ""), p_row.get("ref_id", ""))
        )
        na_scores.append(_score_na(t_row, p_row) if p_row else 0.0)

    n = len(value_scores)
    avg_val = sum(value_scores) / n
    avg_ref = sum(ref_scores) / n
    avg_na = sum(na_scores) / n
    final = 0.75 * avg_val + 0.15 * avg_ref + 0.10 * avg_na

    return {
        "final_score": final,
        "value_score": avg_val,
        "ref_score": avg_ref,
        "na_score": avg_na,
    }


# ============================================================================
# CONTEXT LENGTH MEASUREMENT (no LLM calls)
# ============================================================================


def _rerank_combined_local(matches):
    """Combined reranking (frequency + score)."""
    from collections import defaultdict

    freq = defaultdict(int)
    total_score = defaultdict(float)
    first_match = {}

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


class _DiverseQueryPlanner:
    """Generate diverse query variants without LLM."""

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

    def __init__(self, max_queries=4):
        self._n = max_queries

    async def plan(self, question):
        q = question.strip().rstrip("?.,!").strip()
        seen, out = set(), []

        def _add(t):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

        _add(q)
        ql = q.lower()
        for pfx in self._Q_PREFIXES:
            if ql.startswith(pfx.lower()):
                _add(q[len(pfx) :].strip())
                break
        kw = [w for w in q.split() if w.lower() not in self._STOP and len(w) > 1]
        if kw:
            _add(" ".join(kw))
        if len(kw) > 1:
            _add(" ".join(reversed(kw)))
        if len(out) < self._n and len(kw) > 2:
            mid = len(kw) // 2
            _add(" ".join(kw[mid:] + kw[:mid]))
        while len(out) < self._n:
            out.append(out[0])
        return out[: self._n]


def measure_context_length(expansion: dict, dedup: dict) -> dict:
    """Measure average context length for one config (no LLM calls)."""
    import asyncio
    from kohakurag import NodeKind
    from kohakurag.datastore import KVaultNodeStore, matches_to_snippets
    from kohakurag.embeddings import JinaV4EmbeddingModel

    async def _measure():
        embedder = JinaV4EmbeddingModel(truncate_dim=512, task="retrieval")
        store = KVaultNodeStore(Path(DB), table_prefix=TABLE_PREFIX, dimensions=None)

        with open(QUESTIONS, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            questions = [(row["id"], row["question"]) for row in reader]

        planner = _DiverseQueryPlanner(max_queries=4)

        total_chars = []
        total_snippets = []

        for qid, question in questions:
            queries = list(await planner.plan(question))
            query_vectors = await embedder.embed(queries)

            all_matches = []
            for vector in query_vectors:
                matches = await store.search(
                    vector, k=16, kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH}
                )
                all_matches.extend(matches)

            # Rerank + truncate (like pipeline)
            reranked = _rerank_combined_local(all_matches)
            reranked = reranked[:32]

            # Expand to snippets with dedup
            snippets = await matches_to_snippets(
                reranked,
                store,
                parent_depth=expansion["parent_depth"],
                child_depth=expansion["child_depth"],
                dedup=dedup["snippet_dedup"],
            )

            chars = sum(len(s.text) for s in snippets)
            total_chars.append(chars)
            total_snippets.append(len(snippets))

        import statistics

        return {
            "avg_chars": statistics.mean(total_chars),
            "std_chars": statistics.stdev(total_chars) if len(total_chars) > 1 else 0,
            "avg_snippets": statistics.mean(total_snippets),
            "std_snippets": (
                statistics.stdev(total_snippets) if len(total_snippets) > 1 else 0
            ),
        }

    return asyncio.run(_measure())


# ============================================================================
# REPORTING
# ============================================================================


def _print_results(results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("RESULTS MATRIX")
    print("=" * 80)

    # Header
    print(
        f"\n{'Config':<20} {'Snippets':>10} {'Chars':>12} "
        f"{'Final':>8} {'Value':>8} {'Ref':>8}"
    )
    print("-" * 75)

    for key, data in sorted(results.items()):
        ctx_data = data.get("context", {})
        scores = data.get("scores", {})
        print(
            f"{key:<20} "
            f"{ctx_data.get('avg_snippets', 0):>10.1f} "
            f"{ctx_data.get('avg_chars', 0):>12.0f} "
            f"{scores.get('final_score', 0):>8.3f} "
            f"{scores.get('value_score', 0):>8.3f} "
            f"{scores.get('ref_score', 0):>8.3f}"
        )


def _save_results(results: dict):
    """Save results to JSON."""
    out_path = OUTPUT_DIR / "matrix_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 80)
    print("Dedup Evaluation Matrix")
    print(f"  Expansion modes: {[e['label'] for e in EXPANSION_MODES]}")
    print(f"  Dedup modes: {[d['label'] for d in DEDUP_MODES]}")
    print(f"  Ensemble size: {ENSEMBLE_RUNS}")
    print(f"  Model: {MODEL}")
    print(f"  Max concurrent: {MAX_CONCURRENT}")
    print("=" * 80)

    # Phase 1: Measure context lengths
    if RUN_CONTEXT_MEASUREMENT:
        print("\n--- Phase 1: Context Length Measurement ---\n")
        for exp in EXPANSION_MODES:
            for ded in DEDUP_MODES:
                key = f"{exp['label']}_{ded['label']}"
                print(f"  Measuring context: {key}...")
                ctx_stats = measure_context_length(exp, ded)
                results[key] = {"context": ctx_stats}
                print(
                    f"    avg_chars={ctx_stats['avg_chars']:.0f} "
                    f"avg_snippets={ctx_stats['avg_snippets']:.1f}"
                )

    if not RUN_LLM_EVAL:
        _print_results(results)
        _save_results(results)
        exit()

    # Phase 2: Run inferences
    print("\n--- Phase 2: Running Inferences ---\n")
    for exp in EXPANSION_MODES:
        for ded in DEDUP_MODES:
            key = f"{exp['label']}_{ded['label']}"
            cell_dir = OUTPUT_DIR / key
            print(f"\n  Config: {key}")

            # Run N inferences
            run_inference(cell_dir, exp, ded, ENSEMBLE_RUNS)

            # Aggregate
            print(f"    Aggregating {ENSEMBLE_RUNS} runs...")
            agg_path = aggregate_runs(cell_dir, ENSEMBLE_RUNS)

            # Validate
            print(f"    Validating...")
            scores = validate(agg_path)

            if key not in results:
                results[key] = {}
            results[key]["scores"] = scores
            print(
                f"    final={scores['final_score']:.3f} "
                f"value={scores['value_score']:.3f} "
                f"ref={scores['ref_score']:.3f}"
            )

    _print_results(results)
    _save_results(results)

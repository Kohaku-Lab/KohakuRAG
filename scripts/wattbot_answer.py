"""Answer WattBot questions using the KohakuRAG pipeline + OpenAI.

This script demonstrates end-to-end RAG usage:
- Loads questions from CSV
- Retrieves relevant context from the index
- Generates structured answers via OpenAI
- Handles rate limits automatically
- Supports concurrent processing with asyncio.gather()

Usage (CLI):
    python scripts/wattbot_answer.py \\
        --db artifacts/wattbot.db \\
        --questions data/test_Q.csv \\
        --output artifacts/answers.csv \\
        --model gpt-4o-mini

Usage (KohakuEngine):
    kogine run scripts/wattbot_answer.py --config configs/answer_config.py
"""

import asyncio
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


from kohakurag import RAGPipeline
from kohakurag.datastore import ImageStore, KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel
from kohakurag.llm import OpenAIChatModel, OpenRouterChatModel


Row = dict[str, Any]
BLANK_TOKEN = "is_blank"


# ============================================================================
# ASYNC CSV WRITER
# Non-blocking CSV writer that uses asyncio.to_thread for file I/O
# ============================================================================


class AsyncCSVWriter:
    """Async wrapper for csv.DictWriter that performs non-blocking file writes.

    Uses asyncio.to_thread to offload blocking file I/O to a thread pool.
    Supports csv.DictWriter interface with async methods.
    """

    def __init__(self, file_path: Path, fieldnames: Sequence[str]):
        self._file_path = file_path
        self._fieldnames = list(fieldnames)
        self._file = None
        self._writer = None

    async def __aenter__(self) -> "AsyncCSVWriter":
        # Open file in thread pool to avoid blocking
        self._file = await asyncio.to_thread(
            open, self._file_path, "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            await asyncio.to_thread(self._file.close)

    async def writeheader(self) -> None:
        """Write CSV header row asynchronously."""
        if self._writer is None:
            raise RuntimeError("AsyncCSVWriter not opened")
        # Write to in-memory buffer, then flush to file
        await asyncio.to_thread(self._writer.writeheader)
        await asyncio.to_thread(self._file.flush)

    async def writerow(self, row: Row) -> None:
        """Write a single row asynchronously."""
        if self._writer is None:
            raise RuntimeError("AsyncCSVWriter not opened")
        await asyncio.to_thread(self._writer.writerow, row)

    async def flush(self) -> None:
        """Flush file buffer asynchronously."""
        if self._file:
            await asyncio.to_thread(self._file.flush)


# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"

# LLM settings (defaulting to OpenRouter)
llm_provider = "openrouter"  # Options: "openai", "openrouter"
model = "openai/gpt-5-nano"  # Default model
planner_model = None  # Falls back to model
openrouter_api_key = None  # From env: OPENROUTER_API_KEY
site_url = None  # Optional for OpenRouter
app_name = None  # Optional for OpenRouter

# Retrieval settings
top_k = 5
planner_max_queries = 3
deduplicate_retrieval = False  # Deduplicate text results by node_id across queries
rerank_strategy = None  # Options: None, "frequency", "score", "combined"
top_k_final = None  # Optional: truncate to this many results after dedup+rerank (None = no truncation)
snippet_dedup = (
    "none"  # Snippet-level dedup after context expansion: "none", "node_id", "tree"
)
parent_depth = 1  # How many parent levels to include during context expansion
child_depth = 0  # How many child levels to include during context expansion

# Embedding settings
embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
embedding_dim = None  # For JinaV4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # For JinaV4: "retrieval", "text-matching", "code"

# Paragraph search mode (runtime toggle, requires "both" mode during indexing)
# Options:
#   - "averaged": Use sentence-averaged paragraph embeddings (default)
#   - "full": Use full paragraph embeddings (requires index built with "both" or "full" mode)
paragraph_search_mode = "averaged"

# Other settings
metadata = "data/metadata.csv"
max_retries = 3
max_concurrent = 10
single_run_debug = False
question_id = None
with_images = False
top_k_images = 0

# Vision support: send actual images to vision-capable LLMs
send_images_to_llm = False

# BM25 sparse search: additional results from FTS5 (0 = disabled)
# These are added to dense retrieval, NOT fused with scores
bm25_top_k = 0

# Prompt ordering: if True, use reordered prompt (context before question)
use_reordered_prompt = False


# ============================================================================
# PROMPT TEMPLATES
# WattBot-specific prompts live in this script; core library stays generic.
# ============================================================================
ANSWER_SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
The additional info JSON contains an "answer_unit" field indicating the unit for the final answer_value.
You MUST reason about this unit explicitly in your explanation (e.g., what the unit means and how it is applied or converted) and ensure answer_value is expressed in that unit with no unit name included.
""".strip()

PLANNER_SYSTEM_PROMPT = """
Rewrite the user question into focused document search queries.
- Keep the first query identical to the original question.
- Optionally add a few short queries that highlight key entities, numbers, or model names.
- Respond with JSON: {"queries": ["query 1", "query 2", ...]}.
""".strip()

USER_PROMPT_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank".
- The additional info JSON contains an "answer_unit" field; you MUST interpret what this unit means and explain how you apply it in your reasoning.
- Express answer_value in that unit and do NOT include the unit name in answer_value.
- If the answer is a numeric range, format it as [lower,upper] using the requested unit.

Additional info (JSON): {additional_info_json}

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1–3 sentences explaining how the context supports the answer AND how you use the answer_unit; or "is_blank")
- answer               (short sentence in natural language)
- answer_value         (string with ONLY the numeric or categorical value in the requested unit, or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")

JSON Answer:
""".strip()

# ============================================================================
# REORDERED PROMPT TEMPLATES (context BEFORE question to combat attention sink)
# ============================================================================
ANSWER_SYSTEM_PROMPT_REORDERED = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, output "is_blank" for both answer_value and ref_id.

CRITICAL: Match your answer_value to what the question asks for:
- For "Which X..." questions expecting an identifier, return the NAME/identifier, not a numeric value.
- For numeric questions with a unit, return only the number in that unit.
- The "answer_unit" field in additional info tells you the expected format.
""".strip()

USER_PROMPT_TEMPLATE_REORDERED = """
You will answer a question using ONLY the provided context snippets.
If the context does not clearly support an answer, use "is_blank".

Context snippets from documents:
{context}

---

Now answer the following question based ONLY on the context above.

Question: {question}

Additional info (JSON): {additional_info_json}

IMPORTANT: The "answer_unit" field specifies the expected format/unit for answer_value.
- If answer_unit is a unit (e.g., "kW", "USD"), express answer_value as a number in that unit (no unit name).
- If answer_unit is "is_blank", answer_value should be the exact identifier/name from context that answers the question.
- For "Which X has highest Y?" questions with answer_unit="is_blank", return the NAME of X, NOT the numeric value of Y.
- If the answer is a numeric range, format as [lower,upper].

Return STRICT JSON with these keys in order:
- explanation: 1-3 sentences explaining how context supports the answer and how you applied answer_unit
- answer: Short natural language answer
- answer_value: The value matching the expected format (or "is_blank")
- ref_id: List of document IDs from context used as evidence (or "is_blank")

JSON Answer:
""".strip()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_metadata_records(path: Path) -> dict[str, dict[str, str]]:
    """Load document metadata CSV into a lookup dict."""
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            doc_id = row.get("id")
            if not doc_id:
                continue
            records[doc_id] = row
    if not records:
        raise ValueError(f"No metadata rows found in {path}")
    return records


def build_ref_details(
    ref_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    """Convert reference document IDs into URLs and citations.

    Returns:
        (ref_url, supporting_materials) tuple in WattBot CSV format
    """
    urls: list[str] = []
    snippets: list[str] = []

    # Extract URLs and citations from metadata
    for ref_id in ref_ids:
        key = str(ref_id).strip()
        if not key:
            continue
        row = metadata.get(key)
        if not row:
            continue
        url = (row.get("url") or "").strip()
        if url:
            urls.append(url)
        snippet = (row.get("citation") or row.get("title") or key).strip()
        if snippet:
            snippets.append(snippet)

    # Format as WattBot CSV expects: ['url1','url2']
    if urls:
        joined = ",".join(f"'{u}'" for u in urls)
        ref_url = f"[{joined}]"
    else:
        ref_url = BLANK_TOKEN

    supporting = " | ".join(snippets) if snippets else BLANK_TOKEN
    return ref_url, supporting


def normalize_answer_value(raw: str, question: str) -> str:
    """Apply domain-specific normalization to answer values.

    - True/False questions → 1/0
    - Numeric ranges → [lower,upper] format
    """
    value = (raw or "").strip()
    if not value or value.lower() == BLANK_TOKEN:
        return BLANK_TOKEN

    q_lower = question.strip().lower()

    # Normalize True/False questions to 1/0.
    if q_lower.startswith("true or false"):
        v_lower = value.lower()
        if v_lower in {"true", "1"}:
            return "1"
        if v_lower in {"false", "0"}:
            return "0"
        if "true" in v_lower and "false" not in v_lower:
            return "1"
        if "false" in v_lower and "true" not in v_lower:
            return "0"

    # Normalize simple numeric ranges to [lower,upper].
    # Heuristic: if there are exactly two numbers and the text contains a range marker.
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(nums) == 2 and any(
        marker in value for marker in ["-", "–", "—", " to ", "–"]
    ):
        try:
            a = float(nums[0])
            b = float(nums[1])
            lo, hi = sorted((a, b))
            return f"[{lo},{hi}]"
        except ValueError:
            pass

    return value


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_embedder(config):
    """Create embedder based on config settings.

    Args:
        config: Config object with embedding_model, embedding_dim, embedding_task

    Returns:
        EmbeddingModel instance
    """
    model_type = getattr(config, "embedding_model", "jina")

    if model_type == "jinav4":
        dim = getattr(config, "embedding_dim", 1024)
        task = getattr(config, "embedding_task", "retrieval")
        return JinaV4EmbeddingModel(truncate_dim=dim, task=task)
    else:
        # Default: Jina V3
        return JinaEmbeddingModel()


def create_chat_model(config, system_prompt: str):
    """Create chat model based on config settings.

    Args:
        config: Config object with llm_provider, model, etc.
        system_prompt: System prompt for the model

    Returns:
        ChatModel instance
    """
    provider = getattr(config, "llm_provider", "openrouter")

    if provider == "openrouter":
        return OpenRouterChatModel(
            model=config.model,
            api_key=getattr(config, "openrouter_api_key", None),
            site_url=getattr(config, "site_url", None),
            app_name=getattr(config, "app_name", None),
            system_prompt=system_prompt,
            max_concurrent=config.max_concurrent,
        )
    else:
        # Default: OpenAI
        return OpenAIChatModel(
            model=config.model,
            system_prompt=system_prompt,
            max_concurrent=config.max_concurrent,
        )


# ============================================================================
# GLOBAL SHARED EMBEDDER
# Created in main() after config injection
# ============================================================================

GLOBAL_EMBEDDER = None  # Will be set in main()


# ============================================================================
# QUERY PLANNER
# ============================================================================


class LLMQueryPlanner:
    """LLM-backed planner that proposes follow-up retrieval queries."""

    def __init__(self, chat: OpenAIChatModel, max_queries: int = 3) -> None:
        self._chat = chat
        self._max_queries = max(1, max_queries)

    async def plan(self, question: str) -> Sequence[str]:
        """Generate multiple retrieval queries from a single question.

        Strategy:
        1. Always include the original question
        2. Ask LLM to generate paraphrases/entity-focused queries
        3. Fall back to simple reformulation if LLM fails
        """
        base = [question.strip()]
        prompt = f"""
You convert a WattBot question into targeted document search queries.
- The first retrieval query should remain the original question.
- Generate up to {self._max_queries - 1} additional short queries that highlight key entities, units, or paraphrases.
- Respond with JSON: {{"queries": ["query 1", "query 2"]}}
- Return an empty list if the question is already precise.

Question: {question.strip()}

JSON:
""".strip()

        # Ask LLM to generate query variations
        raw = await self._chat.complete(prompt)

        # Parse JSON response
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            extracted = raw[start:end]
            data = json.loads(extracted)
            items = data.get("queries")
            extra = [str(item).strip() for item in items or [] if str(item).strip()]
        except Exception:
            extra = []  # If LLM returns invalid JSON, just use original question

        # Deduplicate and enforce max_queries limit
        seen = {q.lower() for q in base if q}
        for query in extra:
            key = query.lower()
            if key in seen:
                continue
            base.append(query)
            seen.add(key)
            if len(base) >= self._max_queries:
                break

        # Fallback: add simple reformulation if LLM provided nothing useful
        if len(base) == 1:
            reformulation = question.strip().split("?", 1)[0].strip()
            if reformulation and reformulation.lower() not in seen:
                base.append(reformulation)
        return base


# ============================================================================
# DATA LOADING
# ============================================================================


def ensure_columns(row: Row, columns: Sequence[str]) -> Row:
    """Ensure row has all required columns (filling missing with empty strings)."""
    out = {col: row.get(col, "") for col in columns}
    return out


def load_questions(path: Path) -> tuple[list[Row], list[str]]:
    """Load questions CSV and infer column names."""
    with path.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError("Question CSV is empty.")
        columns = reader.fieldnames or [
            "id",
            "question",
            "answer",
            "answer_value",
            "answer_unit",
            "ref_id",
            "ref_url",
            "supporting_materials",
            "explanation",
        ]
    return rows, list(columns)


# ============================================================================
# SHARED RESOURCES
# All async tasks share the same pipeline (thread-safe via async)
# ============================================================================


@dataclass(frozen=True)
class AnswerResult:
    """Result from answering a single question."""

    position: int
    row: Row
    message: str


def create_pipeline() -> RAGPipeline:
    """Create a shared RAG pipeline using global config variables."""
    # Create config object from globals for factory functions
    config = SimpleNamespace(
        llm_provider=llm_provider,
        model=model,
        openrouter_api_key=openrouter_api_key,
        site_url=site_url,
        app_name=app_name,
        max_concurrent=max_concurrent,
    )

    # Datastore (thread-safe via async executor)
    store = KVaultNodeStore(
        Path(db),
        table_prefix=table_prefix,
        dimensions=None,
        paragraph_search_mode=paragraph_search_mode,
    )

    # Create ImageStore if vision support is enabled
    image_store = ImageStore(Path(db)) if send_images_to_llm else None

    # Query planner LLM (generates retrieval queries)
    planner_chat = create_chat_model(config, PLANNER_SYSTEM_PROMPT)
    planner = LLMQueryPlanner(
        chat=planner_chat,
        max_queries=planner_max_queries,
    )

    # Select prompt based on config (reordered or original)
    answer_system_prompt = (
        ANSWER_SYSTEM_PROMPT_REORDERED if use_reordered_prompt else ANSWER_SYSTEM_PROMPT
    )

    # Answer LLM (generates final structured answers)
    chat = create_chat_model(config, answer_system_prompt)

    # Assemble the full RAG pipeline
    pipeline = RAGPipeline(
        store=store,
        embedder=GLOBAL_EMBEDDER,  # Shared embedder
        chat_model=chat,
        planner=planner,
        deduplicate_retrieval=deduplicate_retrieval,
        rerank_strategy=rerank_strategy,
        top_k_final=top_k_final,
        image_store=image_store,
        bm25_top_k=bm25_top_k,
        parent_depth=parent_depth,
        child_depth=child_depth,
        snippet_dedup=snippet_dedup,
    )

    return pipeline


# ============================================================================
# QUESTION ANSWERING
# ============================================================================


async def _answer_single_row(
    idx: int,
    row: Row,
    columns: Sequence[str],
    metadata_records: Mapping[str, Mapping[str, str]],
    pipeline: RAGPipeline,
    retry_count: int | None = None,
    override_top_k: int | None = None,
) -> AnswerResult:
    """Answer a single question with retry logic for blank responses and context overflow handling.

    Strategy:
    - Start with top_k context snippets (or override_top_k if provided)
    - If answer is blank, retry with 2*top_k, then 3*top_k, etc. (unless retry_count=0)
    - If context overflow (400 error), recursively retry with retry_count=0 and k=k-1
    - Return immediately after successful context overflow recovery

    Args:
        retry_count: If 0, skip blank retries. If None, use config.max_retries.
        override_top_k: If provided, use this instead of config.top_k as base.
    """
    if override_top_k is not None and override_top_k <= 0:
        raise ValueError(f"override_top_k={override_top_k} must be > 0")

    question = row["question"]
    additional_info: dict[str, Any] = {
        "answer_unit": (row.get("answer_unit") or "").strip(),
        "question_id": row.get("id", "").strip(),
    }

    # Determine retry behavior
    max_retries_count = 0 if retry_count == 0 else (retry_count or max_retries)
    base_top_k = override_top_k or top_k

    # Retry loop: increase context window each iteration if answer is blank
    qa_result = None
    structured = None

    for attempt in range(max_retries_count + 1):
        current_top_k = base_top_k * (attempt + 1)

        # Also multiply top_k_final by attempt count if configured
        current_top_k_final = None
        if top_k_final is not None:
            current_top_k_final = top_k_final * (attempt + 1)

        # Select prompts based on config (reordered or original)
        current_system_prompt = (
            ANSWER_SYSTEM_PROMPT_REORDERED
            if use_reordered_prompt
            else ANSWER_SYSTEM_PROMPT
        )
        current_user_template = (
            USER_PROMPT_TEMPLATE_REORDERED
            if use_reordered_prompt
            else USER_PROMPT_TEMPLATE
        )

        try:
            qa_result = await pipeline.run_qa(
                question,
                system_prompt=current_system_prompt,
                user_template=current_user_template,
                additional_info=additional_info,
                top_k=current_top_k,
                with_images=with_images,
                top_k_images=top_k_images,
                top_k_final=current_top_k_final,
                send_images_to_llm=send_images_to_llm,
            )
            structured = qa_result.answer
            is_blank = (
                structured.answer_value.strip().lower() == BLANK_TOKEN
                or not structured.ref_id
            )

            if not is_blank:
                break  # Got a valid answer, stop retrying

        except Exception as e:
            # Check if it's a JSON decode error (reduce context)
            if isinstance(e, json.decoder.JSONDecodeError):
                reduced_top_k = current_top_k - 1
                print(
                    f"JSON decode error for row {idx}, reducing top_k from {current_top_k} to {reduced_top_k}"
                )
                return await _answer_single_row(
                    idx,
                    row,
                    columns,
                    metadata_records,
                    pipeline,
                    retry_count=0,
                    override_top_k=reduced_top_k,
                )

            # Check if it's a context length overflow error
            # OpenRouter: "This endpoint's maximum context length is X tokens"
            # OpenAI: "This model's maximum context length is X tokens" + code: "context_length_exceeded"
            error_msg = str(e).lower()
            is_context_overflow = (
                "maximum context length" in error_msg
                or "context_length_exceeded" in error_msg
            )

            if is_context_overflow:
                reduced_top_k = current_top_k - 1
                print(
                    f"Context overflow for row {idx}, reducing top_k from {current_top_k} to {reduced_top_k}"
                )
                return await _answer_single_row(
                    idx,
                    row,
                    columns,
                    metadata_records,
                    pipeline,
                    retry_count=0,
                    override_top_k=reduced_top_k,
                )
            else:
                raise  # Not a recoverable error, propagate

    # Create blank result if we don't have a valid answer
    if qa_result is None or structured is None:
        structured = SimpleNamespace(
            answer="", answer_value=BLANK_TOKEN, ref_id=[], explanation=""
        )

    is_blank = (
        structured.answer_value.strip().lower() == BLANK_TOKEN or not structured.ref_id
    )

    # Format output row based on answer status
    result = dict(row)

    if is_blank:
        result["answer"] = BLANK_TOKEN
        result["answer_value"] = BLANK_TOKEN
        result["ref_id"] = BLANK_TOKEN
        result["ref_url"] = BLANK_TOKEN
        result["supporting_materials"] = BLANK_TOKEN
        result["explanation"] = BLANK_TOKEN

    else:
        # Populate with structured answer fields
        result["answer"] = structured.answer or BLANK_TOKEN

        normalized_value = normalize_answer_value(structured.answer_value, question)
        result["answer_value"] = normalized_value or BLANK_TOKEN

        # Format ref_id as list: ['doc1','doc2']
        if structured.ref_id:
            joined_ids = ",".join(f"'{rid}'" for rid in structured.ref_id)
            result["ref_id"] = f"[{joined_ids}]"
        else:
            result["ref_id"] = BLANK_TOKEN

        # Resolve URLs and citations from metadata
        ref_url, supporting = build_ref_details(structured.ref_id, metadata_records)
        result["ref_url"] = ref_url
        result["supporting_materials"] = supporting
        result["explanation"] = structured.explanation or BLANK_TOKEN

    # Build progress message
    row_id = row.get("id") or row.get("question", "")[:24] or f"row-{idx}"
    message = f"Answered {row_id} - {question[:60]}..."

    return AnswerResult(
        position=idx,
        row=ensure_columns(result, columns),
        message=message,
    )


async def answer_questions(
    rows: Sequence[Row],
    columns: Sequence[str],
    metadata_records: Mapping[str, Mapping[str, str]],
    pipeline: RAGPipeline,
):
    """Process all questions concurrently, yielding results as they complete."""
    # Create async tasks for all questions
    tasks = [
        asyncio.create_task(
            _answer_single_row(idx, row, columns, metadata_records, pipeline)
        )
        for idx, row in enumerate(rows)
    ]

    # Yield results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result.message)
        yield result


# ============================================================================
# SINGLE-RUN DEBUG MODE
# ============================================================================


async def run_single_question_debug(
    first_row: Row,
    columns: Sequence[str],
    metadata_records: Mapping[str, Mapping[str, str]],
    pipeline: RAGPipeline,
    writer: AsyncCSVWriter,
) -> None:
    """Run a single question with detailed debugging output."""
    question = first_row["question"]
    additional_info: dict[str, Any] = {
        "answer_unit": (first_row.get("answer_unit") or "").strip(),
        "question_id": first_row.get("id", "").strip(),
    }

    attempts_log: list[dict[str, Any]] = []
    qa_result = None
    structured = None
    is_blank = True

    # Select prompts based on config (reordered or original)
    current_system_prompt = (
        ANSWER_SYSTEM_PROMPT_REORDERED if use_reordered_prompt else ANSWER_SYSTEM_PROMPT
    )
    current_user_template = (
        USER_PROMPT_TEMPLATE_REORDERED if use_reordered_prompt else USER_PROMPT_TEMPLATE
    )

    # Retry with increasing context window
    for attempt in range(max_retries + 1):
        current_top_k = top_k * (attempt + 1)
        qa_result = await pipeline.run_qa(
            question,
            system_prompt=current_system_prompt,
            user_template=current_user_template,
            additional_info=additional_info,
            top_k=current_top_k,
            with_images=with_images,
            top_k_images=top_k_images,
            send_images_to_llm=send_images_to_llm,
        )
        structured = qa_result.answer
        is_blank = (
            structured.answer_value.strip().lower() == BLANK_TOKEN
            or not structured.ref_id
        )
        attempts_log.append(
            {
                "attempt": attempt + 1,
                "top_k": current_top_k,
                "is_blank": is_blank,
                "prompt": qa_result.prompt,
                "raw": qa_result.raw_response,
                "parsed": {
                    "answer": structured.answer,
                    "answer_value": structured.answer_value,
                    "ref_id": structured.ref_id,
                    "explanation": structured.explanation,
                },
            }
        )

        if not is_blank:
            break  # Got answer, stop retrying

    assert qa_result is not None and structured is not None

    # Format output row
    result_row: Row = dict(first_row)

    if is_blank:
        result_row["answer"] = BLANK_TOKEN
        result_row["answer_value"] = BLANK_TOKEN
        result_row["ref_id"] = BLANK_TOKEN
        result_row["ref_url"] = BLANK_TOKEN
        result_row["supporting_materials"] = BLANK_TOKEN
        result_row["explanation"] = BLANK_TOKEN
    else:
        result_row["answer"] = structured.answer or BLANK_TOKEN
        normalized_value = normalize_answer_value(structured.answer_value, question)
        result_row["answer_value"] = normalized_value or BLANK_TOKEN
        if structured.ref_id:
            joined_ids = ",".join(f"'{rid}'" for rid in structured.ref_id)
            result_row["ref_id"] = f"[{joined_ids}]"
        else:
            result_row["ref_id"] = BLANK_TOKEN
        ref_url, supporting = build_ref_details(structured.ref_id, metadata_records)
        result_row["ref_url"] = ref_url
        result_row["supporting_materials"] = supporting
        result_row["explanation"] = structured.explanation or BLANK_TOKEN

    # Write result and print debug info
    await writer.writerow(ensure_columns(result_row, columns))
    await writer.flush()

    print("=== Single-run debug ===")
    print(f"Question ID: {first_row.get('id')}")
    print(f"Question: {question}")

    for entry in attempts_log:
        print(f"\n--- Attempt {entry['attempt']} (top_k={entry['top_k']}) ---")
        print(f"is_blank: {entry['is_blank']}")
        print("\nPrompt:\n")
        print(entry["prompt"])
        print("\nRaw model output:\n")
        print(entry["raw"])
        print("\nParsed structured answer:\n")
        print(json.dumps(entry["parsed"], ensure_ascii=False, indent=2))


# ============================================================================
# STANDARD RUN MODE
# ============================================================================


async def run_all_questions(
    rows: Sequence[Row],
    columns: Sequence[str],
    metadata_records: Mapping[str, Mapping[str, str]],
    pipeline: RAGPipeline,
    writer: AsyncCSVWriter,
) -> None:
    """Process all questions and write results as they complete."""
    async for answer in answer_questions(rows, columns, metadata_records, pipeline):
        await writer.writerow(answer.row)
        await writer.flush()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main() -> None:
    """Main entry point: load data, create pipeline, process questions."""
    global GLOBAL_EMBEDDER

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create embedder using injected config values
    embedder_config = SimpleNamespace(
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_task=embedding_task,
    )
    GLOBAL_EMBEDDER = create_embedder(embedder_config)

    # Load input data
    rows, columns = load_questions(Path(questions))
    metadata_records = load_metadata_records(Path(metadata))

    # Create shared pipeline (uses global config variables)
    pipeline = create_pipeline()

    async with AsyncCSVWriter(output_path, columns) as writer:
        await writer.writeheader()

        # DEBUG MODE: Process one question with detailed logging
        if single_run_debug:
            if not rows:
                raise ValueError("No questions found for single-run debug.")

            # Select question to debug (by ID or first row)
            if question_id:
                target_row: Row | None = None
                for row in rows:
                    if row.get("id") == question_id:
                        target_row = row
                        break
                if target_row is None:
                    raise ValueError(
                        f"Question id {question_id} not found in {questions}"
                    )
                first_row = target_row
            else:
                first_row = rows[0]

            await run_single_question_debug(
                first_row, columns, metadata_records, pipeline, writer
            )

        # STANDARD MODE: Process all questions concurrently, streaming results
        else:
            await run_all_questions(rows, columns, metadata_records, pipeline, writer)


def entry_point():
    print(output)
    asyncio.run(main())


if __name__ == "__main__":
    entry_point()

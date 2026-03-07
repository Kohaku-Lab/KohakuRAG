"""High-level RAG pipeline orchestration."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol, Sequence

from .datastore import HierarchicalNodeStore, InMemoryNodeStore, matches_to_snippets

if TYPE_CHECKING:
    from .datastore import ImageStore
from .embeddings import EmbeddingModel, JinaEmbeddingModel
from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


# ============================================================================
# PROTOCOLS
# ============================================================================


class ChatModel(Protocol):
    """Protocol for chat backends."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


class QueryPlanner(Protocol):
    """Protocol for query expansion/rewriting."""

    async def plan(self, question: str) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class RetrievalResult:
    """Container for retrieval outputs."""

    question: str
    matches: list[RetrievalMatch]  # Direct vector search results
    snippets: list[ContextSnippet]  # Expanded with parent/child context
    image_nodes: list[StoredNode] | None = (
        None  # All images (for backward compatibility)
    )
    images_from_text: list[StoredNode] | None = (
        None  # Images from text retrieval (caption only)
    )
    images_from_vision: list[StoredNode] | None = (
        None  # Images from image search (send as images)
    )


@dataclass
class StructuredAnswer:
    """Structured answer format (for WattBot and similar tasks)."""

    answer: str
    answer_value: str
    ref_id: list[str]
    explanation: str


@dataclass
class StructuredAnswerResult:
    """Complete result from structured QA pipeline."""

    answer: StructuredAnswer
    retrieval: RetrievalResult
    raw_response: str
    prompt: str


@dataclass
class PromptTemplate:
    """Template for building LLM prompts with dynamic context."""

    system_prompt: str
    user_template: str  # Must have {question}, {context}, {additional_info_json}
    additional_info: Mapping[str, object] | None = None

    def render(
        self,
        *,
        question: str,
        snippets: Sequence[ContextSnippet],
        image_nodes: Sequence[StoredNode] | None = None,
    ) -> str:
        """Fill template with question and retrieved context.

        Args:
            question: User question
            snippets: Retrieved context snippets
            image_nodes: Optional image nodes from sections (for image-aware RAG)

        Returns:
            Rendered prompt string
        """
        # Format context with optional images
        if image_nodes:
            context = format_context_with_images(snippets, image_nodes)
        else:
            context = format_snippets(snippets)

        extras = self.additional_info or {}
        extras_json = json.dumps(extras, ensure_ascii=False)

        return self.user_template.format(
            question=question,
            context=context,
            additional_info_json=extras_json,
            additional_info=extras,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_snippets(snippets: Sequence[ContextSnippet]) -> str:
    """Render snippets as formatted context string for LLM prompt.

    Format: [ref_id=doc node=sec1:p2 score=0.95] snippet text
    Snippets separated by --- lines for readability.
    """
    blocks: list[str] = []

    for snippet in snippets:
        meta = snippet.metadata or {}
        doc_id = str(meta.get("document_id", "unknown"))
        full_node_id = snippet.node_id

        # Compact node ID: amazon2023:sec1:p2 → sec1:p2
        node_label = (
            full_node_id.split(":", 1)[1] if ":" in full_node_id else full_node_id
        )

        # header = f"[ref_id={doc_id} node={node_label} score={snippet.score:.3f}] "
        header = f"[ref_id={doc_id}] "  # only necessary info to avoid LLM hallucination and waste tokens
        text = snippet.text.strip()
        blocks.append(header + text)

    return "\n---\n".join(blocks)


def format_image_nodes(image_nodes: Sequence[StoredNode]) -> str:
    """Format image nodes for LLM prompt.

    Format:
        [ref_id=doc1] [img:name WxH] Caption text...

        [ref_id=doc2] [img:name2 WxH2] Caption text 2...

    Returns empty string if no images.
    """
    if not image_nodes:
        return ""

    blocks: list[str] = []
    for node in image_nodes:
        # Get document ID from metadata
        doc_id = node.metadata.get("document_id", "unknown")

        # Image text is already in format: [img:name WxH] caption...
        # Add ref_id prefix to match text snippet format
        formatted = f"[ref_id={doc_id}] {node.text.strip()}"
        blocks.append(formatted)

    return "\n\n".join(blocks)


def format_context_with_images(
    snippets: Sequence[ContextSnippet],
    image_nodes: Sequence[StoredNode] | None = None,
) -> str:
    """Format context with separate sections for text and images.

    Format:
        Context snippets:
        [ref_id=doc1] Text...
        ---
        [ref_id=doc2] More text...

        Referenced media:
        [img:Fig1 800x600] Bar chart showing...

        [img:Fig2 1200x900] Diagram of system...
    """
    context = format_snippets(snippets)

    if image_nodes:
        image_text = format_image_nodes(image_nodes)
        if image_text:
            context += "\n\nReferenced media:\n" + image_text

    return context


def build_multimodal_content(
    text_content: str,
    image_nodes: Sequence[StoredNode] | None,
    image_store: ImageStore | None = None,
) -> str | list[dict]:
    """Build multimodal content for vision-capable LLMs.

    Args:
        text_content: The main text prompt
        image_nodes: Image nodes to include (must have image_storage_key in metadata)
        image_store: ImageStore instance for retrieving image bytes

    Returns:
        - If no images or no image_store: returns text_content as string
        - Otherwise: returns list of content parts for multimodal LLM
    """
    if not image_nodes or image_store is None:
        return text_content

    content_parts: list[dict] = [{"type": "text", "text": text_content}]

    for node in image_nodes:
        storage_key = node.metadata.get("image_storage_key")
        if not storage_key:
            continue

        try:
            # Use sync method directly (avoids async complexity in prompt building)
            image_bytes = image_store._sync_get(storage_key)
            if image_bytes:
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{image_b64}"},
                    }
                )
        except Exception:
            continue  # Skip failed images

    return content_parts if len(content_parts) > 1 else text_content


# ============================================================================
# DEFAULT IMPLEMENTATIONS
# ============================================================================


class SimpleQueryPlanner:
    """Pass-through planner that uses the raw question without expansion."""

    async def plan(self, question: str) -> Sequence[str]:
        """Return single-element list containing the original question."""
        return [question]


class MockChatModel:
    """Dummy LLM for testing (returns truncated context)."""

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Extract context from prompt and return as mock answer."""
        return "Mock response:\n" + prompt.split("Context:", 1)[-1].strip()[:200]


# ============================================================================
# RAG PIPELINE
# ============================================================================


class RAGPipeline:
    """Coordinates query planning, retrieval, and LLM answering."""

    def __init__(
        self,
        *,
        store: HierarchicalNodeStore | None = None,
        embedder: EmbeddingModel | None = None,
        chat_model: ChatModel | None = None,
        planner: QueryPlanner | None = None,
        top_k: int = 5,
        deduplicate_retrieval: bool = False,
        rerank_strategy: str | None = None,
        top_k_final: int | None = None,
        image_store: ImageStore | None = None,
        bm25_top_k: int = 0,
        parent_depth: int = 1,
        child_depth: int = 0,
        snippet_dedup: str = "none",
        # Deprecated — kept for backward compatibility
        no_overlap: bool = False,
        tree_dedup: bool = False,
    ) -> None:
        """Initialize RAG pipeline with pluggable components.

        All components default to in-memory/mock implementations for testing.

        Args:
            store: Vector database for storing and searching nodes
            embedder: Model for converting text to embeddings
            chat_model: LLM for generating answers
            planner: Query expansion strategy
            top_k: Default number of results per query
            deduplicate_retrieval: Whether to deduplicate results by node_id
                before reranking (for efficiency; reranking also deduplicates)
            rerank_strategy: Strategy for reranking multi-query results
                            Options: None, "frequency", "score", "combined"
            top_k_final: Optional truncation after dedup+rerank (None = no truncation)
            image_store: Optional ImageStore for vision-enabled LLM support
            bm25_top_k: Number of additional results from BM25 sparse search (0 = disabled).
            parent_depth: How many parent levels to include during context expansion.
            child_depth: How many child levels to include during context expansion.
            snippet_dedup: Snippet-level deduplication applied *after* context expansion.
                Options: "none", "node_id", "tree".
            no_overlap: **Deprecated** — use ``snippet_dedup="tree"`` instead.
            tree_dedup: **Deprecated** — use ``snippet_dedup="tree"`` instead.
        """
        self._store = store or InMemoryNodeStore()
        self._embedder = embedder or JinaEmbeddingModel()
        self._chat = chat_model or MockChatModel()
        self._planner = planner or SimpleQueryPlanner()
        self._top_k = top_k
        self._deduplicate = deduplicate_retrieval
        self._rerank_strategy = rerank_strategy
        self._top_k_final = top_k_final
        self._image_store = image_store
        self._bm25_top_k = bm25_top_k
        self._parent_depth = parent_depth
        self._child_depth = child_depth
        # Backward compatibility: map old flags to snippet_dedup
        if (tree_dedup or no_overlap) and snippet_dedup == "none":
            snippet_dedup = "tree"
        self._snippet_dedup = snippet_dedup

    @property
    def store(self) -> HierarchicalNodeStore:
        return self._store

    async def index_documents(self, documents: Iterable[StoredNode]) -> None:
        """Bulk insert pre-built nodes into the store."""
        await self._store.upsert_nodes(list(documents))

    def _deduplicate_matches(
        self, matches: list[RetrievalMatch]
    ) -> list[RetrievalMatch]:
        """Deduplicate matches by node_id, keeping the first occurrence.

        When the same node appears in results from multiple queries, we keep
        only the first occurrence to avoid duplicate context.

        Args:
            matches: List of retrieval matches (potentially with duplicates)

        Returns:
            Deduplicated list of matches
        """
        seen_ids: set[str] = set()
        unique_matches: list[RetrievalMatch] = []

        for match in matches:
            if match.node.node_id not in seen_ids:
                seen_ids.add(match.node.node_id)
                unique_matches.append(match)

        return unique_matches

    def _tree_deduplicate_matches(
        self, matches: list[RetrievalMatch]
    ) -> list[RetrievalMatch]:
        """Remove matches whose ancestor is also present in the results.

        When multi-level retrieval returns both a parent node and its child
        (e.g., paragraph "doc:sec1:p2" and sentence "doc:sec1:p2:s3"), the
        child's text is contained within the parent. This method removes such
        children to avoid redundant context.

        Ancestor relationship is determined by node_id prefix:
        "doc:sec1:p2:s3" is a descendant of "doc:sec1:p2" because it starts
        with "doc:sec1:p2:".

        Args:
            matches: List of retrieval matches (already deduplicated by node_id)

        Returns:
            Filtered list with no descendant nodes whose ancestor is present
        """
        if not matches:
            return matches

        all_ids = {m.node.node_id for m in matches}

        # Find ids that have an ancestor in the set
        ids_with_ancestor: set[str] = set()
        for node_id in all_ids:
            for other_id in all_ids:
                if other_id != node_id and node_id.startswith(other_id + ":"):
                    ids_with_ancestor.add(node_id)
                    break

        if not ids_with_ancestor:
            return matches

        return [m for m in matches if m.node.node_id not in ids_with_ancestor]

    def _rerank_matches(
        self, matches: list[RetrievalMatch], num_queries: int
    ) -> list[RetrievalMatch]:
        """Rerank matches based on the configured strategy.

        This method aggregates duplicate nodes and ranks them using
        frequency (how many queries returned it) and total score (sum of scores).

        Strategies:
        - "frequency": Sort by (frequency, total_score) descending
        - "score": Sort by total_score only (descending)
        - "combined": Sort by weighted combination of normalized frequency and total_score

        Args:
            matches: List of retrieval matches (potentially with duplicates)
            num_queries: Number of queries used for retrieval

        Returns:
            Reranked and deduplicated list of matches
        """
        if not self._rerank_strategy or not matches:
            return matches

        strategy = self._rerank_strategy.lower()

        # Aggregate stats for each unique node
        node_stats: dict[str, dict] = {}

        for match in matches:
            node_id = match.node.node_id
            if node_id not in node_stats:
                node_stats[node_id] = {
                    "match": match,  # Keep reference to match object
                    "frequency": 0,
                    "total_score": 0.0,
                    "max_score": match.score,
                }

            node_stats[node_id]["frequency"] += 1
            node_stats[node_id]["total_score"] += match.score
            node_stats[node_id]["max_score"] = max(
                node_stats[node_id]["max_score"], match.score
            )

        # Extract unique matches with aggregated scores
        unique_matches = []
        for stats in node_stats.values():
            # Update the match object's score to reflect total_score
            match = stats["match"]
            unique_matches.append(match)

        # Sort based on strategy
        if strategy == "frequency":
            # Primary: frequency, Secondary: total_score
            unique_matches.sort(
                key=lambda m: (
                    node_stats[m.node.node_id]["frequency"],
                    node_stats[m.node.node_id]["total_score"],
                ),
                reverse=True,
            )

        elif strategy == "score":
            # Sort by total_score only
            unique_matches.sort(
                key=lambda m: node_stats[m.node.node_id]["total_score"],
                reverse=True,
            )

        elif strategy == "combined":
            # Weighted combination of normalized frequency and total_score
            max_freq = max(s["frequency"] for s in node_stats.values())
            max_total_score = max(s["total_score"] for s in node_stats.values())

            # Avoid division by zero
            max_freq = max(max_freq, 1)
            max_total_score = max(max_total_score, 0.001)

            unique_matches.sort(
                key=lambda m: (
                    0.4 * (node_stats[m.node.node_id]["frequency"] / max_freq)
                    + 0.6
                    * (node_stats[m.node.node_id]["total_score"] / max_total_score)
                ),
                reverse=True,
            )

        else:
            # Unknown strategy, return as-is (deduplicated by first occurrence)
            pass

        return unique_matches

    async def retrieve(
        self, question: str, *, top_k: int | None = None, bm25_top_k: int | None = None
    ) -> RetrievalResult:
        """Execute multi-query retrieval with hierarchical context expansion.

        For each planner-generated query, we independently search the vector
        store for the top-k matching nodes (sentences or paragraphs).

        Behavior:
        - If deduplicate_retrieval=False and rerank_strategy=None:
          Results are simply concatenated in planner order (original behavior)
        - If deduplicate_retrieval=True:
          Duplicate nodes (by node_id) are removed, keeping first occurrence
        - If snippet_dedup="tree":
          After context expansion, snippets whose ancestor is also present are removed
          (e.g., sentence "doc:sec1:p2:s3" removed if paragraph "doc:sec1:p2" exists)
        - If rerank_strategy is set:
          Results are reranked using the specified strategy
          ("frequency", "score", or "combined") with frequency + total_score
        - If top_k_final is set:
          Results are truncated to top_k_final after dedup+rerank
        - If bm25_top_k > 0:
          Additional BM25 results are appended after dense results for context expansion.
          These are NOT fused with dense scores - they add complementary keyword matches.

        Example configurations:
        1. top_k=16, max_queries=3, deduplicate=False, rerank=None, top_k_final=None
           -> 48 results (16 * 3, with potential duplicates)

        2. top_k=16, max_queries=3, deduplicate=True, rerank=None, top_k_final=None
           -> up to 48 unique results (duplicates removed)

        3. top_k=16, max_queries=3, deduplicate=True, rerank="frequency", top_k_final=20
           -> 20 results (best ranked by frequency + total score)

        4. top_k=8, bm25_top_k=4, deduplicate=True
           -> dense results (up to 8 per query) + additional BM25 results (up to 4)

        Args:
            question: User question
            top_k: Number of results per query (uses default if None)
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            RetrievalResult with matches and expanded snippets
        """
        # Generate multiple retrieval queries (or just one if simple planner)
        queries = list(await self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)
        k = top_k or self._top_k
        bm25_k = bm25_top_k if bm25_top_k is not None else self._bm25_top_k

        # Execute each query independently (dense search)
        all_matches: list[RetrievalMatch] = []
        for vector in query_vectors:
            matches = await self._store.search(
                vector,
                k=k,
                kinds={
                    NodeKind.SENTENCE,
                    NodeKind.PARAGRAPH,
                },  # Skip documents/sections
            )
            all_matches.extend(matches)

        # Apply deduplication if enabled (before reranking)
        if self._deduplicate:
            all_matches = self._deduplicate_matches(all_matches)

        # Apply reranking if strategy is configured
        # Note: reranking also deduplicates and uses frequency + total_score
        if self._rerank_strategy:
            all_matches = self._rerank_matches(all_matches, len(queries))

        # Apply top_k_final truncation if configured
        if self._top_k_final is not None and self._top_k_final > 0:
            all_matches = all_matches[: self._top_k_final]

        # Add BM25 results for additional context (not fused, just appended)
        if bm25_k > 0 and hasattr(self._store, "search_bm25"):
            # Collect node_ids already in dense results
            dense_node_ids = {m.node.node_id for m in all_matches}

            # Search BM25 for each query
            bm25_matches: list[RetrievalMatch] = []
            for query in queries:
                matches = await self._store.search_bm25(
                    query,
                    k=bm25_k,
                    kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH},
                )
                bm25_matches.extend(matches)

            # Deduplicate BM25 results and exclude nodes already in dense results
            seen_bm25_ids: set[str] = set()
            for match in bm25_matches:
                node_id = match.node.node_id
                if node_id not in dense_node_ids and node_id not in seen_bm25_ids:
                    seen_bm25_ids.add(node_id)
                    all_matches.append(match)
                    # Stop if we've added enough BM25 results
                    if len(seen_bm25_ids) >= bm25_k:
                        break

        # Expand each match with hierarchical context, then deduplicate snippets
        snippets = await matches_to_snippets(
            all_matches,
            self._store,
            parent_depth=self._parent_depth,
            child_depth=self._child_depth,
            dedup=self._snippet_dedup,
        )

        return RetrievalResult(
            question=question,
            matches=all_matches,
            snippets=snippets,
        )

    async def _extract_images_from_snippets(
        self, snippets: Sequence[ContextSnippet]
    ) -> list[StoredNode]:
        """Extract image nodes from retrieved sections.

        Looks at all sections containing retrieved snippets and collects
        their image children (paragraphs with attachment_type='image').

        Args:
            snippets: Retrieved context snippets

        Returns:
            List of image nodes
        """
        image_nodes: list[StoredNode] = []
        seen_sections: set[str] = set()

        for snippet in snippets:
            # Get section ID from node ID (format: doc:sec:p:s → doc:sec)
            parts = snippet.node_id.split(":")
            if len(parts) >= 2:
                section_id = ":".join(parts[:2])
            else:
                continue  # Not a hierarchical node

            # Skip if we already processed this section
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)

            try:
                # Get the section node
                section_node = await self._store.get_node(section_id)

                # Check all children for images
                for child_id in section_node.child_ids:
                    try:
                        child_node = await self._store.get_node(child_id)

                        # Check if this is an image node
                        if child_node.metadata.get("attachment_type") == "image":
                            image_nodes.append(child_node)

                    except KeyError:
                        continue  # Child node not found

            except KeyError:
                continue  # Section node not found

        return image_nodes

    async def retrieve_with_images(
        self,
        question: str,
        *,
        top_k: int | None = None,
        top_k_images: int = 0,
        bm25_top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute multi-query retrieval with image extraction.

        Image retrieval strategy:
        1. Extract images from retrieved text sections (use captions only in LLM)
        2. Additionally retrieve from image-only index if top_k_images > 0 (send as actual images to vision LLM)
        3. Combine and deduplicate for backward compatibility (image_nodes)

        Args:
            question: User question
            top_k: Number of text results per query (uses default if None)
            top_k_images: Number of ADDITIONAL images from image-only index
                         (0 = only extract from sections, >0 = also search image index)
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            RetrievalResult with separate image sources:
            - images_from_text: Images from text retrieval (use captions)
            - images_from_vision: Images from image search (send as actual images)
            - image_nodes: Combined for backward compatibility
        """
        # Standard text retrieval (with BM25 if configured)
        result = await self.retrieve(question, top_k=top_k, bm25_top_k=bm25_top_k)

        # Images from text retrieval sections (captions only)
        images_from_sections = await self._extract_images_from_snippets(result.snippets)

        # Images from dedicated image search (send as actual images)
        if top_k_images > 0:
            images_from_index = await self._retrieve_images_only(question, top_k_images)
        else:
            images_from_index = []

        # Combine and deduplicate for backward compatibility
        all_images = []
        seen_ids = set()

        # Prioritize images from sections
        for node in images_from_sections:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        # Add images from index
        for node in images_from_index:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        return RetrievalResult(
            question=result.question,
            matches=result.matches,
            snippets=result.snippets,
            image_nodes=all_images if all_images else None,  # Backward compatibility
            images_from_text=images_from_sections if images_from_sections else None,
            images_from_vision=images_from_index if images_from_index else None,
        )

    async def _retrieve_images_only(self, question: str, k: int) -> list[StoredNode]:
        """Retrieve top-k images using dedicated image-only vector index.

        Args:
            question: User question
            k: Number of images to retrieve

        Returns:
            List of image nodes (empty if image index doesn't exist)
        """
        # Check if image-only index exists
        if not hasattr(self._store, "search_images"):
            return []

        # Generate retrieval queries
        queries = list(await self._planner.plan(question))
        if not queries:
            return []

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)

        # Search image-only index for each query
        all_image_matches: list[StoredNode] = []
        for vector in query_vectors:
            matches = await self._store.search_images(vector, k=k)
            all_image_matches.extend([m.node for m in matches])

        # Deduplicate by node_id (in case same image matched multiple queries)
        seen_ids = set()
        unique_images = []
        for node in all_image_matches:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_images.append(node)

        return unique_images[:k]  # Limit to top-k overall

    async def answer(self, question: str) -> dict:
        """Simple QA: retrieve + prompt + generate (returns unstructured dict)."""
        retrieval = await self.retrieve(question)
        prompt = self._build_prompt(question, retrieval.snippets)
        response = await self._chat.complete(prompt)

        return {
            "question": question,
            "response": response,
            "snippets": retrieval.snippets,
        }

    async def structured_answer(
        self,
        question: str,
        prompt: PromptTemplate,
        *,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
        top_k_final: int | None = None,
        send_images_to_llm: bool = False,
        bm25_top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """QA with custom prompt template and structured JSON parsing.

        Args:
            question: User question
            prompt: Prompt template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (0 = extract from sections)
            top_k_final: Optional override for final result truncation
            send_images_to_llm: If True and image_store is set, send actual images
                               to vision-capable LLMs instead of just captions
            bm25_top_k: Number of additional BM25 results (uses default if None)

        Returns:
            Complete structured answer result
        """
        # Temporarily override top_k_final if provided
        original_top_k_final = self._top_k_final
        if top_k_final is not None:
            self._top_k_final = top_k_final

        try:
            # Use image-aware retrieval if requested
            if with_images:
                retrieval = await self.retrieve_with_images(
                    question,
                    top_k=top_k,
                    top_k_images=top_k_images,
                    bm25_top_k=bm25_top_k,
                )
            else:
                retrieval = await self.retrieve(
                    question, top_k=top_k, bm25_top_k=bm25_top_k
                )
        finally:
            # Restore original top_k_final
            self._top_k_final = original_top_k_final

        # Render user prompt with context (and images if present as captions)
        rendered_prompt = prompt.render(
            question=question,
            snippets=retrieval.snippets,
            image_nodes=retrieval.image_nodes,
        )

        # Build multimodal content if vision support is enabled
        if send_images_to_llm and retrieval.images_from_vision and self._image_store:
            prompt_content = build_multimodal_content(
                rendered_prompt,
                retrieval.images_from_vision,
                self._image_store,
            )
        else:
            prompt_content = rendered_prompt

        # Get LLM response
        raw = await self._chat.complete(
            prompt_content, system_prompt=prompt.system_prompt
        )

        # Parse JSON structure
        parsed = self._parse_structured_response(raw)

        return StructuredAnswerResult(
            answer=parsed,
            retrieval=retrieval,
            raw_response=raw,
            prompt=rendered_prompt,
        )

    async def run_qa(
        self,
        question: str,
        *,
        system_prompt: str,
        user_template: str,
        additional_info: Mapping[str, object] | None = None,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
        top_k_final: int | None = None,
        send_images_to_llm: bool = False,
        bm25_top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """High-level entry point for structured question answering.

        This method keeps the RAG core generic by requiring callers to supply
        their own system prompt, user prompt template, and any per-call metadata
        via ``additional_info``.

        Args:
            question: User question
            system_prompt: System prompt for LLM
            user_template: User prompt template
            additional_info: Extra metadata for template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (requires wattbot_build_image_index.py)
            top_k_final: Optional override for final result truncation
            send_images_to_llm: If True, send actual images to vision-capable LLMs
            bm25_top_k: Number of additional BM25 results (uses pipeline default if None)

        Returns:
            Structured answer result
        """
        template = PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            additional_info=additional_info,
        )
        return await self.structured_answer(
            question=question,
            prompt=template,
            top_k=top_k,
            with_images=with_images,
            top_k_images=top_k_images,
            top_k_final=top_k_final,
            send_images_to_llm=send_images_to_llm,
            bm25_top_k=bm25_top_k,
        )

    def _build_prompt(
        self,
        question: str,
        snippets: Sequence[ContextSnippet],
    ) -> str:
        """Build simple prompt for answer() method."""
        context_blocks = []
        for snippet in snippets:
            context_blocks.append(
                f"[{snippet.document_title} | node={snippet.node_id} | score={snippet.score:.3f}]\n{snippet.text}"
            )

        context_text = "\n\n".join(context_blocks) if context_blocks else "None"

        return (
            "You are an assistant.\n"
            "Use only the provided context to answer the question.\n"
            "If the context is insufficient, respond with 'NOT ENOUGH DATA'.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )

    def _parse_structured_response(self, raw: str) -> StructuredAnswer:
        """Extract JSON from LLM response and validate fields."""
        try:
            # Find JSON block in response
            start = raw.index("{")
            end = raw.rindex("}") + 1
            snippet = raw[start:end]
            data = json.loads(snippet)

        except Exception:
            # Return empty structure if parsing fails
            return StructuredAnswer(
                answer="",
                answer_value="",
                ref_id=[],
                explanation="",
            )

        # Extract and normalize fields
        answer = str(data.get("answer", "")).strip()
        answer_value = str(data.get("answer_value", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        # Parse ref_id (can be string or list)
        ref_ids_raw = data.get("ref_id", [])
        ref_ids: list[str] = []

        if isinstance(ref_ids_raw, str):
            ref_ids_raw = [ref_ids_raw]

        if isinstance(ref_ids_raw, Sequence):
            for item in ref_ids_raw:
                text = str(item).strip()
                if text:
                    # Clean up common LLM mistakes: strip "ref_id=" prefix
                    # LLM sometimes copies the context format like [ref_id=doc1]
                    if text.lower().startswith("ref_id="):
                        text = text[7:].strip()  # Remove "ref_id=" prefix
                    ref_ids.append(text)

        return StructuredAnswer(
            answer=answer,
            answer_value=answer_value,
            ref_id=ref_ids,
            explanation=explanation,
        )

"""Simple hierarchical vector store implementations."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from kohakuvault import KVault, TextVault, VectorKVault

from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode

# Runtime embedding selection mode for paragraphs
ParagraphSearchMode = Literal["averaged", "full", "both"]


class HierarchicalNodeStore:
    """Abstract interface for node stores."""

    async def upsert_nodes(
        self, nodes: Sequence[StoredNode]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    async def get_node(self, node_id: str) -> StoredNode:  # pragma: no cover
        raise NotImplementedError

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:  # pragma: no cover
        raise NotImplementedError

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:  # pragma: no cover
        raise NotImplementedError


class InMemoryNodeStore(HierarchicalNodeStore):
    """In-memory store with brute-force cosine search (for testing/development)."""

    def __init__(self) -> None:
        self._nodes: dict[str, StoredNode] = {}

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Store nodes in memory (overwrites existing)."""
        for node in nodes:
            self._nodes[node.node_id] = node

    async def get_node(self, node_id: str) -> StoredNode:
        return self._nodes[node_id]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        """Brute-force linear scan with cosine similarity."""
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D numpy array.")

        normalized_query = self._normalize(query_vector)
        matches: list[RetrievalMatch] = []

        # Compare query against all nodes
        for node in self._nodes.values():
            # Filter by node type if specified
            if kinds is not None and node.kind not in kinds:
                continue

            node_vec = self._normalize(node.embedding)
            score = float(np.dot(normalized_query, node_vec))
            matches.append(RetrievalMatch(node=node, score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children)."""
        node = await self.get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = await self.get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children
        if child_depth > 0:
            await self._collect_children(
                node, depth=child_depth, accumulator=context, seen=set()
            )

        # Deduplicate
        unique = []
        seen_ids: set[str] = set()
        for item in context:
            if item.node_id in seen_ids:
                continue
            seen_ids.add(item.node_id)
            unique.append(item)

        return unique

    async def _collect_children(
        self,
        node: StoredNode,
        *,
        depth: int,
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        """Recursively collect children up to specified depth."""
        if depth <= 0:
            return

        for child_id in node.child_ids:
            if child_id in seen:
                continue

            seen.add(child_id)
            child = await self.get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            await self._collect_children(
                child,
                depth=depth - 1,
                accumulator=accumulator,
                seen=seen,
            )


class KVaultNodeStore(HierarchicalNodeStore):
    """SQLite-backed store using KohakuVault (key-value) + sqlite-vec (vectors).

    - Metadata stored in KohakuVault table
    - Embeddings indexed in sqlite-vec for fast similarity search
    - Both tables live in the same .db file
    - Supports optional full paragraph embeddings in separate vector table
    """

    META_KEY = "__kohakurag_meta__"

    def __init__(
        self,
        path: str | Path,
        *,
        table_prefix: str = "rag_nodes",
        dimensions: int | None = None,
        metric: str = "cosine",
        paragraph_search_mode: ParagraphSearchMode = "averaged",
    ) -> None:
        """Initialize or open existing datastore.

        Args:
            path: SQLite database file path
            table_prefix: Logical namespace for tables
            dimensions: Embedding dimension (auto-detected if None and DB exists)
            metric: Distance metric ("cosine" or "l2")
            paragraph_search_mode: Which embedding to use for paragraph search:
                - "averaged": Use sentence-averaged embeddings (default)
                - "full": Use full paragraph embeddings (requires "both" indexing mode)
                - "both": Search both tables and merge results by score
        """
        self._path = str(path)
        self._table_prefix = table_prefix
        self._paragraph_search_mode = paragraph_search_mode

        # Open key-value table for metadata
        self._kv = KVault(self._path, table=f"{table_prefix}_kv")
        self._kv.enable_auto_pack()

        # Validate or infer dimensions from stored metadata
        stored_meta = self._kv.get(self.META_KEY, None)
        inferred_dimensions: int | None = None

        if stored_meta is not None:
            inferred_dimensions = int(stored_meta.get("dimensions"))
            inferred_metric = stored_meta.get("metric", metric)
            if inferred_metric != metric:
                metric = inferred_metric  # Use stored metric

        # If dimensions not provided and not in metadata, try to infer from vector table
        if dimensions is None and inferred_dimensions is None:
            try:
                # Try opening existing vector table to get dimensions
                existing_vectors = VectorKVault(
                    self._path,
                    table=f"{table_prefix}_vec",
                    dimensions=1,  # Dummy, will be overwritten
                    metric=metric,
                )
                info = existing_vectors.info()
                if info.get("count", 0) > 0:
                    inferred_dimensions = int(info.get("dimensions", 0))
                    if inferred_dimensions > 0:
                        # Update metadata with inferred dimensions
                        self._kv[self.META_KEY] = {
                            "dimensions": inferred_dimensions,
                            "metric": metric,
                        }
            except Exception:
                pass

        # Determine final dimensions
        if dimensions is not None:
            final_dimensions = dimensions
        elif inferred_dimensions is not None:
            final_dimensions = inferred_dimensions
        else:
            raise ValueError(
                "Embedding dimension required for new store. Pass dimensions=... "
                "when creating the index."
            )

        self._dimensions = int(final_dimensions)

        # Check dimension consistency if both provided and stored
        if (
            dimensions is not None
            and inferred_dimensions is not None
            and dimensions != inferred_dimensions
        ):
            raise ValueError(
                f"Existing store was built with dimension {inferred_dimensions}, "
                f"but {dimensions} was requested."
            )

        # Store/update metadata
        self._kv[self.META_KEY] = {"dimensions": self._dimensions, "metric": metric}

        # Open vector table (main embeddings - averaged for paragraphs)
        self._vectors = VectorKVault(
            self._path,
            table=f"{table_prefix}_vec",
            dimensions=self._dimensions,
            metric=metric,
        )
        self._vectors.enable_auto_pack()
        self._metric = metric

        # Optional: full paragraph vector table (for "both" embedding mode)
        self._para_full_vectors: VectorKVault | None = None
        try:
            self._para_full_vectors = VectorKVault(
                self._path,
                table=f"{table_prefix}_para_full_vec",
                dimensions=self._dimensions,
                metric=metric,
            )
            self._para_full_vectors.enable_auto_pack()
        except Exception:
            # Table doesn't exist - that's fine, it's created on demand
            pass

        # Optional: image-only vector table (created by wattbot_build_image_index.py)
        self._image_vectors: VectorKVault | None = None
        try:
            self._image_vectors = VectorKVault(
                self._path,
                table=f"{table_prefix}_images_vec",
                dimensions=self._dimensions,
                metric=metric,
            )
            self._image_vectors.enable_auto_pack()
        except Exception:
            # Image-only table doesn't exist - that's fine, it's optional
            pass

        # Optional: BM25 (FTS5) text search table (created by wattbot_build_bm25_index.py)
        self._bm25: TextVault | None = None
        try:
            self._bm25 = TextVault(self._path, table=f"{table_prefix}_bm25")
            self._bm25.enable_auto_pack()
        except Exception:
            # BM25 table doesn't exist - that's fine, it's optional
            pass

        # Single-worker executor for thread-safe async SQLite operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _sync_upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Synchronous upsert logic (called via executor)."""
        for node in nodes:
            record = self._serialize_node(node)
            row_id = record.get("vector_row_id")

            # Insert or update vector
            if row_id is None:
                # New node: insert into vector table
                vector_row = self._vectors.insert(
                    node.embedding.astype(np.float32),
                    node.node_id,
                )
            else:
                # Existing node: update vector
                self._vectors.update(
                    row_id,
                    vector=node.embedding.astype(np.float32),
                    value=node.node_id,
                )
                vector_row = row_id

            # Store metadata with vector row reference
            record["vector_row_id"] = vector_row

            # Handle full paragraph embeddings if present in metadata
            if node.kind == NodeKind.PARAGRAPH and "full_embedding" in node.metadata:
                # Decode the full embedding from hex
                full_emb_hex = node.metadata["full_embedding"]
                full_emb = np.frombuffer(
                    bytes.fromhex(full_emb_hex), dtype=np.float32
                ).copy()

                # Ensure para_full_vectors table exists
                if self._para_full_vectors is None:
                    self._para_full_vectors = VectorKVault(
                        self._path,
                        table=f"{self._table_prefix}_para_full_vec",
                        dimensions=self._dimensions,
                        metric=self._metric,
                    )
                    self._para_full_vectors.enable_auto_pack()

                # Insert or update full paragraph embedding
                para_row_id = record.get("para_full_row_id")
                if para_row_id is None:
                    para_row = self._para_full_vectors.insert(full_emb, node.node_id)
                else:
                    self._para_full_vectors.update(
                        para_row_id, vector=full_emb, value=node.node_id
                    )
                    para_row = para_row_id

                record["para_full_row_id"] = para_row

            self._kv[node.node_id] = record

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Insert or update nodes (metadata + embeddings)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_upsert_nodes, nodes)

    def _sync_get_node(self, node_id: str) -> StoredNode:
        """Synchronous get_node logic (called via executor)."""
        record = self._kv[node_id]
        return self._deserialize_node(record)

    async def get_node(self, node_id: str) -> StoredNode:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_get_node, node_id)

    def _sync_search(
        self,
        query_vector: np.ndarray,
        k: int,
        kinds: set[NodeKind] | None,
        paragraph_search_mode: ParagraphSearchMode | None = None,
    ) -> list[RetrievalMatch]:
        """Synchronous search logic (called via executor).

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            kinds: Optional filter for node types
            paragraph_search_mode: Override instance-level paragraph search mode
        """
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D numpy array.")

        # Determine which search mode to use
        mode = paragraph_search_mode or self._paragraph_search_mode

        # Decide which vector tables to search based on mode
        search_main = True
        search_para_full = False
        skip_paragraphs_in_main = False  # For "full" mode only

        if mode == "full" and self._para_full_vectors is not None:
            # For "full" mode with paragraph filter, only search para_full table
            if kinds is not None and kinds == {NodeKind.PARAGRAPH}:
                search_main = False
                search_para_full = True
            elif kinds is None or NodeKind.PARAGRAPH in kinds:
                # Mixed search: search both tables, skip paragraphs in main
                search_para_full = True
                skip_paragraphs_in_main = True

        elif mode == "both" and self._para_full_vectors is not None:
            # For "both" mode: search both tables for paragraphs, merge by score
            if kinds is None or NodeKind.PARAGRAPH in kinds:
                search_para_full = True
                # Don't skip paragraphs in main - we want results from both tables

        all_matches: list[RetrievalMatch] = []
        seen_node_ids: set[str] = set()  # For deduplication in "both" mode

        # Search main vector table
        if search_main:
            results = self._vectors.search(
                query_vector.astype(np.float32),
                k=k,
            )

            for row_id, distance, node_id in results:
                node = self._sync_get_node(node_id)

                # Skip if node type doesn't match filter
                if kinds is not None and node.kind not in kinds:
                    continue

                # In full mode, skip paragraphs from main table (use para_full instead)
                if skip_paragraphs_in_main and node.kind == NodeKind.PARAGRAPH:
                    continue

                score = (
                    1.0 - float(distance)
                    if self._metric == "cosine"
                    else -float(distance)
                )
                all_matches.append(RetrievalMatch(node=node, score=score))
                seen_node_ids.add(node_id)

        # Search full paragraph vector table
        if search_para_full and self._para_full_vectors is not None:
            para_results = self._para_full_vectors.search(
                query_vector.astype(np.float32),
                k=k,
            )

            for row_id, distance, node_id in para_results:
                # In "both" mode, skip if already seen from main table
                # (keep the higher score by sorting later)
                if mode == "both" and node_id in seen_node_ids:
                    # Still add it - we'll deduplicate by keeping max score
                    pass

                try:
                    node = self._sync_get_node(node_id)
                    score = (
                        1.0 - float(distance)
                        if self._metric == "cosine"
                        else -float(distance)
                    )
                    all_matches.append(RetrievalMatch(node=node, score=score))
                except Exception:
                    continue

        # Sort all matches by score
        all_matches.sort(key=lambda item: item.score, reverse=True)

        # For "both" mode: deduplicate keeping highest score (already sorted)
        if mode == "both":
            deduped: list[RetrievalMatch] = []
            dedup_ids: set[str] = set()
            for match in all_matches:
                if match.node.node_id not in dedup_ids:
                    dedup_ids.add(match.node.node_id)
                    deduped.append(match)
            return deduped[:k]

        return all_matches[:k]

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
        paragraph_search_mode: ParagraphSearchMode | None = None,
    ) -> list[RetrievalMatch]:
        """Vector similarity search with optional node type filtering.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            kinds: Optional filter for node types
            paragraph_search_mode: Override instance-level paragraph search mode:
                - "averaged": Use sentence-averaged embeddings
                - "full": Use full paragraph embeddings (if available)
                - "both": Search both tables and merge by score (requires "both" indexing)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_search,
            query_vector,
            k,
            kinds,
            paragraph_search_mode,
        )

    def _sync_search_images(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> list[RetrievalMatch]:
        """Search image-only vector table (synchronous, called via executor)."""
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D numpy array.")

        if self._image_vectors is None:
            # No image-only table - return empty
            return []

        # Query image-only index
        results = self._image_vectors.search(
            query_vector.astype(np.float32),
            k=k,
        )

        # Fetch full nodes
        matches: list[RetrievalMatch] = []
        for row_id, distance, node_id in results:
            try:
                node = self._sync_get_node(node_id)

                # Convert distance to similarity score
                score = (
                    1.0 - float(distance)
                    if self._metric == "cosine"
                    else -float(distance)
                )
                matches.append(RetrievalMatch(node=node, score=score))
            except Exception:
                continue

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def search_images(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
    ) -> list[RetrievalMatch]:
        """Search ONLY image caption nodes using dedicated image vector table.

        This requires the image-only index to be built first via
        wattbot_build_image_index.py. If the table doesn't exist, returns empty list.

        Args:
            query_vector: Query embedding
            k: Number of image results to return

        Returns:
            List of image node matches (sorted by score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_search_images, query_vector, k
        )

    def has_image_index(self) -> bool:
        """Check if image-only vector table exists."""
        return self._image_vectors is not None

    def has_full_paragraph_index(self) -> bool:
        """Check if full paragraph embedding vector table exists."""
        return self._para_full_vectors is not None

    def set_paragraph_search_mode(self, mode: ParagraphSearchMode) -> None:
        """Set the default paragraph search mode at runtime.

        Args:
            mode: "averaged" or "full"
        """
        self._paragraph_search_mode = mode

    def has_bm25_index(self) -> bool:
        """Check if BM25 (FTS5) text search table exists."""
        return self._bm25 is not None

    def _sync_search_bm25(
        self,
        query: str,
        k: int,
        kinds: set[NodeKind] | None,
    ) -> list[RetrievalMatch]:
        """Synchronous BM25 search logic (called via executor).

        Args:
            query: Text query for FTS5 search
            k: Number of results to return
            kinds: Optional filter for node types

        Returns:
            List of retrieval matches sorted by BM25 score
        """
        if self._bm25 is None:
            return []

        # Search TextVault - returns list of (id, bm25_score, value)
        # where value is the node_id we stored
        results = self._bm25.search(query, k=k * 2)  # Fetch extra for filtering

        matches: list[RetrievalMatch] = []
        for row_id, bm25_score, node_id in results:
            try:
                node = self._sync_get_node(node_id)

                # Filter by node type if specified
                if kinds is not None and node.kind not in kinds:
                    continue

                # BM25 scores are negative (lower is better), convert to similarity
                # Typical BM25 scores range from -20 to 0, normalize to 0-1 range
                score = max(0.0, min(1.0, (bm25_score + 20) / 20))
                matches.append(RetrievalMatch(node=node, score=score))

                if len(matches) >= k:
                    break

            except Exception:
                continue

        return matches

    async def search_bm25(
        self,
        query: str,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        """BM25 (FTS5) text search for sparse retrieval.

        This requires the BM25 index to be built first via
        wattbot_build_bm25_index.py. If the table doesn't exist, returns empty list.

        Args:
            query: Text query for BM25 search
            k: Number of results to return
            kinds: Optional filter for node types

        Returns:
            List of retrieval matches (sorted by BM25 score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_search_bm25, query, k, kinds
        )

    def _sync_get_context(
        self,
        node_id: str,
        parent_depth: int,
        child_depth: int,
    ) -> list[StoredNode]:
        """Synchronous get_context logic (called via executor)."""
        node = self._sync_get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain (sentence → paragraph → section → document)
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = self._sync_get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children (paragraph → sentences)
        if child_depth > 0:
            self._sync_collect_children(
                node,
                depth=child_depth,
                accumulator=context,
                seen={node.node_id},
            )

        # Deduplicate in case parent/child overlap
        unique: list[StoredNode] = []
        seen_ids: set[str] = set()
        for item in context:
            if item.node_id in seen_ids:
                continue
            seen_ids.add(item.node_id)
            unique.append(item)

        return unique

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children).

        Example: For a matched sentence, get its paragraph and section too.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_get_context, node_id, parent_depth, child_depth
        )

    def _sync_collect_children(
        self,
        node: StoredNode,
        *,
        depth: int,
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        """Recursively collect children up to specified depth."""
        if depth <= 0:
            return

        for child_id in node.child_ids:
            if child_id in seen:
                continue

            seen.add(child_id)
            child = self._sync_get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            self._sync_collect_children(
                child,
                depth=depth - 1,
                accumulator=accumulator,
                seen=seen,
            )

    def _serialize_node(self, node: StoredNode) -> dict:
        """Convert StoredNode to KohakuVault-compatible dict."""
        record = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "kind": node.kind.value,
            "title": node.title,
            "text": node.text,
            "metadata": node.metadata,
            "child_ids": node.child_ids,
        }

        # Preserve vector row ID if updating existing node
        existing = None
        try:
            existing = self._kv[node.node_id]
        except KeyError:
            existing = None

        if existing:
            record["vector_row_id"] = existing.get("vector_row_id")

        return record

    def _deserialize_node(self, record: dict) -> StoredNode:
        """Reconstruct StoredNode from KohakuVault record + vector lookup."""
        vector_row = record.get("vector_row_id")
        if vector_row is None:
            raise ValueError(f"Node {record['node_id']} missing vector_row_id.")

        # Fetch embedding from vector table
        embedding, _ = self._vectors.get_by_id(vector_row)
        embedding_arr = np.array(embedding, dtype=np.float32, copy=True)

        return StoredNode(
            node_id=record["node_id"],
            parent_id=record.get("parent_id"),
            kind=NodeKind(record["kind"]),
            title=record["title"],
            text=record["text"],
            metadata=record.get("metadata", {}),
            embedding=embedding_arr,
            child_ids=list(record.get("child_ids", [])),
        )


async def matches_to_snippets(
    matches: Sequence[RetrievalMatch],
    store: HierarchicalNodeStore,
    *,
    parent_depth: int = 1,
    child_depth: int = 0,
    dedup: str = "none",
    no_overlap: bool = False,
) -> list[ContextSnippet]:
    """Convert retrieval matches into context snippets using hierarchical context.

    For each matched node, this helper pulls a small neighborhood of parents
    and children via ``get_context`` and flattens them into ``ContextSnippet``
    objects.  Deduplication is applied **after** expansion.

    Args:
        matches: List of retrieval matches
        store: Node store for context lookup
        parent_depth: How many parent levels to include
        child_depth: How many child levels to include
        dedup: Snippet-level deduplication mode:
            - ``"none"``: keep all snippets (including duplicates from expansion)
            - ``"node_id"``: remove duplicate snippets by ``node_id``
            - ``"tree"``: remove duplicates *and* remove snippets whose ancestor
              is also present (subsumes ``"node_id"``)
        no_overlap: **Deprecated** — equivalent to ``dedup="tree"``.

    Returns:
        List of context snippets
    """
    # Backward compatibility
    if no_overlap and dedup == "none":
        dedup = "tree"

    snippets: list[ContextSnippet] = []
    for rank, match in enumerate(matches, 1):
        nodes = await store.get_context(
            match.node.node_id,
            parent_depth=parent_depth,
            child_depth=child_depth,
        )
        for context_node in nodes:
            snippets.append(
                ContextSnippet(
                    node_id=context_node.node_id,
                    document_title=context_node.metadata.get(
                        "document_title", context_node.title
                    ),
                    text=context_node.text,
                    metadata=context_node.metadata,
                    rank=rank,
                    score=match.score,
                )
            )

    if dedup == "node_id":
        snippets = _dedup_snippets_by_node_id(snippets)
    elif dedup == "tree":
        snippets = _remove_overlapping_snippets(snippets)

    return snippets


def _dedup_snippets_by_node_id(
    snippets: list[ContextSnippet],
) -> list[ContextSnippet]:
    """Deduplicate snippets by ``node_id``, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[ContextSnippet] = []
    for s in snippets:
        if s.node_id not in seen:
            seen.add(s.node_id)
            result.append(s)
    return result


def _remove_overlapping_snippets(
    snippets: list[ContextSnippet],
) -> list[ContextSnippet]:
    """Remove overlapping snippets by keeping only the largest (parent) nodes.

    When a parent and child node both exist in snippets, the child's text is
    contained within the parent's text. This function removes such children
    to avoid redundant context.

    The hierarchy is determined by node_id structure:
    - Parent: "doc:sec1"
    - Child: "doc:sec1:p1" (starts with parent_id + ":")

    Args:
        snippets: List of context snippets (may contain overlaps)

    Returns:
        Filtered list with no overlapping nodes (keeps parents)
    """
    if not snippets:
        return snippets

    # Collect all unique node_ids
    all_node_ids = {s.node_id for s in snippets}

    # Find node_ids that have an ancestor in the set
    nodes_with_ancestor: set[str] = set()
    for node_id in all_node_ids:
        # Check if any other node_id is an ancestor of this one
        # Ancestor relationship: ancestor_id is a prefix and followed by ":"
        for other_id in all_node_ids:
            if other_id != node_id and node_id.startswith(other_id + ":"):
                # other_id is an ancestor of node_id
                nodes_with_ancestor.add(node_id)
                break

    # Filter out snippets whose node_id has an ancestor present
    # Also deduplicate by node_id (keep first occurrence)
    seen_ids: set[str] = set()
    filtered: list[ContextSnippet] = []
    for snippet in snippets:
        if snippet.node_id in nodes_with_ancestor:
            continue  # Skip - parent already covers this content
        if snippet.node_id in seen_ids:
            continue  # Skip - already included
        seen_ids.add(snippet.node_id)
        filtered.append(snippet)

    return filtered


class ImageStore:
    """Simple key-value store for compressed image blobs.

    Uses KohakuVault for storage in same database file as nodes.
    Key format: img:{doc_id}:p{page}:i{idx}
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "image_blobs",
    ) -> None:
        """Initialize or open existing image store.

        Args:
            path: SQLite database file path (same as node store)
            table: Table name for image storage
        """
        self._path = str(path)
        self._kv = KVault(self._path, table=table)
        self._kv.enable_auto_pack()

        # Single-worker executor for thread-safe async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @staticmethod
    def make_key(doc_id: str, page: int, index: int) -> str:
        """Generate storage key for an image.

        Args:
            doc_id: Document identifier
            page: Page number (1-indexed)
            index: Image index on page (1-indexed)

        Returns:
            Storage key string
        """
        return f"img:{doc_id}:p{page}:i{index}"

    def _sync_store(self, key: str, data: bytes) -> None:
        """Synchronous store logic (called via executor)."""
        self._kv[key] = data

    async def store_image(self, key: str, data: bytes) -> None:
        """Store image bytes.

        Args:
            key: Storage key (use make_key() to generate)
            data: Compressed image bytes (webp recommended)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_store, key, data)

    def _sync_get(self, key: str) -> bytes | None:
        """Synchronous get logic (called via executor)."""
        try:
            return self._kv[key]
        except KeyError:
            return None

    async def get_image(self, key: str) -> bytes | None:
        """Retrieve image bytes.

        Args:
            key: Storage key

        Returns:
            Image bytes or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_get, key)

    def _sync_exists(self, key: str) -> bool:
        """Synchronous exists check (called via executor)."""
        try:
            _ = self._kv[key]
            return True
        except KeyError:
            return False

    async def exists(self, key: str) -> bool:
        """Check if image exists.

        Args:
            key: Storage key

        Returns:
            True if image exists
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_exists, key)

    def _sync_list_keys(self, prefix: str) -> list[str]:
        """Synchronous list keys logic (called via executor)."""
        # KohakuVault doesn't have a native prefix scan, so we scan all keys
        # This is acceptable since image counts are typically small
        all_keys = []
        prefix_bytes = prefix.encode() if isinstance(prefix, str) else prefix
        for key in self._kv.keys():
            # Handle both str and bytes keys from KohakuVault
            if isinstance(key, bytes):
                if key.startswith(prefix_bytes):
                    all_keys.append(key.decode())
            else:
                if key.startswith(prefix):
                    all_keys.append(key)
        return all_keys

    async def list_images(self, prefix: str = "img:") -> list[str]:
        """List all image keys with given prefix.

        Args:
            prefix: Key prefix to filter (default: "img:")

        Returns:
            List of matching keys
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_list_keys, prefix)

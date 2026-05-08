"""Build a KohakuVault-backed index for WattBot documents.

Usage (CLI):
    python scripts/wattbot_build_index.py --docs-dir artifacts/docs --db artifacts/wattbot.db

Usage (KohakuEngine):
    kogine run scripts/wattbot_build_index.py --config configs/index_config.py
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Iterable

from kohakurag import (
    DocumentIndexer,
    dict_to_payload,
    text_to_payload,
)
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs"
db = "artifacts/wattbot.db"
table_prefix = "wattbot"
use_citations = False

# Embedding settings
embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
embedding_dim = None  # For JinaV4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # For JinaV4: "retrieval", "text-matching", "code"

# Paragraph embedding mode
# Options:
#   - "averaged": Paragraph embedding = average of sentence embeddings (default for backward compat)
#   - "full": Paragraph embedding = direct embedding of paragraph text
#   - "both": Store both averaged (main) and full (separate table) - allows runtime toggle
paragraph_embedding_mode = "both"


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["id"]] = row
    return records


def iter_structured_docs(docs_dir: Path) -> Iterable[dict]:
    if not docs_dir.exists():
        return []
    for json_path in sorted(docs_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        yield data


def iter_documents(
    docs_dir: Path | None,
    metadata: dict[str, dict[str, str]],
    use_citations: bool,
):
    if docs_dir and docs_dir.exists():
        for data in iter_structured_docs(docs_dir):
            yield dict_to_payload(data)
        return
    if use_citations:
        for doc_id, info in metadata.items():
            citation = info.get("citation") or info.get("title") or doc_id
            yield text_to_payload(
                document_id=doc_id,
                title=info.get("title", doc_id),
                text=citation,
                metadata={
                    "document_id": doc_id,
                    "document_title": info.get("title", doc_id),
                    "url": info.get("url"),
                    "type": info.get("type"),
                    "year": info.get("year"),
                },
            )
        return

    raise SystemExit(
        "Provide --docs-dir with structured JSON files or use --use-citations."
    )


# ============================================================================
# EMBEDDER FACTORY
# ============================================================================


def create_embedder():
    """Create embedder based on module-level config."""
    if embedding_model == "jinav4":
        print(f"Using JinaV4 embeddings (dim={embedding_dim}, task={embedding_task})")
        return JinaV4EmbeddingModel(
            truncate_dim=embedding_dim or 1024,
            task=embedding_task,
        )
    else:
        print("Using JinaV3 embeddings (default)")
        return JinaEmbeddingModel()


# ============================================================================
# MAIN INDEXING PIPELINE
# ============================================================================


async def main() -> None:
    """Build the hierarchical index from documents."""
    # Load documents to index
    metadata_records = load_metadata(Path(metadata))
    documents = list(iter_documents(Path(docs_dir), metadata_records, use_citations))
    total_docs = len(documents)

    if not total_docs:
        raise SystemExit("No documents found to index.")

    # Setup indexer and datastore
    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create embedder using factory
    embedder = create_embedder()
    indexer = DocumentIndexer(
        embedding_model=embedder,
        paragraph_embedding_mode=paragraph_embedding_mode,
    )
    print(f"Paragraph embedding mode: {paragraph_embedding_mode}")

    store: KVaultNodeStore | None = None  # Lazy init after first document
    total_nodes = 0

    # Index each document and upsert nodes
    for idx, payload in enumerate(documents, start=1):
        print(f"[{idx}/{total_docs}] indexing {payload.document_id}...", flush=True)

        # Build hierarchical tree and compute embeddings
        nodes = await indexer.index(payload)
        if not nodes:
            print("  -> no nodes generated, skipping.", flush=True)
            continue

        # Initialize store on first document (infer dimensions)
        if store is None:
            store = KVaultNodeStore(
                db_path,
                table_prefix=table_prefix,
                dimensions=nodes[0].embedding.shape[0],
            )

        # Persist nodes to SQLite + sqlite-vec
        await store.upsert_nodes(nodes)
        total_nodes += len(nodes)
        print(
            f"  -> added {len(nodes)} nodes (running total {total_nodes})", flush=True
        )

    print(f"Indexed {len(documents)} documents with {total_nodes} nodes into {db_path}")


if __name__ == "__main__":
    asyncio.run(main())

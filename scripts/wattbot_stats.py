"""Report basic statistics for a KohakuVault-backed index.

Usage (CLI):
    python scripts/wattbot_stats.py --db artifacts/wattbot.db

Usage (KohakuEngine):
    kogine run scripts/wattbot_stats.py --config configs/stats_config.py
"""

import asyncio
from collections import Counter, defaultdict
from pathlib import Path

from kohakurag import NodeKind
from kohakurag.datastore import ImageStore, KVaultNodeStore

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"


async def main() -> None:
    store = KVaultNodeStore(Path(db), table_prefix=table_prefix, dimensions=None)
    counters = Counter()
    paragraph_per_doc = defaultdict(int)
    sentence_per_doc = defaultdict(int)
    attachment_count = 0
    image_paragraph_count = 0  # Paragraphs with image captions

    info = store._vectors.info()  # type: ignore[attr-defined]
    total_entries = info.get("count", 0)

    for row_id in range(1, total_entries + 1):
        try:
            _, node_id = store._vectors.get_by_id(row_id)  # type: ignore[attr-defined]
            if isinstance(node_id, bytes):
                node_id = node_id.decode()
            node = await store.get_node(node_id)
        except Exception:
            continue
        counters[node.kind] += 1
        doc_id = node.metadata.get("document_id", node.node_id.split(":")[0])
        if node.kind == NodeKind.PARAGRAPH:
            paragraph_per_doc[doc_id] += 1
            # Check if this paragraph is an image caption
            if node.metadata.get("attachment_type") == "image":
                image_paragraph_count += 1
        if node.kind == NodeKind.SENTENCE:
            sentence_per_doc[doc_id] += 1
        if node.kind == NodeKind.ATTACHMENT:
            attachment_count += 1

    total_docs = counters[NodeKind.DOCUMENT]
    print(f"Database: {db}")
    print(f"Total vectors: {total_entries}")
    print(f"Total documents: {total_docs}")
    print(f"Total nodes: {sum(counters.values())}")
    for kind in NodeKind:
        print(f"  {kind.value:>10}: {counters[kind]}")

    if total_docs:
        avg_paragraphs = sum(paragraph_per_doc.values()) / total_docs
        avg_sentences = sum(sentence_per_doc.values()) / total_docs
        print(f"\nAverage paragraphs per document: {avg_paragraphs:.2f}")
        print(f"Average sentences per document: {avg_sentences:.2f}")
    print(f"Attachment nodes: {attachment_count}")

    # Check for image data
    print("\nImage Statistics:")
    print(f"  Image caption nodes: {image_paragraph_count}")

    # Try to access image store to count compressed images
    try:
        image_store = ImageStore(db, table="image_blobs")
        image_keys = await image_store.list_images(prefix="img:")
        print(f"  Compressed image blobs: {len(image_keys)}")
    except Exception as e:
        print(f"  Compressed image blobs: 0 (error: {e})")


if __name__ == "__main__":
    asyncio.run(main())

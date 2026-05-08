"""Build a BM25 (FTS5) index for WattBot documents using KohakuVault TextVault.

This script creates a sparse text search index alongside the existing dense vector index.
It reads all nodes from an existing KVaultNodeStore and indexes sentence/paragraph text
in a TextVault for BM25-based retrieval.

The BM25 index enables hybrid search (dense + sparse) by providing additional context
that may be missed by pure semantic search.

Prerequisites:
    - An existing dense index must be built first via wattbot_build_index.py
    - The dense index DB will be augmented with BM25 tables

Usage (CLI):
    python scripts/wattbot_build_bm25_index.py --db artifacts/wattbot.db

Usage (KohakuEngine):
    kogine run scripts/wattbot_build_bm25_index.py --config configs/bm25_index_config.py

Migration (existing index without BM25):
    Simply run this script pointing to your existing .db file. It will add the BM25
    table without affecting the existing vector index.
"""

import asyncio
from pathlib import Path

from kohakuvault import TextVault

from kohakurag.datastore import KVaultNodeStore
from kohakurag.types import NodeKind

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"

# Node types to index (sentence and paragraph match dense search behavior)
index_node_kinds = ["sentence", "paragraph"]


# ============================================================================
# MAIN INDEXING PIPELINE
# ============================================================================


def get_bm25_table_name(prefix: str) -> str:
    """Generate BM25 table name from prefix."""
    return f"{prefix}_bm25"


async def main() -> None:
    """Build BM25 index from existing dense index."""
    db_path = Path(db)

    if not db_path.exists():
        raise SystemExit(
            f"Database {db_path} does not exist. "
            "Build the dense index first with wattbot_build_index.py"
        )

    print(f"Building BM25 index for {db_path}")
    print(f"Table prefix: {table_prefix}")
    print(f"Node types to index: {index_node_kinds}")

    # Open existing node store (read-only for nodes)
    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=None,  # Will be inferred from existing index
    )

    # Create TextVault for BM25 index (single column for text)
    bm25_table = get_bm25_table_name(table_prefix)
    tv = TextVault(str(db_path), table=bm25_table)
    tv.enable_auto_pack()

    print(f"BM25 table: {bm25_table}")

    # Convert string kinds to NodeKind enum
    target_kinds = {NodeKind(k) for k in index_node_kinds}

    # Iterate through all nodes in KVault metadata table
    kv = store._kv
    total_indexed = 0
    skipped = 0

    # Get all keys with a large limit (default is 10000 which is too small)
    # Use limit=1_000_000 to get all keys in large indexes
    all_keys = list(kv.keys(limit=1_000_000))
    print(f"\nIndexing nodes... (found {len(all_keys)} keys in KVault)")

    for key in all_keys:
        # Normalize key to string for comparison
        key_str = key.decode() if isinstance(key, bytes) else key

        # Skip metadata key
        if key_str == KVaultNodeStore.META_KEY:
            continue

        try:
            record = kv[key]
            node_kind = NodeKind(record["kind"])

            # Skip if not in target kinds
            if node_kind not in target_kinds:
                skipped += 1
                continue

            node_id = record["node_id"]
            text = record.get("text", "").strip()

            # Skip empty text
            if not text:
                skipped += 1
                continue

            # Insert into TextVault: text as key (indexed), node_id as value (payload)
            tv.insert(text, node_id)
            total_indexed += 1

            if total_indexed % 1000 == 0:
                print(f"  Indexed {total_indexed} nodes...", flush=True)

        except Exception as e:
            print(f"  Warning: Failed to process key {key}: {e}")
            skipped += 1
            continue

    print("\nBM25 Index Complete!")
    print(f"  Total indexed: {total_indexed}")
    print(f"  Skipped: {skipped}")
    print(f"  Database: {db_path}")
    print(f"  Table: {bm25_table}")


if __name__ == "__main__":
    asyncio.run(main())

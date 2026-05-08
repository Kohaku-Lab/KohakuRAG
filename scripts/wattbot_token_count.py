"""Approximate token count for all sentence nodes in the database.

Usage (CLI):
    python scripts/wattbot_token_count.py --db artifacts/wattbot.db

Usage (KohakuEngine):
    kogine run scripts/wattbot_token_count.py --config configs/stats_config.py
"""

import asyncio
from pathlib import Path

from transformers import AutoTokenizer

from kohakurag import NodeKind
from kohakurag.datastore import KVaultNodeStore

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
tokenizer_name = "jinaai/jina-embeddings-v3"
batch_size = 256


async def main() -> None:
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Opening database: {db}")
    store = KVaultNodeStore(Path(db), table_prefix=table_prefix, dimensions=None)

    info = store._vectors.info()
    total_entries = info.get("count", 0)
    print(f"Total vectors in database: {total_entries}")

    # Collect all sentence texts
    sentence_texts = []
    sentence_count = 0

    print("Collecting sentence nodes...")
    for row_id in range(1, total_entries + 1):
        try:
            _, node_id = store._vectors.get_by_id(row_id)
            if isinstance(node_id, bytes):
                node_id = node_id.decode()
            node = await store.get_node(node_id)
        except Exception:
            continue

        if node.kind == NodeKind.SENTENCE:
            sentence_texts.append(node.text)
            sentence_count += 1

    print(f"Found {sentence_count} sentence nodes")

    # Batch tokenize
    print(f"Tokenizing in batches of {batch_size}...")
    total_tokens = 0

    for i in range(0, len(sentence_texts), batch_size):
        batch = sentence_texts[i : i + batch_size]
        # Tokenize without special tokens (no BOS/EOS)
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True,
        )
        batch_tokens = sum(encoded["length"])
        total_tokens += batch_tokens

        if (i // batch_size) % 10 == 0:
            print(
                f"  Processed {min(i + batch_size, len(sentence_texts))}/{len(sentence_texts)} sentences..."
            )

    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Total sentences: {sentence_count}")
    print(f"Total tokens (no special tokens): {total_tokens}")
    print(f"Average tokens per sentence: {total_tokens / sentence_count:.2f}")
    print(f"Approximate corpus size: {total_tokens / 1000:.1f}K tokens")


if __name__ == "__main__":
    asyncio.run(main())

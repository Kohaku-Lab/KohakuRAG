"""Build a separate image-only vector index for dedicated image retrieval.

This script is OPTIONAL and works on top of an existing image-enhanced database.
It creates a separate vector table containing ONLY image caption embeddings,
enabling fast top-k image retrieval alongside text retrieval.

Workflow:
1. Text-only:              wattbot_text_only.db (no images)
2. Text + images in tree:  wattbot_with_images.db (images as part of main hierarchy)
3. + Image-only retrieval: Run THIS script on wattbot_with_images.db

Usage (CLI):
    python scripts/wattbot_build_image_index.py \\
        --db artifacts/wattbot_with_images.db \\
        --table-prefix wattbot_img

Usage (KohakuEngine):
    kogine run scripts/wattbot_build_image_index.py --config configs/image_index_config.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
from kohakuvault import VectorKVault

from kohakurag import NodeKind
from kohakurag.datastore import KVaultNodeStore, ImageStore
from kohakurag.embeddings import JinaEmbeddingModel, JinaV4EmbeddingModel

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
image_table = None  # Default: {prefix}_images_vec

# Embedding settings (for direct image embedding with JinaV4)
embedding_model = "jina"  # Options: "jina" (use captions), "jinav4" (direct embedding)
embedding_dim = None  # For JinaV4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # For JinaV4
embed_images_directly = False  # True = use JinaV4.encode_image(), False = use captions


async def main() -> None:
    """Build image-only vector index from existing node store."""
    # Derive image table name
    actual_image_table = image_table or f"{table_prefix}_images_vec"
    db_path = Path(db)

    print("=" * 60)
    print("Building Image-Only Vector Index")
    print("=" * 60)
    print(f"Database:     {db_path}")
    print(f"Main prefix:  {table_prefix}")
    print(f"Image table:  {actual_image_table}")
    print("=" * 60)

    if not db_path.exists():
        print(f"\nERROR: Database not found: {db_path}")
        print("Run wattbot_build_index.py first to create the main index.")
        sys.exit(1)

    # Load existing node store
    print("\nLoading existing node store...")
    try:
        store = KVaultNodeStore(
            db_path,
            table_prefix=table_prefix,
            dimensions=None,  # Auto-detect
        )
    except Exception as e:
        print(f"ERROR: Failed to load store: {e}")
        sys.exit(1)

    print(f"✓ Loaded (dimensions: {store.dimensions})")

    # Scan for image nodes
    print("\nScanning for image caption nodes...")

    info = store._vectors.info()
    total_entries = info.get("count", 0)

    image_nodes = []

    for row_id in range(1, total_entries + 1):
        try:
            _, node_id = store._vectors.get_by_id(row_id)
            if isinstance(node_id, bytes):
                node_id = node_id.decode()

            node = await store.get_node(node_id)

            # Check if this is an image caption node
            if node.metadata.get("attachment_type") == "image":
                image_nodes.append(node)

        except Exception:
            continue

    print(f"✓ Found {len(image_nodes)} image caption nodes")

    if not image_nodes:
        print("\n⚠️  No image nodes found in database!")
        print("Make sure you:")
        print("  1. Ran wattbot_add_image_captions.py to add captions")
        print("  2. Ran wattbot_build_index.py on docs_with_images/")
        sys.exit(0)

    # Determine embedding strategy
    use_direct_embedding = embed_images_directly and embedding_model == "jinav4"

    if use_direct_embedding:
        print("\n🎨 Using JinaV4 direct image embedding...")
        print(f"   Dimension: {embedding_dim or 1024}")
        print(f"   Task: {embedding_task}")

        # Create JinaV4 embedder
        embedder = JinaV4EmbeddingModel(
            truncate_dim=embedding_dim or 1024,
            task=embedding_task,
        )

        # Load ImageStore to retrieve image bytes
        print("📦 Loading ImageStore...")
        image_store = ImageStore(db_path, table="image_blobs")

        # Extract image bytes from ImageStore using storage_key
        image_bytes_list = []
        valid_nodes = []

        for node in image_nodes:
            # Get storage key from metadata (added by wattbot_add_image_captions.py)
            storage_key = node.metadata.get("image_storage_key")
            if not storage_key:
                continue

            try:
                # Load image bytes from ImageStore (method is get_image, not load_image)
                img_bytes = await image_store.get_image(storage_key)
                if img_bytes:
                    image_bytes_list.append(img_bytes)
                    valid_nodes.append(node)
            except Exception:
                continue

        if not valid_nodes:
            print("\n⚠️  No images loaded from ImageStore!")
            print("Make sure you ran wattbot_add_image_captions.py first.")
            sys.exit(1)

        print(f"✓ Loaded {len(valid_nodes)} images from ImageStore")

        # Embed images directly using JinaV4
        print(f"\nEmbedding {len(image_bytes_list)} images with JinaV4...")
        embeddings = await embedder.embed_images(image_bytes_list)
        print(f"✓ Generated embeddings: shape={embeddings.shape}")

        # Use JinaV4 dimensions for image table
        actual_dimensions = embedder.dimension
        image_nodes_to_index = valid_nodes

        # Assign new embeddings to nodes (for insertion)
        for i, node in enumerate(valid_nodes):
            node.embedding = embeddings[i]

    else:
        print("\n📝 Using caption-based embeddings (from text index)...")
        actual_dimensions = store.dimensions
        embeddings = np.vstack([node.embedding for node in image_nodes])
        image_nodes_to_index = image_nodes

    # Create image-only vector table
    print(f"\nCreating image-only vector table: {actual_image_table}")
    print(f"Dimensions: {actual_dimensions}")

    try:
        image_vec_store = VectorKVault(
            str(db_path),
            table=actual_image_table,
            dimensions=actual_dimensions,
            metric="cosine",
        )
        image_vec_store.enable_auto_pack()
    except Exception as e:
        print(f"ERROR: Failed to create image vector table: {e}")
        sys.exit(1)

    # Insert image embeddings
    print(f"Inserting {len(image_nodes_to_index)} image embeddings...")

    inserted = 0
    for node in image_nodes_to_index:
        try:
            image_vec_store.insert(
                node.embedding.astype(np.float32),
                node.node_id,  # Store node_id as value for lookup
            )
            inserted += 1
        except Exception as e:
            print(f"  ⚠️  Failed to insert {node.node_id}: {e}")

    print(f"✓ Inserted {inserted}/{len(image_nodes_to_index)} image embeddings")

    # Summary
    print("\n" + "=" * 60)
    print("Image-Only Index Complete")
    print("=" * 60)
    print(f"Image embeddings: {inserted}")
    print(f"Table: {actual_image_table}")
    print(f"Dimensions: {actual_dimensions}")
    if use_direct_embedding:
        print("Method: JinaV4 direct image embedding ✨")
    else:
        print("Method: Caption-based (JinaV3 or existing)")
    print("\nNow you can use --top-k-images flag with wattbot_answer.py")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    asyncio.run(main())

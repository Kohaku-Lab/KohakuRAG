"""Visualize embedding distributions using dimensionality reduction.

Supports PCA, t-SNE, and UMAP for visualizing the distribution of different
node types (document, section, paragraph, sentence, image) in embedding space.

Usage (CLI):
    python scripts/visualize_embeddings.py

Usage (KohakuEngine):
    kogine run scripts/visualize_embeddings.py --config configs/visualize.py

Configuration:
    db: Path to the database file
    table_prefix: Table prefix for the node store
    output: Output image path (default: artifacts/embedding_viz.png)
    method: Dimensionality reduction method ("pca", "tsne", "umap")
    show_types: List of node types to display (e.g., ["paragraph", "sentence"])
                Options: "document", "section", "paragraph", "paragraph_full",
                         "paragraph_avg", "sentence", "image"
                If None or empty, shows all available types.

    Sampling modes (mutually exclusive):
    - max_samples: Maximum samples per node type (flat sampling)
    - hierarchical_sampling: If True, sample based on tree structure:
        - samples_per_document: How many sections per document
        - samples_per_section: How many paragraphs per section
        - samples_per_paragraph: How many sentences per paragraph
        All documents are included.

    perplexity: t-SNE perplexity parameter (default: 30)
    n_neighbors: UMAP n_neighbors parameter (default: 15)
    min_dist: UMAP min_dist parameter (default: 0.1)
    random_state: Random seed for reproducibility (default: 42)
"""

import asyncio
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from kohakurag import NodeKind
from kohakurag.datastore import KVaultNodeStore

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
output = "artifacts/embedding_viz.png"

# Dimensionality reduction method: "pca", "tsne", "umap"
method = "pca"

# Node types to display (None or [] = all types)
# Options: "document", "section", "paragraph", "paragraph_full", "paragraph_avg",
#          "sentence", "image"
# "paragraph" shows both full and avg if available
# "paragraph_full" shows only full embeddings
# "paragraph_avg" shows only averaged embeddings
show_types: list[str] | None = None

# Flat sampling: Maximum samples per node type (None = all)
max_samples: int | None = 500

# Hierarchical sampling: sample based on tree structure
# Set hierarchical_sampling = True to enable
hierarchical_sampling = False
samples_per_document: int | None = 3  # Sections per document (None = all)
samples_per_section: int | None = 3  # Paragraphs per section (None = all)
samples_per_paragraph: int | None = 3  # Sentences per paragraph (None = all)

# t-SNE parameters
perplexity = 30

# UMAP parameters
n_neighbors = 15
min_dist = 0.1
n_jobs = 32  # Parallel jobs for UMAP (-1 for all cores)

# Random seed (None for UMAP to enable parallelism)
random_state = 42

# Figure size
figsize = (12, 10)

# Color mode: "type" (color by node type) or "document" (color by document)
# When "document", shape indicates node type
color_mode = "type"

# Axis limit percentile: clip outliers by using percentile-based axis limits
# e.g., 99 means use 1st-99th percentile range (clips top/bottom 1%)
# Set to None or 100 to disable (use full range)
axis_percentile: float | None = 99


# ============================================================================
# COLORS FOR NODE TYPES
# ============================================================================

TYPE_COLORS = {
    "document": "#e41a1c",  # Red
    "section": "#377eb8",  # Blue
    "paragraph_avg": "#4daf4a",  # Green
    "paragraph_full": "#984ea3",  # Purple
    "sentence": "#ff7f00",  # Orange
    "image": "#a65628",  # Brown
}

TYPE_MARKERS = {
    "document": "p",  # Pentagon
    "section": "^",  # Triangle up
    "paragraph_avg": "o",  # Circle
    "paragraph_full": "D",  # Diamond
    "sentence": "s",  # Square
    "image": "*",  # Star
}


def _expand_show_types(
    show_types: list[str] | None,
    has_full_paragraph: bool,
) -> set[str]:
    """Expand show_types to actual type names."""
    all_types = {"document", "section", "paragraph_avg", "sentence", "image"}
    if has_full_paragraph:
        all_types.add("paragraph_full")

    if show_types:
        expanded_types = set()
        for t in show_types:
            if t == "paragraph":
                expanded_types.add("paragraph_avg")
                if has_full_paragraph:
                    expanded_types.add("paragraph_full")
            else:
                expanded_types.add(t)
        return expanded_types & all_types
    return all_types


async def collect_embeddings_hierarchical(
    store: KVaultNodeStore,
    show_types: list[str] | None,
    samples_per_document: int | None,
    samples_per_section: int | None,
    samples_per_paragraph: int | None,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Collect embeddings using hierarchical tree-based sampling.

    Samples based on document structure:
    - All documents are included
    - N sections sampled per document
    - M paragraphs sampled per section
    - K sentences sampled per paragraph

    Args:
        store: The node store
        show_types: List of types to collect (None = all)
        samples_per_document: Sections per document (None = all)
        samples_per_section: Paragraphs per section (None = all)
        samples_per_paragraph: Sentences per paragraph (None = all)
        seed: Random seed

    Returns:
        Tuple of:
        - Dict mapping type name to (N, D) embedding array
        - Dict mapping type name to list of doc_ids (same order as embeddings)
    """
    random.seed(seed)
    types_to_collect = _expand_show_types(show_types, store.has_full_paragraph_index())

    embeddings: dict[str, list[np.ndarray]] = {t: [] for t in types_to_collect}
    doc_ids_per_type: dict[str, list[str]] = {t: [] for t in types_to_collect}
    node_ids_collected: dict[str, set[str]] = {t: set() for t in types_to_collect}
    # Track which doc each paragraph belongs to for full paragraph collection
    para_to_doc: dict[str, str] = {}

    # Get total count from vector table
    info = store._vectors.info()
    total_entries = info.get("count", 0)

    print(f"Scanning {total_entries} vectors for hierarchical sampling...")
    print(f"  Sections per document: {samples_per_document or 'all'}")
    print(f"  Paragraphs per section: {samples_per_section or 'all'}")
    print(f"  Sentences per paragraph: {samples_per_paragraph or 'all'}")

    # First pass: collect all nodes and build tree structure
    documents: dict[str, dict] = (
        {}
    )  # doc_id -> {node, sections: {sec_id -> {node, paragraphs: ...}}}

    for row_id in range(1, total_entries + 1):
        try:
            vector, node_id = store._vectors.get_by_id(row_id)
            if isinstance(node_id, bytes):
                node_id = node_id.decode()

            node = await store.get_node(node_id)
            vector_arr = np.array(vector, dtype=np.float32)

            if node.kind == NodeKind.DOCUMENT:
                if node_id not in documents:
                    documents[node_id] = {
                        "node": node,
                        "vector": vector_arr,
                        "sections": {},
                    }
                else:
                    documents[node_id]["node"] = node
                    documents[node_id]["vector"] = vector_arr

            elif node.kind == NodeKind.SECTION:
                doc_id = node.parent_id
                if doc_id not in documents:
                    documents[doc_id] = {"node": None, "vector": None, "sections": {}}
                if node_id not in documents[doc_id]["sections"]:
                    documents[doc_id]["sections"][node_id] = {
                        "node": node,
                        "vector": vector_arr,
                        "paragraphs": {},
                    }
                else:
                    documents[doc_id]["sections"][node_id]["node"] = node
                    documents[doc_id]["sections"][node_id]["vector"] = vector_arr

            elif node.kind == NodeKind.PARAGRAPH:
                # Parse node_id to get section: doc:sec:p
                parts = node_id.split(":")
                if len(parts) >= 2:
                    doc_id = parts[0]
                    sec_id = f"{parts[0]}:{parts[1]}"
                else:
                    continue

                if doc_id not in documents:
                    documents[doc_id] = {"node": None, "vector": None, "sections": {}}
                if sec_id not in documents[doc_id]["sections"]:
                    documents[doc_id]["sections"][sec_id] = {
                        "node": None,
                        "vector": None,
                        "paragraphs": {},
                    }
                if node_id not in documents[doc_id]["sections"][sec_id]["paragraphs"]:
                    documents[doc_id]["sections"][sec_id]["paragraphs"][node_id] = {
                        "node": node,
                        "vector": vector_arr,
                        "sentences": [],
                    }
                else:
                    documents[doc_id]["sections"][sec_id]["paragraphs"][node_id][
                        "node"
                    ] = node
                    documents[doc_id]["sections"][sec_id]["paragraphs"][node_id][
                        "vector"
                    ] = vector_arr

            elif node.kind == NodeKind.SENTENCE:
                # Parse node_id: doc:sec:p:s
                parts = node_id.split(":")
                if len(parts) >= 3:
                    doc_id = parts[0]
                    sec_id = f"{parts[0]}:{parts[1]}"
                    para_id = f"{parts[0]}:{parts[1]}:{parts[2]}"
                else:
                    continue

                if doc_id not in documents:
                    documents[doc_id] = {"node": None, "vector": None, "sections": {}}
                if sec_id not in documents[doc_id]["sections"]:
                    documents[doc_id]["sections"][sec_id] = {
                        "node": None,
                        "vector": None,
                        "paragraphs": {},
                    }
                if para_id not in documents[doc_id]["sections"][sec_id]["paragraphs"]:
                    documents[doc_id]["sections"][sec_id]["paragraphs"][para_id] = {
                        "node": None,
                        "vector": None,
                        "sentences": [],
                    }
                documents[doc_id]["sections"][sec_id]["paragraphs"][para_id][
                    "sentences"
                ].append({"node": node, "vector": vector_arr, "node_id": node_id})

        except Exception:
            continue

    # Second pass: sample from tree
    for doc_id, doc_data in documents.items():
        # Add document
        if "document" in types_to_collect and doc_data["vector"] is not None:
            embeddings["document"].append(doc_data["vector"])
            doc_ids_per_type["document"].append(doc_id)

        # Sample sections
        section_ids = list(doc_data["sections"].keys())
        if samples_per_document is not None and len(section_ids) > samples_per_document:
            section_ids = random.sample(section_ids, samples_per_document)

        for sec_id in section_ids:
            sec_data = doc_data["sections"][sec_id]

            # Add section
            if "section" in types_to_collect and sec_data["vector"] is not None:
                embeddings["section"].append(sec_data["vector"])
                doc_ids_per_type["section"].append(doc_id)

            # Sample paragraphs
            para_ids = list(sec_data["paragraphs"].keys())
            if samples_per_section is not None and len(para_ids) > samples_per_section:
                para_ids = random.sample(para_ids, samples_per_section)

            for para_id in para_ids:
                para_data = sec_data["paragraphs"][para_id]

                # Add paragraph (check if image)
                if para_data["node"] is not None and para_data["vector"] is not None:
                    is_image = (
                        para_data["node"].metadata.get("attachment_type") == "image"
                    )
                    if is_image and "image" in types_to_collect:
                        embeddings["image"].append(para_data["vector"])
                        doc_ids_per_type["image"].append(doc_id)
                    elif not is_image and "paragraph_avg" in types_to_collect:
                        embeddings["paragraph_avg"].append(para_data["vector"])
                        doc_ids_per_type["paragraph_avg"].append(doc_id)
                        node_ids_collected["paragraph_avg"].add(para_id)
                        para_to_doc[para_id] = doc_id

                # Sample sentences
                sentences = para_data["sentences"]
                if (
                    samples_per_paragraph is not None
                    and len(sentences) > samples_per_paragraph
                ):
                    sentences = random.sample(sentences, samples_per_paragraph)

                for sent_data in sentences:
                    if (
                        "sentence" in types_to_collect
                        and sent_data["vector"] is not None
                    ):
                        embeddings["sentence"].append(sent_data["vector"])
                        doc_ids_per_type["sentence"].append(doc_id)

    # Collect full paragraph embeddings for sampled paragraphs
    if "paragraph_full" in types_to_collect and store.has_full_paragraph_index():
        para_info = store._para_full_vectors.info()
        para_count = para_info.get("count", 0)
        print(
            f"Collecting full paragraph embeddings for {len(node_ids_collected.get('paragraph_avg', set()))} sampled paragraphs..."
        )

        for row_id in range(1, para_count + 1):
            try:
                vector, node_id = store._para_full_vectors.get_by_id(row_id)
                if isinstance(node_id, bytes):
                    node_id = node_id.decode()
                # Only include if the paragraph was sampled
                if node_id in node_ids_collected.get("paragraph_avg", set()):
                    embeddings["paragraph_full"].append(
                        np.array(vector, dtype=np.float32)
                    )
                    doc_ids_per_type["paragraph_full"].append(para_to_doc[node_id])
            except Exception:
                continue

    # Convert to numpy arrays
    result = {}
    result_doc_ids: dict[str, list[str]] = {}
    for type_name, vecs in embeddings.items():
        if vecs:
            result[type_name] = np.vstack(vecs)
            result_doc_ids[type_name] = doc_ids_per_type[type_name]
            print(f"  {type_name}: {len(vecs)} embeddings")

    return result, result_doc_ids


async def collect_embeddings_flat(
    store: KVaultNodeStore,
    show_types: list[str] | None,
    max_samples: int | None,
) -> tuple[dict[str, np.ndarray], None]:
    """Collect embeddings using flat sampling (max per type).

    Args:
        store: The node store
        show_types: List of types to collect (None = all)
        max_samples: Max samples per type

    Returns:
        Tuple of:
        - Dict mapping type name to (N, D) embedding array
        - None (no doc_id tracking in flat mode)
    """
    types_to_collect = _expand_show_types(show_types, store.has_full_paragraph_index())

    # Collect embeddings
    embeddings: dict[str, list[np.ndarray]] = {t: [] for t in types_to_collect}

    # Get total count from vector table
    info = store._vectors.info()
    total_entries = info.get("count", 0)

    print(
        f"Scanning {total_entries} vectors (flat sampling, max {max_samples or 'unlimited'} per type)..."
    )

    # Collect from main vector table
    for row_id in range(1, total_entries + 1):
        try:
            vector, node_id = store._vectors.get_by_id(row_id)
            if isinstance(node_id, bytes):
                node_id = node_id.decode()

            node = await store.get_node(node_id)
            vector_arr = np.array(vector, dtype=np.float32)

            # Determine type
            if node.kind == NodeKind.DOCUMENT:
                type_name = "document"
            elif node.kind == NodeKind.SECTION:
                type_name = "section"
            elif node.kind == NodeKind.PARAGRAPH:
                # Check if it's an image caption
                if node.metadata.get("attachment_type") == "image":
                    type_name = "image"
                else:
                    type_name = "paragraph_avg"
            elif node.kind == NodeKind.SENTENCE:
                type_name = "sentence"
            else:
                continue

            if type_name in types_to_collect:
                if max_samples is None or len(embeddings[type_name]) < max_samples:
                    embeddings[type_name].append(vector_arr)

        except Exception:
            continue

    # Collect full paragraph embeddings if requested
    if "paragraph_full" in types_to_collect and store.has_full_paragraph_index():
        para_info = store._para_full_vectors.info()
        para_count = para_info.get("count", 0)
        print(f"Scanning {para_count} full paragraph vectors...")

        for row_id in range(1, para_count + 1):
            try:
                vector, node_id = store._para_full_vectors.get_by_id(row_id)
                vector_arr = np.array(vector, dtype=np.float32)

                if (
                    max_samples is None
                    or len(embeddings["paragraph_full"]) < max_samples
                ):
                    embeddings["paragraph_full"].append(vector_arr)
            except Exception:
                continue

    # Convert to numpy arrays
    result = {}
    for type_name, vecs in embeddings.items():
        if vecs:
            result[type_name] = np.vstack(vecs)
            print(f"  {type_name}: {len(vecs)} embeddings")

    return result, None


def reduce_dimensions(
    embeddings: dict[str, np.ndarray],
    method: str,
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 42,
    n_jobs: int = 32,
) -> dict[str, np.ndarray]:
    """Reduce embeddings to 2D using specified method.

    Args:
        embeddings: Dict of type -> (N, D) arrays
        method: "pca", "tsne", or "umap"
        perplexity: t-SNE perplexity
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        random_state: Random seed (None for UMAP parallelism)
        n_jobs: Number of parallel jobs for UMAP

    Returns:
        Dict of type -> (N, 2) arrays
    """
    # Combine all embeddings for fitting
    all_vecs = []
    type_indices = {}
    start_idx = 0

    for type_name, vecs in embeddings.items():
        all_vecs.append(vecs)
        type_indices[type_name] = (start_idx, start_idx + len(vecs))
        start_idx += len(vecs)

    combined = np.vstack(all_vecs)
    print(
        f"\nReducing {combined.shape[0]} embeddings from {combined.shape[1]}D to 2D using {method.upper()}..."
    )

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced = reducer.fit_transform(combined)
        print(f"  Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    elif method == "tsne":
        # Adjust perplexity if needed
        effective_perplexity = min(perplexity, (combined.shape[0] - 1) // 3)
        if effective_perplexity < perplexity:
            print(f"  Adjusted perplexity from {perplexity} to {effective_perplexity}")
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=random_state,
            max_iter=1000,
        )
        reduced = reducer.fit_transform(combined)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP requires 'umap-learn' package. Install with: pip install umap-learn"
            )

        # Note: random_state=None enables parallelism in UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        reduced = reducer.fit_transform(combined)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    # Split back into types
    result = {}
    for type_name, (start, end) in type_indices.items():
        result[type_name] = reduced[start:end]

    return result


def _get_doc_colors(doc_ids: list[str]) -> dict[str, tuple]:
    """Generate distinct colors for each document.

    Uses colormaps that are darker/more saturated for visibility on white background.
    """
    unique_docs = sorted(set(doc_ids))
    n_docs = len(unique_docs)

    # Use Dark2 and Set1 for smaller sets (darker, more saturated)
    # For larger sets, use a cyclic approach with multiple dark palettes
    if n_docs <= 8:
        cmap = plt.cm.get_cmap("Dark2", 8)
    elif n_docs <= 17:
        # Combine Dark2 (8) + Set1 (9) for up to 17 distinct dark colors
        dark2 = plt.cm.get_cmap("Dark2", 8)
        set1 = plt.cm.get_cmap("Set1", 9)
        colors = {}
        for i, doc in enumerate(unique_docs):
            if i < 8:
                colors[doc] = dark2(i)
            else:
                colors[doc] = set1(i - 8)
        return colors
    else:
        # For many documents, use HSV but with reduced lightness
        # Generate colors in HSV space with lower value (darker)
        import colorsys

        colors = {}
        for i, doc in enumerate(unique_docs):
            hue = i / n_docs
            # Use saturation=0.8 and value=0.7 for darker colors
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.7)
            colors[doc] = (r, g, b, 1.0)
        return colors

    return {doc: cmap(i) for i, doc in enumerate(unique_docs)}


def plot_embeddings(
    reduced: dict[str, np.ndarray],
    method: str,
    output_path: str,
    figsize: tuple[int, int] = (12, 10),
    doc_ids: dict[str, list[str]] | None = None,
    color_mode: str = "type",
    axis_percentile: float | None = 99,
) -> None:
    """Create scatter plot of reduced embeddings.

    Args:
        reduced: Dict of type -> (N, 2) arrays
        method: Method name for title
        output_path: Path to save the figure
        figsize: Figure size
        doc_ids: Optional dict mapping type -> list of doc_ids
        color_mode: "type" (color by node type) or "document" (color by document)
        axis_percentile: Percentile for axis limits (e.g., 99 clips outliers)
    """
    _, ax = plt.subplots(figsize=figsize)

    # Compute axis limits based on percentile to avoid outlier-stretched axes
    if axis_percentile is not None and axis_percentile < 100:
        all_coords = np.vstack(list(reduced.values()))
        low_pct = (100 - axis_percentile) / 2
        high_pct = 100 - low_pct
        x_min, x_max = np.percentile(all_coords[:, 0], [low_pct, high_pct])
        y_min, y_max = np.percentile(all_coords[:, 1], [low_pct, high_pct])
        # Add small margin around limits, x_margin is larger for displaying legend
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Build doc color mapping if needed
    doc_colors = None
    if color_mode == "document" and doc_ids is not None:
        all_docs = []
        for type_docs in doc_ids.values():
            all_docs.extend(type_docs)
        doc_colors = _get_doc_colors(all_docs)
        print(f"Coloring by document ({len(doc_colors)} documents)")

    # Track which docs we've added to legend (for document mode)
    docs_in_legend: set[str] = set()

    # Plot each type
    for type_name, coords in reduced.items():
        marker = TYPE_MARKERS.get(type_name, "o")

        # Adjust marker size based on type
        if type_name == "sentence":
            size = 25
            alpha = 0.6
        elif type_name in ("paragraph_avg", "paragraph_full"):
            size = 30
            alpha = 0.6
        elif type_name == "image":
            size = 80
            alpha = 0.7
        else:
            size = 50
            alpha = 0.7

        if color_mode == "document" and doc_ids is not None and type_name in doc_ids:
            # Color by document - plot each point with its document's color
            type_doc_ids = doc_ids[type_name]
            for i, (x, y) in enumerate(coords):
                doc_id = type_doc_ids[i]
                color = doc_colors[doc_id]

                # Add to legend only once per doc (first occurrence)
                label = None
                if doc_id not in docs_in_legend:
                    # Shorten doc_id for legend
                    short_id = doc_id[:20] + "..." if len(doc_id) > 20 else doc_id
                    label = short_id
                    docs_in_legend.add(doc_id)

                ax.scatter(
                    [x],
                    [y],
                    c=[color],
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    label=label,
                    edgecolors="none" if type_name == "sentence" else "white",
                    linewidths=0.5,
                )
        else:
            # Color by type (original behavior)
            color = TYPE_COLORS.get(type_name, "#999999")
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=f"{type_name} (n={len(coords)})",
                edgecolors="none" if type_name == "sentence" else "white",
                linewidths=0.5,
            )

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    if color_mode == "document":
        ax.set_title(f"Embedding Distribution by Document ({method.upper()})")
    else:
        ax.set_title(f"Embedding Distribution by Node Type ({method.upper()})")

    if color_mode == "document":
        # Create two legends: one for documents (colors), one for node types (shapes)
        from matplotlib.lines import Line2D

        # Document legend (existing scatter handles)
        doc_legend = ax.legend(
            loc="upper left", framealpha=0.9, fontsize=7, title="Documents"
        )
        ax.add_artist(doc_legend)

        # Shape legend for node types
        shape_handles = []
        for type_name in reduced.keys():
            marker = TYPE_MARKERS.get(type_name, "o")
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                markerfacecolor="gray",
                markersize=8,
                linestyle="None",
                label=type_name,
            )
            shape_handles.append(handle)
        ax.legend(
            handles=shape_handles,
            loc="upper right",
            framealpha=0.9,
            fontsize=8,
            title="Node Types",
        )
    else:
        ax.legend(loc="best", framealpha=0.9, fontsize=8)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

    # Also try to show if in interactive mode
    try:
        plt.show()
    except Exception:
        pass


async def main() -> None:
    """Main entry point."""
    print(f"Database: {db}")
    print(f"Table prefix: {table_prefix}")
    print(f"Method: {method}")
    print(f"Show types: {show_types or 'all'}")
    if hierarchical_sampling:
        print("Sampling mode: hierarchical")
        print(f"  Sections per doc: {samples_per_document or 'all'}")
        print(f"  Paragraphs per section: {samples_per_section or 'all'}")
        print(f"  Sentences per paragraph: {samples_per_paragraph or 'all'}")
    else:
        print("Sampling mode: flat")
        print(f"  Max samples per type: {max_samples or 'unlimited'}")
    print()

    # Open store
    store = KVaultNodeStore(Path(db), table_prefix=table_prefix)

    # Collect embeddings using appropriate sampling method
    if hierarchical_sampling:
        embeddings, doc_ids = await collect_embeddings_hierarchical(
            store,
            show_types,
            samples_per_document,
            samples_per_section,
            samples_per_paragraph,
            seed=random_state,
        )
    else:
        embeddings, doc_ids = await collect_embeddings_flat(
            store, show_types, max_samples
        )

    if not embeddings:
        print("No embeddings found!")
        return

    # Reduce dimensions
    reduced = reduce_dimensions(
        embeddings,
        method=method,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=(
            random_state if method != "umap" else None
        ),  # None for UMAP parallelism
        n_jobs=n_jobs,
    )

    # Plot
    plot_embeddings(
        reduced,
        method,
        output,
        figsize,
        doc_ids=doc_ids,
        color_mode=color_mode,
        axis_percentile=axis_percentile,
    )

    # Save 2D coordinates and metadata to JSON for downstream analysis
    import json

    output_json = output.rsplit(".", 1)[0] + "_data.json"
    export_data = {"method": method, "points": []}
    for type_name, coords in reduced.items():
        type_doc_ids = doc_ids.get(type_name, []) if doc_ids else []
        for i, (x, y) in enumerate(coords):
            point = {
                "x": float(x),
                "y": float(y),
                "type": type_name,
                "doc_id": type_doc_ids[i] if i < len(type_doc_ids) else None,
            }
            export_data["points"].append(point)

    _write_export_data(output_json, export_data)
    print(f"Saved coordinate data to: {output_json}")


def _write_export_data(output_json: str, export_data: dict) -> None:
    """Synchronous helper so the async caller does not perform sync file I/O."""
    with open(output_json, "w") as f:
        json.dump(export_data, f)


if __name__ == "__main__":
    asyncio.run(main())

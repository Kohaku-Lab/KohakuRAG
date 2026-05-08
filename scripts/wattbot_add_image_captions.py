"""Add AI-generated captions to images in parsed PDF documents.

This script uses 3-phase parallel processing:
1. Read ALL images from ALL documents (parallel)
2. Compress ALL images (parallel)
3. Caption ALL images (parallel)

Usage (CLI):
    python scripts/wattbot_add_image_captions.py \\
        --docs-dir artifacts/docs \\
        --pdf-dir artifacts/raw_pdfs \\
        --output-dir artifacts/docs_with_images \\
        --db artifacts/wattbot_with_images.db \\
        --vision-model qwen/qwen3-vl-235b-a22b-instruct \\
        --limit 10

Usage (KohakuEngine):
    kogine run scripts/wattbot_add_image_captions.py --config configs/caption_config.py
"""

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kohakurag.datastore import ImageStore
from kohakurag.image_utils import compress_image, get_image_dimensions
from kohakurag.parsers import dict_to_payload, payload_to_dict
from kohakurag.pdf_utils import _extract_images
from kohakurag.types import SentencePayload
from kohakurag.vision import OpenAIVisionModel, OpenRouterVisionModel

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

docs_dir = "artifacts/docs"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"

# Vision model settings
llm_provider = "openrouter"  # Options: "openai", "openrouter"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
openrouter_api_key = None  # From env: OPENROUTER_API_KEY

limit = 0
max_concurrent = 5


@dataclass
class ImageTask:
    """Represents a single image to process."""

    doc_id: str
    page_num: int
    img_idx: int
    img_name: str
    storage_key: str
    raw_data: bytes | None = None
    compressed_data: bytes | None = None
    width: int | str = "unknown"
    height: int | str = "unknown"
    caption: str | None = None


def read_images_from_document(
    json_path: Path, pdf_dir: Path, idx: int, total: int
) -> tuple[str, list[ImageTask]]:
    """Read all images from one document (synchronous, runs in thread).

    Returns:
        (doc_id, list of ImageTask with raw_data populated)
    """
    doc_id = json_path.stem
    pdf_path = pdf_dir / f"{doc_id}.pdf"

    print(f"[{idx}/{total}] {doc_id}... ", end="", flush=True)

    if not pdf_path.exists():
        print("SKIP (no PDF)")
        return (doc_id, [])

    try:
        # Load JSON payload
        payload = dict_to_payload(json.loads(json_path.read_text(encoding="utf-8")))

        # Load PDF
        reader = PdfReader(str(pdf_path))

        tasks = []

        # Process each section/page
        for section in payload.sections or []:
            page_num = section.metadata.get("page", 1)
            if page_num < 1 or page_num > len(reader.pages):
                continue

            page = reader.pages[page_num - 1]

            # Extract images from this page
            try:
                images = _extract_images(page)
            except Exception:
                continue

            if not images:
                continue

            # Create lookup
            image_lookup = dict(enumerate(images, 1))

            # Find image paragraphs
            for para in section.paragraphs:
                if para.metadata.get("attachment_type") != "image":
                    continue

                img_idx = para.metadata.get("image_index")
                img_info = image_lookup.get(img_idx)

                if not img_info or not img_info.get("data"):
                    continue

                # Create task with raw data
                img_name = para.metadata.get("image_name", f"img{img_idx}")
                storage_key = ImageStore.make_key(doc_id, page_num, img_idx)

                task = ImageTask(
                    doc_id=doc_id,
                    page_num=page_num,
                    img_idx=img_idx,
                    img_name=img_name,
                    storage_key=storage_key,
                    raw_data=img_info["data"],
                )
                tasks.append(task)

        if tasks:
            print(f"✓ {len(tasks)} images")
        else:
            print("no images")
        return (doc_id, tasks)

    except Exception as e:
        print(f"ERROR: {e}")
        return (doc_id, [])


def compress_one_image(task: ImageTask, idx: int, total: int) -> ImageTask:
    """Compress one image (synchronous, runs in thread).

    Updates task.compressed_data, width, height in place.
    Returns the task for chaining.
    """
    try:
        compressed = compress_image(
            task.raw_data,
            max_size=1024,
            format="jpeg",  # JPEG is much faster than WebP
            quality=95,
        )

        dims = get_image_dimensions(compressed)
        width, height = dims if dims else ("unknown", "unknown")

        task.compressed_data = compressed
        task.width = width
        task.height = height

        # Log progress
        orig_kb = len(task.raw_data) / 1024
        comp_kb = len(compressed) / 1024
        ratio = (1 - len(compressed) / len(task.raw_data)) * 100
        print(
            f"  [{idx}/{total}] ✓ {task.doc_id} p{task.page_num}:i{task.img_idx} - {orig_kb:.1f}KB→{comp_kb:.1f}KB ({ratio:.0f}% saved)"
        )

    except Exception as e:
        print(f"  [{idx}/{total}] ❌ {task.storage_key}: {e}")

    return task


def create_vision_model():
    """Create vision model based on llm_provider config.

    Returns:
        VisionModel instance (OpenAI or OpenRouter)
    """
    if llm_provider == "openrouter":
        return OpenRouterVisionModel(
            model=vision_model,
            api_key=openrouter_api_key,
            max_concurrent=max_concurrent,
        )
    else:
        # Default: OpenAI
        return OpenAIVisionModel(
            model=vision_model,
            max_concurrent=max_concurrent,
        )


async def main() -> None:
    """Main entry point."""
    docs_dir_path = Path(docs_dir)
    pdf_dir_path = Path(pdf_dir)
    output_dir_path = Path(output_dir)
    db_path = Path(db)

    # Validate
    if not docs_dir_path.exists():
        print(f"ERROR: {docs_dir_path} not found")
        sys.exit(1)
    if not pdf_dir_path.exists():
        print(f"ERROR: {pdf_dir_path} not found")
        sys.exit(1)

    json_files = sorted(docs_dir_path.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]

    if not json_files:
        print(f"No JSON files in {docs_dir_path}")
        sys.exit(1)

    print("=" * 60)
    print("KohakuRAG - Parallel Batch Image Captioning")
    print("=" * 60)
    print(f"Documents: {len(json_files)}")
    print(f"Database:  {db_path}")
    print(f"Provider:  {llm_provider}")
    print(f"Model:     {vision_model}")
    print(f"Concurrency: {max_concurrent}")
    print("=" * 60)

    # Initialize components
    vision_client = create_vision_model()
    image_store = ImageStore(db_path, table="image_blobs")

    executor = ThreadPoolExecutor(max_workers=8)
    loop = asyncio.get_event_loop()

    # ========================================================================
    # PHASE 1: Read all images from all documents (parallel)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Reading images from all documents (parallel)")
    print("=" * 60)

    t_start = time.time()

    # Run all PDF reads in parallel threads
    read_tasks = [
        loop.run_in_executor(
            executor,
            read_images_from_document,
            json_path,
            pdf_dir_path,
            i + 1,
            len(json_files),
        )
        for i, json_path in enumerate(json_files)
    ]

    results = await asyncio.gather(*read_tasks)

    # Flatten all tasks
    all_tasks = []
    doc_to_json = {}
    for doc_id, tasks in results:
        doc_to_json[doc_id] = docs_dir_path / f"{doc_id}.json"
        all_tasks.extend(tasks)

    t_read = time.time()
    print(
        f"\n✓ Collected {len(all_tasks)} images from {len(json_files)} docs ({t_read - t_start:.2f}s)"
    )

    if not all_tasks:
        print("\n⚠️  No images found!")
        sys.exit(0)

    # ========================================================================
    # PHASE 2: Compress all images (parallel)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Compressing all images (parallel)")
    print("=" * 60)

    t_start = time.time()

    print()  # New line after Phase 1 results

    # Compress all in parallel threads
    compress_tasks = [
        loop.run_in_executor(executor, compress_one_image, task, i + 1, len(all_tasks))
        for i, task in enumerate(all_tasks)
    ]

    compressed_tasks = await asyncio.gather(*compress_tasks)

    # Count successes
    successful = [t for t in compressed_tasks if t.compressed_data is not None]
    t_compress = time.time()

    print(
        f"\n✓ Compressed {len(successful)}/{len(all_tasks)} images ({t_compress - t_start:.2f}s)"
    )

    # ========================================================================
    # PHASE 3: Caption all images (parallel)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Captioning all images (parallel)")
    print("=" * 60)

    t_start = time.time()

    async def caption_one(task: ImageTask, idx: int) -> ImageTask:
        """Caption one image."""
        if task.compressed_data is None:
            return task

        try:
            caption = await vision_client.caption(task.compressed_data)
            task.caption = caption
            print(
                f"  [{idx}/{len(successful)}] ✓ {task.doc_id} p{task.page_num}:i{task.img_idx}"
            )
        except Exception as e:
            print(
                f"  [{idx}/{len(successful)}] ❌ {task.doc_id} p{task.page_num}:i{task.img_idx}: {e}"
            )

        return task

    # Caption all concurrently
    caption_tasks = [caption_one(task, i + 1) for i, task in enumerate(successful)]
    captioned = await asyncio.gather(*caption_tasks)

    # Count successes
    with_captions = [t for t in captioned if t.caption]
    t_caption = time.time()

    print(
        f"\n✓ Captioned {len(with_captions)}/{len(successful)} images ({t_caption - t_start:.2f}s)"
    )

    # ========================================================================
    # PHASE 4: Update documents and store images
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Storing images and updating documents")
    print("=" * 60)

    t_start = time.time()

    # Group tasks by document (only docs with captions)
    tasks_by_doc = {}
    for task in with_captions:
        if task.doc_id not in tasks_by_doc:
            tasks_by_doc[task.doc_id] = []
        tasks_by_doc[task.doc_id].append(task)

    # IMPORTANT: Process ALL documents, not just ones with images!
    all_doc_ids = set(doc_to_json.keys())

    stats = {"docs": 0, "captions": 0, "images": 0, "errors": 0}

    for idx, doc_id in enumerate(sorted(all_doc_ids), 1):
        tasks = tasks_by_doc.get(doc_id, [])  # Empty list if no images
        print(f"\n[{idx}/{len(all_doc_ids)}] {doc_id}", end="")

        json_path = doc_to_json.get(doc_id)
        if not json_path:
            continue

        try:
            # Load payload
            payload = dict_to_payload(json.loads(json_path.read_text(encoding="utf-8")))

            # Update image paragraphs
            for task in tasks:
                if not task.caption:
                    continue

                # Find the paragraph
                found = False
                for section in payload.sections or []:
                    if section.metadata.get("page") != task.page_num:
                        continue

                    for para in section.paragraphs:
                        if (
                            para.metadata.get("attachment_type") == "image"
                            and para.metadata.get("image_index") == task.img_idx
                        ):
                            # Update with formatted caption
                            formatted = f"[img:{task.img_name} {task.width}x{task.height}] {task.caption}"
                            para.text = formatted
                            para.sentences = [SentencePayload(text=formatted)]
                            para.metadata.update(
                                {
                                    "caption_source": "vision_model",
                                    "image_storage_key": task.storage_key,
                                    "compressed_width": (
                                        task.width
                                        if isinstance(task.width, int)
                                        else None
                                    ),
                                    "compressed_height": (
                                        task.height
                                        if isinstance(task.height, int)
                                        else None
                                    ),
                                }
                            )

                            found = True
                            stats["captions"] += 1
                            break

                    if found:
                        break

                # Store compressed image
                if task.compressed_data:
                    try:
                        await image_store.store_image(
                            task.storage_key, task.compressed_data
                        )
                        stats["images"] += 1
                    except Exception as e:
                        print(f"    ⚠️  Failed to store {task.storage_key}: {e}")

            # Save updated payload
            output_path = output_dir_path / f"{doc_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload_to_dict(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stats["docs"] += 1

            if len(tasks) > 0:
                print(f" - ✓ {len(tasks)} captions")
            else:
                print(" - ✓ (no images)")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            stats["errors"] += 1

    t_store = time.time()
    print(f"\n✓ Stored images and updated documents ({t_store - t_start:.2f}s)")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Documents updated:     {stats['docs']}")
    print(f"Captions added:        {stats['captions']}")
    print(f"Images stored:         {stats['images']}")
    print(f"Errors:                {stats['errors']}")
    print("=" * 60)

    executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())

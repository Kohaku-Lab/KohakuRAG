"""Utilities for parsing PDFs (arXiv papers, reports) into structured payloads."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.generic import DictionaryObject, IndirectObject

from .text_utils import split_paragraphs, split_sentences
from .types import (
    DocumentPayload,
    ParagraphPayload,
    SectionPayload,
    SentencePayload,
)


def _resolve(obj):
    return obj.get_object() if isinstance(obj, IndirectObject) else obj


def _extract_images(page) -> list[dict[str, Any]]:
    """Extract images from a PDF page using pypdf's images API.

    Returns list of dicts with image metadata and optionally the image data.
    """
    images: list[dict[str, Any]] = []
    try:
        # Modern pypdf API (>= 3.1.0) - much more reliable
        for img in page.images:
            image_info = {
                "name": img.name,
                "data": img.data,  # Raw image bytes
            }
            # Try to get dimensions from the image object
            if hasattr(img, "image") and img.image:
                try:
                    image_info["width"] = img.image.width
                    image_info["height"] = img.image.height
                except Exception:
                    pass
            images.append(image_info)
    except Exception:
        # Fallback to old method if page.images fails
        images = _extract_images_fallback(page)
    return images


def _extract_images_fallback(page) -> list[dict[str, Any]]:
    """Fallback method using low-level PDF dictionary access."""
    images: list[dict[str, Any]] = []
    resources = page.get("/Resources")
    if not resources:
        return images
    resources = _resolve(resources)
    xobject = (
        resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
    )
    if xobject is None:
        return images
    xobject = _resolve(xobject)
    if not isinstance(xobject, DictionaryObject):
        return images
    for name, obj in xobject.items():
        resolved = _resolve(obj)
        if not isinstance(resolved, DictionaryObject):
            continue
        subtype = resolved.get("/Subtype")
        # Convert to string for comparison - pypdf returns NameObject, not str
        if str(subtype) == "/Image":
            images.append(
                {
                    "name": str(name),
                    "width": resolved.get("/Width"),
                    "height": resolved.get("/Height"),
                    "color_space": resolved.get("/ColorSpace"),
                    "data": None,  # Not extracted in fallback
                }
            )
    return images


def pdf_to_document_payload(
    pdf_path: Path,
    *,
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
) -> DocumentPayload:
    reader = PdfReader(str(pdf_path))
    sections: list[SectionPayload] = []
    all_paragraph_texts: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        paragraphs = []
        for paragraph_text in split_paragraphs(raw_text):
            sentences = [
                SentencePayload(text=sentence)
                for sentence in split_sentences(paragraph_text)
            ]
            paragraphs.append(
                ParagraphPayload(
                    text=paragraph_text,
                    sentences=sentences or None,
                    metadata={"page": page_num},
                )
            )
            all_paragraph_texts.append(paragraph_text)
        images = _extract_images(page)
        for idx, info in enumerate(images, start=1):
            # Create informative caption with available metadata
            width = info.get("width", "unknown")
            height = info.get("height", "unknown")
            has_data = info.get("data") is not None
            data_size = len(info.get("data", b"")) if has_data else 0

            caption = (
                f"[Image page={page_num} idx={idx} name={info.get('name', 'unknown')}] "
                f"Size: {width}x{height}, Data: {data_size} bytes"
            )
            sentences = [SentencePayload(text=caption)]

            # Store image data in metadata for later processing
            image_metadata = {
                "page": page_num,
                "image_index": idx,
                "image_name": info.get("name"),
                "image_width": info.get("width"),
                "image_height": info.get("height"),
                "attachment_type": "image",
                "has_image_data": has_data,
                "image_data_size": data_size,
            }

            # Optionally include the actual image data (can be large!)
            # Uncomment if you want to store the raw bytes
            # if has_data:
            #     image_metadata["image_data"] = info.get('data')

            paragraphs.append(
                ParagraphPayload(
                    text=caption,
                    sentences=sentences,
                    metadata=image_metadata,
                )
            )
            all_paragraph_texts.append(caption)
        if paragraphs:
            sections.append(
                SectionPayload(
                    title=f"Page {page_num}",
                    paragraphs=paragraphs,
                    metadata={"page": page_num},
                )
            )
    combined_text = "\n\n".join(all_paragraph_texts)
    return DocumentPayload(
        document_id=doc_id,
        title=title,
        text=combined_text,
        metadata=metadata,
        sections=sections,
    )


def pdf_to_markdown(
    pdf_path: Path,
    *,
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
) -> str:
    payload = pdf_to_document_payload(
        pdf_path, doc_id=doc_id, title=title, metadata=metadata
    )
    lines = [f"# {title}", ""]
    for section in payload.sections or []:
        lines.append(f"## {section.title}")
        lines.append("")
        for paragraph in section.paragraphs:
            lines.append(paragraph.text)
            lines.append("")
    return "\n".join(lines)

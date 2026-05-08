"""Convert PDF to LLM-friendly document format.

Exports a PDF to:
    <output>/main.txt        - Markdown-like text content with image references
    <output>/pages/*.png     - Rendered page images
    <output>/images/*.png    - Extracted embedded images

Usage (CLI):
    python scripts/pdf_to_documents.py input.pdf
    python scripts/pdf_to_documents.py input.pdf --output ./my_output

Usage (KohakuEngine):
    kogine run scripts/pdf_to_documents.py --config configs/pdf_to_documents.py

Configuration:
    input: Path to input PDF file
    output: Output directory (default: same name as PDF without extension)
    page_dpi: DPI for rendered page images (default: 150)
    image_format: Format for images - "png" or "jpg" (default: "png")
    extract_images: Whether to extract embedded images (default: True)
    render_pages: Whether to render page images (default: True)
    min_image_size: Minimum image dimension to extract (default: 50)
"""

import re
import sys
from pathlib import Path

import fitz  # pymupdf


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

input: str = ""
output: str | None = None  # None = use input filename without extension
page_dpi: int = 150
image_format: str = "png"  # "png" or "jpg" (images with alpha always saved as PNG)
extract_images: bool = True
render_pages: bool = True
min_image_size: int = 50  # Minimum width/height to extract embedded images


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    # Remove or replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name[:100] if name else "image"


def _save_pixmap(pix: fitz.Pixmap, output_path: Path, fmt: str) -> None:
    """Save pixmap to file with format handling.

    - Images with alpha channel are saved as PNG (RGBA)
    - Images without alpha are saved in the requested format (png/jpg)
    - When converting RGBA to JPG, a white background is composited
    """
    from PIL import Image

    # If pixmap has alpha, always save as PNG to preserve transparency
    if pix.alpha:
        img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
        # Change extension to .png for alpha images
        output_path = output_path.with_suffix(".png")
        img.save(output_path, "PNG")
    elif fmt == "jpg":
        # No alpha, save as JPEG
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_path, "JPEG", quality=95)
    else:
        # No alpha, save as PNG
        pix.save(output_path)


def _extract_embedded_images(
    doc: fitz.Document,
    images_dir: Path,
    fmt: str,
    min_size: int,
) -> dict[tuple[int, int], str]:
    """Extract embedded images from PDF.

    Returns:
        Dict mapping (xref, page_num) to image filename
    """
    from PIL import Image
    import io

    images_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[tuple[int, int], str] = {}
    image_counter = 0

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            key = (xref, page_num)

            # Skip if already extracted (same xref can appear multiple times)
            if any(k[0] == xref for k in image_map):
                # Find existing filename for this xref
                for k, v in image_map.items():
                    if k[0] == xref:
                        image_map[key] = v
                        break
                continue

            try:
                base_image = doc.extract_image(xref)
                if base_image is None:
                    continue

                width = base_image["width"]
                height = base_image["height"]

                # Skip small images (likely icons, bullets, etc.)
                if width < min_size or height < min_size:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                smask_xref = base_image.get("smask", 0)  # Soft mask (alpha) xref

                # Load the base image
                img = Image.open(io.BytesIO(image_bytes))

                # Check if image has alpha - either in mode or via separate soft mask
                has_alpha = img.mode in ("RGBA", "LA", "PA") or "A" in img.mode

                # Handle soft mask (separate alpha channel in PDF)
                # SMask always takes priority - base image alpha is often just placeholder (all 255)
                if smask_xref:
                    try:
                        smask_image = doc.extract_image(smask_xref)
                        if smask_image:
                            alpha_bytes = smask_image["image"]
                            alpha_img = Image.open(io.BytesIO(alpha_bytes)).convert("L")
                            # Resize alpha if dimensions don't match
                            if alpha_img.size != img.size:
                                alpha_img = alpha_img.resize(img.size, Image.LANCZOS)
                            # Convert base image to RGB first (discard any fake alpha), then add real alpha
                            img = img.convert("RGB")
                            img = img.convert("RGBA")
                            img.putalpha(alpha_img)
                            has_alpha = True
                    except Exception as e:
                        print(f"  Warning: Failed to extract soft mask: {e}")

                # Force PNG for images with alpha to preserve transparency
                actual_fmt = "png" if has_alpha else fmt

                image_counter += 1
                filename = f"img_{image_counter:04d}.{actual_fmt}"
                output_path = images_dir / filename

                # Save image with proper handling
                # - Images with alpha: always save as RGBA PNG
                # - Images without alpha: save in requested format (png/jpg)
                if has_alpha:
                    # For images with alpha, convert to RGBA and save as PNG
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    img.save(output_path, "PNG")
                elif actual_fmt == "jpg":
                    # Save as JPEG - convert to RGB if needed
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    img.save(output_path, "JPEG", quality=95)
                elif actual_fmt == image_ext:
                    # Same format, no conversion needed - just write bytes
                    output_path.write_bytes(image_bytes)
                else:
                    # Convert to PNG
                    img.save(output_path, "PNG")

                image_map[key] = filename
                smask_info = f", smask={smask_xref}" if smask_xref else ""
                print(
                    f"  Extracted: {filename} ({width}x{height}, mode={img.mode}{smask_info})"
                )

            except Exception as e:
                print(f"  Warning: Failed to extract image xref={xref}: {e}")
                continue

    return image_map


def _render_pages(
    doc: fitz.Document,
    pages_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    """Render all pages to images."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    total_pages = len(doc)
    pad = len(str(total_pages))

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        filename = f"{str(page_num + 1).zfill(pad)}.{fmt}"
        output_path = pages_dir / filename
        _save_pixmap(pix, output_path, fmt)
        print(f"  Rendered: {filename}")


def _extract_text_with_images(
    doc: fitz.Document,
    image_map: dict[tuple[int, int], str],
) -> str:
    """Extract text content with image references."""
    lines: list[str] = []
    total_pages = len(doc)

    for page_num, page in enumerate(doc):
        # Page header
        lines.append(f"\n{'='*60}")
        lines.append(f"PAGE {page_num + 1} / {total_pages}")
        lines.append(f"{'='*60}\n")

        # Get text blocks with position info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Sort blocks by position (top to bottom, left to right)
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    text_parts = []
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if text.strip():
                            text_parts.append(text)
                    if text_parts:
                        line_text = "".join(text_parts)
                        lines.append(line_text)

            elif block["type"] == 1:  # Image block
                # Find image reference
                xref = block.get("xref", 0)
                key = (xref, page_num)
                if key in image_map:
                    img_filename = image_map[key]
                    lines.append(f"\n[IMAGE: images/{img_filename}]\n")
                else:
                    # Check if any image on this page matches
                    for k, v in image_map.items():
                        if k[1] == page_num and k[0] == xref:
                            lines.append(f"\n[IMAGE: images/{v}]\n")
                            break

        # Add page image reference
        pad = len(str(total_pages))
        page_img = f"{str(page_num + 1).zfill(pad)}.{image_format}"
        lines.append(f"\n[PAGE IMAGE: pages/{page_img}]\n")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    # Handle CLI arguments
    global input, output

    if not input and len(sys.argv) > 1:
        input = sys.argv[1]
        if len(sys.argv) > 3 and sys.argv[2] == "--output":
            output = sys.argv[3]

    if not input:
        print("Error: No input file specified")
        print("Usage: python pdf_to_documents.py <input.pdf> [--output <dir>]")
        return

    input_path = Path(input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Determine output directory
    if output:
        output_dir = Path(output)
    else:
        output_dir = input_path.parent / input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    images_dir = output_dir / "images"

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Page DPI: {page_dpi}")
    print(f"Image format: {image_format}")
    print()

    # Open PDF
    doc = fitz.open(input_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    # Extract embedded images
    image_map: dict[tuple[int, int], str] = {}
    if extract_images:
        print("\nExtracting embedded images...")
        image_map = _extract_embedded_images(
            doc, images_dir, image_format, min_image_size
        )
        print(f"Extracted {len(set(image_map.values()))} unique images")

    # Render page images
    if render_pages:
        print("\nRendering page images...")
        _render_pages(doc, pages_dir, page_dpi, image_format)
        print(f"Rendered {total_pages} pages")

    # Extract text with image references
    print("\nExtracting text content...")
    text_content = _extract_text_with_images(doc, image_map)

    # Write main.txt
    main_txt = output_dir / "main.txt"
    main_txt.write_text(text_content, encoding="utf-8")
    print(f"Wrote: {main_txt}")

    doc.close()
    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

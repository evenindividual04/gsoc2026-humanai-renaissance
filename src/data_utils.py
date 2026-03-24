"""
Data utilities for the RenAIssance OCR pipeline.

Handles:
- Ground truth extraction from DOCX transcription files
- PDF page loading via PyMuPDF
- Page-to-GT alignment
- Visualization helpers
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

def _normalize_stem(s: str) -> str:
    s = s.lower()

    # Remove "transcription"
    s = re.sub(r"\btranscription\b", "", s)

    # Normalize all dash variants → "-"
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)

    # Normalize colon variants → "."
    s = s.replace("&#x3a;", ".").replace(":", ".")

    # Remove ALL spaces + hyphens
    s = re.sub(r"[\s\-]+", "", s)

    return s


# ── Ground Truth Parsing ──────────────────────────────────────────────────────

def extract_ground_truth(docx_path: str | Path) -> Dict[int, str]:
    """
    Parse a transcription DOCX and return a mapping of {page_number: text}.

    Page markers appear as "PDF p2", "PDF p.2", "[p. 2]", etc. Lines that
    look like transcriber notes (square-bracketed annotations other than page
    markers, lines starting with '#') are stripped.

    Args:
        docx_path: Path to the DOCX ground-truth file.

    Returns:
        Dictionary mapping integer page numbers to their transcribed text.
        Page 1 is used as default when no explicit marker appears before the
        first content block.
    """
    from docx import Document

    doc = Document(str(docx_path))
    full_text = "\n".join(p.text for p in doc.paragraphs)

    # Normalise Unicode dashes / spaces that Word sometimes inserts
    full_text = full_text.replace("\u2013", "-").replace("\u2014", "-")
    full_text = full_text.replace("\u00a0", " ")

    # Regex to detect page markers: "PDF p2", "PDF p. 2", "[p. 2]", "p.2", etc.
    page_marker_re = re.compile(
        r"(?:PDF\s+p\.?\s*(\d+)|\[p\.?\s*(\d+)\]|^\s*p\.?\s*(\d+)\s*$)",
        re.MULTILINE | re.IGNORECASE,
    )
    # Note / annotation lines to strip (but NOT page markers)
    note_re = re.compile(r"^\s*\[(?!p\.?\s*\d).*?\]\s*$", re.MULTILINE)
    comment_re = re.compile(r"^\s*#.*$", re.MULTILINE)
    # "NOTES:" block — transcriber metadata header (tab-separated columns of rules)
    notes_block_re = re.compile(r"^\s*NOTES\s*:", re.IGNORECASE)

    lines = full_text.split("\n")
    pages: Dict[int, List[str]] = {}
    current_page = 1
    in_notes_block = False

    for line in lines:
        # Detect start of NOTES: block
        if notes_block_re.match(line):
            in_notes_block = True
            continue
        # NOTES block continues while lines are blank or tab/heavily-indented
        if in_notes_block:
            if not line.strip() or line.startswith("\t") or line.startswith("    "):
                continue
            else:
                in_notes_block = False  # Block ended, resume normal processing

        # Check for page marker
        m = page_marker_re.search(line)
        if m:
            page_num = int(m.group(1) or m.group(2) or m.group(3))
            current_page = page_num
            if current_page not in pages:
                pages[current_page] = []
            continue

        # Strip transcriber notes and comments
        if note_re.match(line) or comment_re.match(line):
            continue

        if current_page not in pages:
            pages[current_page] = []
        pages[current_page].append(line)

    # Join and clean each page
    result = {}
    for page_num, page_lines in pages.items():
        text = "\n".join(page_lines).strip()
        # Collapse multiple blank lines to a single one
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            result[page_num] = text

    return result


def load_all_ground_truth(transcriptions_dir: str | Path) -> Dict[str, Dict[int, str]]:
    """
    Load all DOCX GT files from a directory.

    Returns:
        {stem_name: {page_num: text}} for each DOCX found.
    """
    transcriptions_dir = Path(transcriptions_dir)
    gt_map = {}
    for docx_file in sorted(transcriptions_dir.glob("*.docx")):
        try:
            gt_map[docx_file.stem] = extract_ground_truth(docx_file)
        except Exception as e:
            print(f"[WARN] Could not parse {docx_file.name}: {e}")
    return gt_map


# ── PDF Loading ───────────────────────────────────────────────────────────────

def load_pdf_pages(
    pdf_path: str | Path,
    dpi: int = 300,
    pages: Optional[set] = None,
) -> List[Image.Image]:
    """
    Render pages of a PDF to PIL Images at the given DPI.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Render resolution (300 recommended for OCR quality).
        pages: Set of 1-indexed page numbers to render. If None, renders all
               pages. Pass the GT page numbers here to avoid loading entire
               multi-hundred-page books into memory.

    Returns:
        List of (page_number, PIL.Image) tuples for the requested pages.
        Stored as a plain list of Images when pages=None (backwards compat).
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Build a dict {page_num: image} so alignment stays correct when only
    # a subset of pages is loaded.
    page_images: Dict[int, Image.Image] = {}
    for page in doc:
        page_num = page.number + 1  # 1-indexed
        if pages is not None and page_num not in pages:
            continue
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images[page_num] = img

    doc.close()

    if pages is None:
        # Backwards-compatible: return a flat list (0-indexed)
        return [page_images[i + 1] for i in range(len(page_images))]
    return page_images  # Dict[int, Image] when pages filter is used


def load_all_pdfs(
    sources_dir: str | Path,
    dpi: int = 300,
    gt_map: Optional[Dict] = None,
) -> Dict[str, List[Image.Image]]:
    """
    Load PDFs from a directory, optionally restricted to GT-covered pages.

    Args:
        sources_dir: Directory containing PDF files.
        dpi: Render resolution.
        gt_map: If provided ({stem: {page_num: text}}), only the pages that
                have GT entries are rendered. This prevents loading entire
                multi-hundred-page books for documents where GT covers only
                a handful of pages.

    Returns:
        {stem_name: [page_images]} for each PDF found. When gt_map is used,
        only GT-covered pages are present (list may be shorter than total
        page count).
    """
    sources_dir = Path(sources_dir)
    pdf_map = {}
    for pdf_file in sorted(sources_dir.glob("*.pdf")):
        stem = pdf_file.stem
        try:
            pages_to_load = None
            if gt_map is not None:
                gt_key = stem
                if gt_key not in gt_map:
                    norm_stem = _normalize_stem(stem)
                    for gk in gt_map:
                        norm_gk = _normalize_stem(gk)
                        if norm_stem in norm_gk or norm_gk in norm_stem:
                            gt_key = gk
                            break
                if gt_key in gt_map:
                    pages_to_load = set(gt_map[gt_key].keys())

            result = load_pdf_pages(pdf_file, dpi=dpi, pages=pages_to_load)

            if isinstance(result, dict):
                pdf_map[stem] = result
            else:
                pdf_map[stem] = result

            n = len(result)
            total_str = f" ({n} of requested pages)" if pages_to_load else f" ({n} pages)"
            print(f"  {pdf_file.name}: loaded{total_str}")
        except Exception as e:
            print(f"[WARN] Could not load {pdf_file.name}: {e}")
    return pdf_map


# ── Page Alignment ────────────────────────────────────────────────────────────

def align_pages_to_gt(
    page_images,
    gt_dict: Dict[int, str],
) -> List[Tuple[Image.Image, str]]:
    """
    Pair page images with their ground-truth text.

    Accepts either a list (0-indexed, all pages) or a dict {page_num: image}
    (subset of pages, as returned by load_pdf_pages with a pages filter).

    Args:
        page_images: List of PIL Images (0-indexed) OR Dict[int, PIL.Image]
                     (1-indexed page numbers → images).
        gt_dict: {page_num (1-indexed): ground_truth_text}

    Returns:
        List of (image, gt_text) tuples for pages with GT coverage.
    """
    pairs = []
    if isinstance(page_images, dict):
        # Dict path: keys are already 1-indexed page numbers
        for page_num in sorted(page_images):
            if page_num in gt_dict:
                pairs.append((page_images[page_num], gt_dict[page_num]))
    else:
        # List path: 0-indexed, convert to 1-indexed
        for idx, img in enumerate(page_images):
            page_num = idx + 1
            if page_num in gt_dict:
                pairs.append((img, gt_dict[page_num]))
    return pairs


def build_dataset_pairs(
    pdf_map: Dict[str, List[Image.Image]],
    gt_map: Dict[str, Dict[int, str]],
) -> Dict[str, List[Tuple[Image.Image, str]]]:
    """
    Align all PDFs with their GT files by matching stem names.

    Returns:
        {doc_name: [(page_image, gt_text), ...]}
    """
    pairs_map = {}
    for stem, images in pdf_map.items():
        # Try exact match, then fuzzy match (strip " transcription" suffix from GT)
        gt_key = stem
        if gt_key not in gt_map:
            # Try matching by looking for a GT key that shares a prefix
            # Normalize dashes and case to handle em-dash vs hyphen mismatches
            norm_stem = _normalize_stem(stem)
            for gk in gt_map:
                norm_gk = _normalize_stem(gk)
                if norm_stem in norm_gk or norm_gk in norm_stem:
                    gt_key = gk
                    break
        if gt_key in gt_map:
            pairs_map[stem] = align_pages_to_gt(images, gt_map[gt_key])
        else:
            print(f"[WARN] No GT found for {stem} — skipping.")
    return pairs_map


# ── Text Normalisation ────────────────────────────────────────────────────────

def normalize_for_eval(text: str) -> str:
    """
    Light normalization applied to both hypothesis and reference before scoring.

    - Collapse whitespace
    - Strip leading/trailing whitespace per line
    - Preserve historical spellings (do NOT modernize)
    """
    lines = [l.strip() for l in text.splitlines()]
    text = " ".join(l for l in lines if l)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── Visualization ─────────────────────────────────────────────────────────────

def show_page_with_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    title: str = "",
    figsize: Tuple[int, int] = (14, 18),
):
    """
    Display a page image with bounding boxes overlaid (for notebooks).

    Args:
        image: PIL Image of the page.
        boxes: List of (x1, y1, x2, y2) bounding boxes in pixel coords.
        title: Plot title.
        figsize: Matplotlib figure size.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(np.array(image))
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor="red", facecolor="none", alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, str(i), fontsize=6, color="red")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_preprocessing_comparison(
    original: Image.Image,
    processed: Image.Image,
    titles: Tuple[str, str] = ("Original", "Preprocessed"),
    figsize: Tuple[int, int] = (16, 8),
):
    """Show original vs. processed page side by side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [original, processed], titles):
        ax.imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_line_crops(
    line_images: List[Image.Image],
    n: int = 10,
    figsize: Tuple[int, int] = (16, 12),
):
    """Display a grid of line crops."""
    import matplotlib.pyplot as plt

    n = min(n, len(line_images))
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, line_images[:n]):
        ax.imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

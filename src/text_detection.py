"""
Text line detection using CRAFT (Character Region Awareness for Text Detection).

CRAFT (Baek et al., CVPR 2019) predicts character-level region maps and
affinity maps, making it robust to irregular text layouts — including the
wide inter-word spacing and varied line heights common in 17th-century
Spanish printed documents.

Reference:
  Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019).
  Character region awareness for text detection. CVPR 2019.
  arXiv:1904.01941
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── CRAFT Weight Loading ──────────────────────────────────────────────────────

CRAFT_WEIGHTS_URL = (
    "https://github.com/clovaai/CRAFT-pytorch/releases/download/"
    "pre-trained.pth/craft_mlt_25k.pth"
)
CRAFT_WEIGHTS_FILENAME = "craft_mlt_25k.pth"


def download_craft_weights(save_dir: str | Path = "/tmp") -> Path:
    """
    Download CRAFT pre-trained weights if not already present.

    Args:
        save_dir: Directory to save the weights file.

    Returns:
        Path to the downloaded weights file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / CRAFT_WEIGHTS_FILENAME

    if not weights_path.exists():
        print(f"Downloading CRAFT weights to {weights_path}...")
        urllib.request.urlretrieve(CRAFT_WEIGHTS_URL, str(weights_path))
        print("Download complete.")
    else:
        print(f"CRAFT weights found at {weights_path}.")

    return weights_path


# ── CRAFT Model ───────────────────────────────────────────────────────────────

def load_craft_model(weights_path: str | Path, use_cuda: bool = True):
    """
    Load the CRAFT model with pre-trained weights.

    Falls back to the craft_text_detector package if the raw CRAFT-pytorch
    implementation is not installed.

    Args:
        weights_path: Path to craft_mlt_25k.pth.
        use_cuda: Whether to use GPU acceleration.

    Returns:
        Loaded CRAFT model in eval mode.
    """
    import torch

    try:
        from craft_text_detector import load_craftnet_model
        model = load_craftnet_model(cuda=use_cuda and torch.cuda.is_available())
        return model
    except ImportError:
        pass

    # Fallback: manual CRAFT load (requires CRAFT-pytorch source on PYTHONPATH)
    try:
        import sys
        craft_src = Path(__file__).parent.parent / "CRAFT-pytorch"
        if craft_src.exists():
            sys.path.insert(0, str(craft_src))

        from craft import CRAFT  # type: ignore

        model = CRAFT()
        device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        state_dict = torch.load(str(weights_path), map_location=device)

        # Handle DataParallel prefix
        if list(state_dict.keys())[0].startswith("module."):
            from collections import OrderedDict
            new_sd = OrderedDict()
            for k, v in state_dict.items():
                new_sd[k[7:]] = v
            state_dict = new_sd

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        raise RuntimeError(
            "Could not load CRAFT model. Install craft-text-detector: "
            "pip install craft-text-detector\n"
            f"Original error: {e}"
        )


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_text_regions(
    image: Image.Image,
    craft_model=None,
    weights_path: Optional[str | Path] = None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
    use_cuda: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Run CRAFT to detect axis-aligned text line bounding boxes.

    The function uses craft_text_detector's high-level API when available,
    falling back to the raw CRAFT inference loop otherwise.

    Args:
        image: PIL Image of a document page.
        craft_model: Pre-loaded CRAFT model (loaded on first call if None).
        weights_path: Path to CRAFT weights (used if craft_model is None).
        text_threshold: Score map threshold for text region classification.
        link_threshold: Affinity map threshold for character linking.
        low_text: Lower bound text score for region selection.
        canvas_size: Maximum canvas side length for CRAFT inference.
        mag_ratio: Image magnification ratio during inference.
        use_cuda: Use GPU if available.

    Returns:
        List of (x1, y1, x2, y2) axis-aligned bounding boxes in original
        image coordinates, sorted top-to-bottom then left-to-right.
    """
    try:
        from craft_text_detector import get_prediction, empty_cuda_cache
        import torch

        img_np = np.array(image.convert("RGB"))

        if craft_model is None:
            if weights_path is None:
                weights_path = download_craft_weights()
            craft_model = load_craft_model(weights_path, use_cuda=use_cuda)

        prediction_result = get_prediction(
            image=img_np,
            craft_net=craft_model,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            cuda=use_cuda and torch.cuda.is_available(),
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
        )

        # prediction_result["boxes"] contains quadrilateral boxes (4×2 arrays)
        boxes_quad = prediction_result.get("boxes", [])
        boxes = _quads_to_aabb(boxes_quad)
        return _sort_boxes_reading_order(boxes)

    except ImportError:
        # Fallback: connected-component analysis on a rough binarised image
        return _detect_lines_fallback(image)


def _quads_to_aabb(
    quads: List[np.ndarray],
) -> List[Tuple[int, int, int, int]]:
    """Convert CRAFT quadrilateral outputs to axis-aligned bounding boxes."""
    aabb = []
    for quad in quads:
        pts = np.array(quad, dtype=np.float32)
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        aabb.append((x1, y1, x2, y2))
    return aabb


def _sort_boxes_reading_order(
    boxes: List[Tuple[int, int, int, int]],
    line_merge_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Sort boxes in Western reading order: top-to-bottom, left-to-right.

    Boxes whose vertical centres are within `line_merge_threshold × median_height`
    of each other are treated as belonging to the same line.
    """
    if not boxes:
        return []

    heights = [b[3] - b[1] for b in boxes]
    median_h = float(np.median(heights))
    tol = line_merge_threshold * median_h

    # Group by approximate vertical position
    sorted_by_top = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    lines = []
    current_line = [sorted_by_top[0]]
    current_cy = (sorted_by_top[0][1] + sorted_by_top[0][3]) / 2

    for box in sorted_by_top[1:]:
        cy = (box[1] + box[3]) / 2
        if abs(cy - current_cy) <= tol:
            current_line.append(box)
            current_cy = np.mean([(b[1] + b[3]) / 2 for b in current_line])
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            current_cy = cy

    lines.append(sorted(current_line, key=lambda b: b[0]))

    return [box for line in lines for box in line]


# ── Marginalia Filtering ──────────────────────────────────────────────────────

def filter_main_text(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    margin_fraction: float = 0.12,
    min_height: int = 10,
    min_width: int = 50,
    max_height_ratio: float = 0.15,
) -> List[Tuple[int, int, int, int]]:
    """
    Remove marginal annotations and noise boxes, keeping main text body.

    Historical documents often have side notes, folio numbers, and stamps that
    should not be included in the body-text OCR transcript.

    Args:
        boxes: List of (x1, y1, x2, y2) bounding boxes.
        image_shape: (height, width) of the source image.
        margin_fraction: Fraction of image width to treat as outer margin.
        min_height: Minimum box height in pixels.
        min_width: Minimum box width in pixels.
        max_height_ratio: Maximum box height as fraction of image height
                          (removes full-page elements / decorations).

    Returns:
        Filtered list of bounding boxes.
    """
    img_h, img_w = image_shape[:2]
    margin_px = int(img_w * margin_fraction)
    max_h_px = int(img_h * max_height_ratio)

    filtered = []
    for x1, y1, x2, y2 in boxes:
        bw = x2 - x1
        bh = y2 - y1

        # Size filters
        if bh < min_height or bw < min_width:
            continue
        if bh > max_h_px:
            continue

        # Margin filter (both left and right)
        if x1 < margin_px or x2 > img_w - margin_px:
            continue

        filtered.append((x1, y1, x2, y2))

    return filtered


# ── Line Cropping ─────────────────────────────────────────────────────────────

def crop_line_images(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    pad_y: int = 4,
    pad_x: int = 2,
) -> List[Image.Image]:
    """
    Crop individual line images from a page using bounding boxes.

    A small padding is added to avoid cutting off ascenders/descenders.

    Args:
        image: PIL Image of the full page.
        boxes: List of (x1, y1, x2, y2) bounding boxes.
        pad_y: Vertical padding in pixels.
        pad_x: Horizontal padding in pixels.

    Returns:
        List of PIL Image crops, one per text line.
    """
    img_w, img_h = image.size
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1c = max(0, x1 - pad_x)
        y1c = max(0, y1 - pad_y)
        x2c = min(img_w, x2 + pad_x)
        y2c = min(img_h, y2 + pad_y)
        crop = image.crop((x1c, y1c, x2c, y2c))
        crops.append(crop)
    return crops


# ── Column Layout Handling ────────────────────────────────────────────────────

def detect_column_split(
    image: Image.Image,
    min_gap_width: int = 40,
) -> Optional[int]:
    """
    Detect whether the page has a two-column layout.

    Uses a vertical projection profile of a binarised version of the image.
    A wide valley in the centre of the profile indicates a column separator.

    Args:
        image: PIL Image.
        min_gap_width: Minimum width (pixels) to consider a column gap.

    Returns:
        x-coordinate of the column split, or None for single-column.
    """
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection (sum each column)
    proj = binary.sum(axis=0)

    # Smooth to reduce noise
    kernel = np.ones(15, dtype=float) / 15
    proj_smooth = np.convolve(proj, kernel, mode="same")

    img_w = binary.shape[1]
    # Only look in centre 40% of the image
    search_start = int(img_w * 0.3)
    search_end = int(img_w * 0.7)
    centre_proj = proj_smooth[search_start:search_end]

    # Find the minimum (gap region)
    min_idx = int(np.argmin(centre_proj))
    min_val = centre_proj[min_idx]
    mean_val = centre_proj.mean()

    # Gap must be significantly below average
    if min_val > mean_val * 0.3:
        return None

    # Find gap extent
    threshold = mean_val * 0.3
    left = min_idx
    right = min_idx
    while left > 0 and centre_proj[left] < threshold:
        left -= 1
    while right < len(centre_proj) - 1 and centre_proj[right] < threshold:
        right += 1

    gap_width = right - left
    if gap_width < min_gap_width:
        return None

    return search_start + (left + right) // 2


def process_two_column_page(
    image: Image.Image,
    craft_model=None,
    weights_path=None,
    **detect_kwargs,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect text lines in a two-column page, processing each column independently.

    Returns boxes in reading order: all of left column, then all of right column.
    """
    split_x = detect_column_split(image)

    if split_x is None:
        # Single column — process normally
        boxes = detect_text_regions(image, craft_model, weights_path, **detect_kwargs)
        img_arr = np.array(image)
        return filter_main_text(boxes, img_arr.shape)

    img_w, img_h = image.size
    left_col = image.crop((0, 0, split_x, img_h))
    right_col = image.crop((split_x, 0, img_w, img_h))

    left_boxes = detect_text_regions(left_col, craft_model, weights_path, **detect_kwargs)
    right_boxes = detect_text_regions(right_col, craft_model, weights_path, **detect_kwargs)

    # Adjust right column box coordinates to full-page coordinate system
    right_boxes = [(x1 + split_x, y1, x2 + split_x, y2)
                   for (x1, y1, x2, y2) in right_boxes]

    img_arr = np.array(image)
    left_filtered = filter_main_text(left_boxes, np.array(left_col).shape)
    right_filtered = filter_main_text(
        [(x1 - split_x, y1, x2 - split_x, y2) for (x1, y1, x2, y2) in right_boxes],
        np.array(right_col).shape,
    )
    # Re-add offset to right boxes after filtering
    right_filtered = [(x1 + split_x, y1, x2 + split_x, y2)
                      for (x1, y1, x2, y2) in right_filtered]

    return left_filtered + right_filtered


# ── Fallback Detection ────────────────────────────────────────────────────────

def _detect_lines_fallback(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Fallback text line detection via horizontal projection profiling.

    Used when CRAFT weights are unavailable. Significantly less robust but
    sufficient for well-preserved printed documents.
    """
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to merge characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 8:
            boxes.append((x, y, x + w, y + h))

    return _sort_boxes_reading_order(boxes)

"""
Image preprocessing for historical document OCR.

Pipeline: deskew → denoise → binarize (Sauvola) → CLAHE

Each function accepts and returns numpy arrays (uint8, BGR or grayscale)
unless stated otherwise. PIL Images are converted at the entry points.

References:
  - Sauvola & Pietikäinen (2000). "Adaptive document image binarization."
    Pattern Recognition, 33(2), 225–236.
  - Riba et al. (2020). "Document enhancement using visibility graphs."
    arXiv:2002.01076 — motivates CLAHE + contrast normalisation.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_gray(image: Image.Image) -> np.ndarray:
    """Convert PIL Image → grayscale uint8 numpy array."""
    return np.array(image.convert("L"))


def gray_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert grayscale uint8 numpy array → PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def rgb_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert RGB uint8 numpy array → PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ── Deskew ────────────────────────────────────────────────────────────────────

def deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Correct small rotations using a projection-profile approach.

    Sweeps angles in [-max_angle, +max_angle] and picks the rotation that
    maximises the variance of the horizontal projection profile — a signal
    that text lines are well-aligned.

    Args:
        image: Grayscale uint8 numpy array.
        max_angle: Search range in degrees.

    Returns:
        Rotated (deskewed) grayscale image, same shape.
    """
    # Binarise lightly for projection
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -np.inf

    angles = np.arange(-max_angle, max_angle + 0.5, 0.5)
    h, w = binary.shape
    cx, cy = w / 2, h / 2

    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderValue=0)
        # Horizontal projection profile variance
        profile = rotated.sum(axis=1).astype(float)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.3:
        return image  # No meaningful skew detected

    M = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
    deskewed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


# ── Denoising ─────────────────────────────────────────────────────────────────

def denoise(
    image: np.ndarray,
    h: int = 10,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-local means denoising for scanned document images.

    Removes scanner noise while preserving fine strokes, which is important
    for historical typefaces with thin ligatures.

    Args:
        image: Grayscale uint8 numpy array.
        h: Filter strength (larger = more smoothing, less detail).
        template_window: Template patch size in pixels (must be odd).
        search_window: Search window size in pixels (must be odd).

    Returns:
        Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(
        image, h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


# ── Binarisation (Sauvola) ────────────────────────────────────────────────────

def binarize(image: np.ndarray, window_size: int = 51, k: float = 0.2) -> np.ndarray:
    """
    Sauvola adaptive binarisation for document images.

    Outperforms global Otsu on documents with uneven illumination — a common
    artefact in microfilm scans of historical documents.

    T(x,y) = mean(x,y) * [1 + k * (std(x,y)/R - 1)]
    where R = 128 (half the dynamic range).

    Args:
        image: Grayscale uint8 numpy array.
        window_size: Local neighbourhood size in pixels (must be odd, ≥3).
        k: Sensitivity parameter (0.1–0.5; higher = more aggressive).

    Returns:
        Binary image (0 = background, 255 = text).
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    img_float = image.astype(np.float32)

    # Integral images for efficient local mean/variance
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img_float ** 2, -1, (window_size, window_size),
                             normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary = np.where(img_float >= threshold, 255, 0).astype(np.uint8)

    return binary


# ── CLAHE ─────────────────────────────────────────────────────────────────────

def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Enhances local contrast without over-amplifying noise — beneficial for
    faded ink on aged parchment or paper.

    Args:
        image: Grayscale uint8 numpy array.
        clip_limit: Threshold for contrast limiting (OpenCV default: 40.0).
        tile_grid: Number of tiles in (rows, cols) for local processing.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


# ── Full Preprocessing Chain ──────────────────────────────────────────────────

def preprocess_page(
    image: Image.Image,
    deskew_max_angle: float = 10.0,
    denoise_h: int = 10,
    clahe_clip: float = 3.0,
    clahe_tile: Tuple[int, int] = (8, 8),
    binarize_window: int = 51,
    binarize_k: float = 0.2,
    return_gray: bool = True,
) -> Image.Image:
    """
    Apply the full preprocessing chain to a document page.

    Order: RGB → Gray → CLAHE → Deskew → Denoise → Binarize

    CLAHE before deskew improves projection profile accuracy on low-contrast
    documents; denoising before binarisation reduces false positives.

    Args:
        image: PIL Image (any mode).
        deskew_max_angle: Maximum rotation to correct (degrees).
        denoise_h: NL-means filter strength.
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile grid size.
        binarize_window: Sauvola local window size (pixels, odd).
        binarize_k: Sauvola sensitivity.
        return_gray: If True return grayscale; if False return binary.

    Returns:
        Preprocessed PIL Image (mode "L").
    """
    gray = pil_to_gray(image)
    gray = enhance_contrast(gray, clip_limit=clahe_clip, tile_grid=clahe_tile)
    gray = deskew(gray, max_angle=deskew_max_angle)
    gray = denoise(gray, h=denoise_h)
    if return_gray:
        return gray_to_pil(gray)
    binary = binarize(gray, window_size=binarize_window, k=binarize_k)
    return gray_to_pil(binary)


# ── Line-Level Preprocessing ──────────────────────────────────────────────────

def prepare_line_for_trocr(
    line_image: Image.Image,
    target_height: int = 64,
) -> Image.Image:
    """
    Resize and pad a line crop for TrOCR input.

    TrOCR's ViT encoder expects a fixed 384×384 image (handled internally by
    the processor), but pre-scaling line crops to a standard height improves
    consistency and speeds up processor resizing.

    Args:
        line_image: PIL Image of a single text line.
        target_height: Height to scale to (width scales proportionally).

    Returns:
        RGB PIL Image at target_height pixels tall.
    """
    img = line_image.convert("RGB")
    w, h = img.size
    if h == 0:
        return img
    new_w = max(1, int(w * target_height / h))
    img = img.resize((new_w, target_height), Image.LANCZOS)
    return img


# ── Augmentation (training only) ─────────────────────────────────────────────

def get_train_augmentation():
    """
    Build an albumentations augmentation pipeline for TrOCR fine-tuning.

    Transformations are chosen to simulate realistic document degradation:
    - Rotation: covers residual skew after deskewing
    - Gaussian noise: scanner sensor noise
    - Optical distortion / elastic transform: paper warping
    - Blur: focus variation in digitisation

    Returns:
        albumentations.Compose pipeline.
    """
    import albumentations as A
    import albumentations

    # albumentations 2.x changed several APIs — handle both old and new
    major_ver = int(albumentations.__version__.split(".")[0])

    transforms = [
        A.Rotate(limit=3, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ]

    if major_ver >= 2:
        # albumentations 2.x API
        transforms += [
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.OpticalDistortion(distort_limit=0.05, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        ]
    else:
        # albumentations 1.x API
        transforms += [
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        ]

    return A.Compose(transforms)

"""
Zero-shot OCR using Surya (Paruchuri, 2024).

Surya is a document OCR model supporting 90+ languages, including Spanish. It
performs its own text detection (BLLA layout analysis) so CRAFT is not needed
for this inference path. This makes it a clean parallel evaluation track — the
entire pipeline is: page image → Surya → transcribed text.

Reference: Paruchuri, V. (2024). Surya OCR. github.com/datalab-to/surya.
"""

import io
from typing import Dict, List, Optional

from PIL import Image


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_surya_models(device: str = "cuda"):
    """
    Load Surya recognition and detection predictors.

    Args:
        device: 'cuda' or 'cpu'. Surya auto-selects dtype (fp16 on CUDA).

    Returns:
        (rec_predictor, det_predictor) tuple.
    """
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor

    foundation = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    det_predictor = DetectionPredictor()

    return rec_predictor, det_predictor


# ── Single-Page Inference ──────────────────────────────────────────────────────

def predict_page_surya(
    page_image: Image.Image,
    rec_predictor,
    det_predictor,
    langs: Optional[List[str]] = None,
) -> str:
    """
    Transcribe a single page image using Surya.

    Surya runs its own BLLA-based line detection internally — no pre-segmented
    line crops needed. We pass the full page and let Surya return line-level
    TextLine objects, then join them into a single page string.

    Args:
        page_image: PIL Image of the page (RGB or grayscale).
        rec_predictor: Surya RecognitionPredictor instance.
        det_predictor: Surya DetectionPredictor instance.
        langs: Language hint list, e.g. ['es']. None = auto-detect.

    Returns:
        Full page text as a single string (lines joined with '\n').
    """
    if langs is None:
        langs = ["es"]

    # Ensure RGB
    if page_image.mode != "RGB":
        page_image = page_image.convert("RGB")

    # Run recognition (newer Surya API: task_names is task type, not language)
    task = "ocr_with_boxes" if det_predictor is not None else "ocr_without_boxes"
    rec_results = rec_predictor(
        [page_image],
        task_names=[task],
        det_predictor=det_predictor,
    )
    rec_result = rec_results[0]  # RecognitionResult for this page

    # Assemble page text from TextLine objects (sorted top-to-bottom)
    lines = sorted(rec_result.text_lines, key=lambda l: l.bbox[1])
    page_text = "\n".join(line.text for line in lines if line.text.strip())
    return page_text


# ── Multi-Document Inference ───────────────────────────────────────────────────

def predict_all_pages_surya(
    pairs_map: Dict,
    rec_predictor,
    det_predictor,
    langs: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Run Surya inference on all aligned pages across all documents.

    Mirrors the structure of the TrOCR inference loop so results can be
    passed directly to evaluate_predictions().

    Args:
        pairs_map: {doc_name: [(page_img, gt_text), ...]} — aligned pairs.
                   Page images are read directly from this structure.
        rec_predictor: Surya RecognitionPredictor.
        det_predictor: Surya DetectionPredictor.
        langs: Language hint list.

    Returns:
        {doc_name_pN: transcribed_text} keyed the same way as baseline_preds.
    """
    from tqdm import tqdm

    surya_preds: Dict[str, str] = {}

    for doc_name, pairs in pairs_map.items():
        for i, (page_img, _gt_text) in enumerate(
            tqdm(pairs, desc=f"Surya: {doc_name}", leave=False)
        ):
            page_text = predict_page_surya(
                page_img, rec_predictor, det_predictor, langs=langs
            )
            surya_preds[f"{doc_name}_p{i + 1}"] = page_text

    return surya_preds

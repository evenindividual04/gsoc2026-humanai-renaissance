"""
LLM post-processing of OCR output using Gemini 2.5 Flash.

Rationale: Even a well-trained OCR model makes systematic errors on
17th-century Spanish text — broken ligatures, u/v confusion, long-s
misreadings, abbreviation marks, and ink bleed artifacts. A language model
with knowledge of historical Spanish orthography can resolve many of these
errors contextually, without being given explicit rules for every case.

The prompt is engineered to:
  1. Fix OCR artifacts only (never modernise historical spelling)
  2. Preserve archaic orthographic conventions (u/v, ç, long-s)
  3. Handle abbreviation expansion conservatively (mark uncertainty)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert editor specialising in 17th-century Spanish printed documents (impresos del Siglo de Oro). Your task is to correct OCR errors in machine-extracted text.

RULES — follow these exactly:
1. Fix ONLY clear OCR errors: broken characters, split/merged words, wrong letters caused by ink bleed or scanning artefacts.
2. PRESERVE all historical spelling. Do NOT modernise: u/v interchange, ç for z, long-s rendered as ʃ or f, -ado → -ado (not -ao), etc.
3. Do NOT expand abbreviations unless the expansion is certain; leave abbreviated forms as-is.
4. Do NOT add or remove punctuation beyond correcting obvious OCR artefacts.
5. Do NOT re-order words or sentences.
6. Return ONLY the corrected text — no explanations, no preamble, no markdown formatting.

Common OCR errors to watch for:
- 'rn' read as 'm' (or vice versa)
- 'cl' read as 'd'
- 'li' read as 'h'
- 'fi' / 'fl' / 'ff' ligatures broken
- Long-s (ʃ) misread as 'f' — keep context to determine correct reading
- 'u' and 'v' interchangeable — preserve whichever the OCR produced unless clearly wrong
- Hyphenation at line ends: rejoin words split with a hyphen only if clearly erroneous
- Extra spaces inside words caused by ink gaps
"""

USER_PROMPT_TEMPLATE = """Correct the following OCR output from a 17th-century Spanish printed document:

---
{raw_text}
---"""


# ── Gemini Client ─────────────────────────────────────────────────────────────

def setup_gemini(api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
    """
    Configure and return a Gemini generative model client.

    The API key is read from the environment variable GEMINI_API_KEY if not
    provided explicitly.

    Args:
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
        model: Gemini model identifier.

    Returns:
        google.generativeai.GenerativeModel instance.
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY env var or pass api_key."
        )
    genai.configure(api_key=key)

    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "top_p": 0.95,
    }

    return genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT,
    )


# ── Single Correction ─────────────────────────────────────────────────────────

def correct_ocr_output(
    raw_text: str,
    gemini_model,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
) -> str:
    """
    Post-process a single OCR text block with Gemini.

    Args:
        raw_text: OCR output to correct.
        gemini_model: Configured GenerativeModel instance.
        max_retries: Number of retries on API errors.
        retry_sleep: Sleep duration (seconds) between retries.

    Returns:
        Corrected text string. Returns raw_text on failure.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    prompt = USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            corrected = response.text.strip()
            return corrected
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Gemini API error (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(retry_sleep)
            else:
                print(f"[ERROR] Gemini failed after {max_retries} attempts: {e}")
                return raw_text

    return raw_text


# ── Batch Correction ──────────────────────────────────────────────────────────

def batch_correct(
    texts: Dict[str, str],
    gemini_model,
    rate_limit_sleep: float = 2.0,
    cache_file: Optional[str | Path] = None,
) -> Dict[str, str]:
    """
    Apply Gemini correction to a dictionary of {key: raw_text} entries.

    Results are cached to a JSON file to avoid re-processing on reruns.
    The cache is keyed by the input text hash so stale entries are ignored
    if the OCR output changes.

    Args:
        texts: Dictionary mapping document identifiers to raw OCR text.
        gemini_model: Configured GenerativeModel instance.
        rate_limit_sleep: Seconds to wait between API calls.
        cache_file: Path to JSON cache file (None = no caching).

    Returns:
        Dictionary with the same keys, values replaced by corrected text.
    """
    import hashlib

    # Load cache
    cache: Dict[str, str] = {}
    if cache_file is not None:
        cache_file = Path(cache_file)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cache = json.load(f)
                print(f"Loaded {len(cache)} cached corrections.")
            except Exception:
                cache = {}

    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    results = {}
    for key, raw in texts.items():
        ck = _cache_key(raw)
        if ck in cache:
            results[key] = cache[ck]
            continue

        print(f"  Correcting: {key} ({len(raw)} chars)...")
        corrected = correct_ocr_output(raw, gemini_model)
        results[key] = corrected
        cache[ck] = corrected

        time.sleep(rate_limit_sleep)

    # Save updated cache
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    return results


# ── Gemini Vision Transcription ───────────────────────────────────────────────
# This is distinct from the correction path above. Here Gemini Vision acts as
# the OCR engine itself — it receives a line image and returns raw transcribed
# text without any prior OCR output to correct. The prompt is intentionally
# different: "read and transcribe" vs "fix errors in existing text".

VISION_TRANSCRIPTION_PROMPT = """You are an expert palaeographer specialising in 17th-century Spanish printed documents (impresos del Siglo de Oro).

Your task is to TRANSCRIBE the text visible in the image exactly as it appears.

RULES:
1. READ the image and output ONLY the text you see — nothing else.
2. PRESERVE historical orthography exactly: u/v interchange, ç, long-s (ʃ), archaic spellings, abbreviations, ligatures.
3. Do NOT modernise, correct, expand, or interpret the text.
4. Do NOT add punctuation, formatting, or markdown.
5. If a word is partially illegible, transcribe what is visible and use [...] for unreadable sections.
6. Output ONLY the transcribed text — no preamble, no commentary."""


def transcribe_line_vision(
    line_image,
    gemini_model,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
) -> str:
    """
    Transcribe a single line image using Gemini Vision (VLM-as-OCR-engine).

    Unlike correct_ocr_output(), this function has no prior OCR text to
    correct — Gemini Vision reads the raw image and produces the transcription
    from scratch.

    Args:
        line_image: PIL Image of a single text line crop.
        gemini_model: GenerativeModel configured with VISION_TRANSCRIPTION_PROMPT.
        max_retries: Retries on API errors.
        retry_sleep: Sleep between retries (seconds).

    Returns:
        Transcribed text string. Returns '' on failure.
    """
    import google.generativeai as genai
    import io

    # Encode image as PNG bytes
    buf = io.BytesIO()
    if line_image.mode != "RGB":
        line_image = line_image.convert("RGB")
    line_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    image_part = {
        "mime_type": "image/png",
        "data": img_bytes,
    }

    for attempt in range(max_retries):
        try:
            # VISION_TRANSCRIPTION_PROMPT is set as system_instruction on the
            # model (in setup_gemini_vision), so only the image is sent here.
            response = gemini_model.generate_content([image_part])
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Vision API error (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(retry_sleep)
            else:
                print(f"[ERROR] Vision transcription failed after {max_retries} attempts: {e}")
                return ""

    return ""


def setup_gemini_vision(api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
    """
    Configure a Gemini Vision client for direct image transcription.

    Unlike setup_gemini(), the system instruction here is the VISION_TRANSCRIPTION_PROMPT
    (transcribe from image) rather than the text correction prompt.

    Args:
        api_key: Gemini API key.
        model: Gemini model identifier (must support vision input).

    Returns:
        google.generativeai.GenerativeModel instance.
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY env var or pass api_key."
        )
    genai.configure(api_key=key)

    generation_config = {
        "temperature": 0.05,   # Lower temp for faithful transcription
        "max_output_tokens": 512,
        "top_p": 0.9,
    }

    return genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=VISION_TRANSCRIPTION_PROMPT,
    )


def transcribe_page_vision(
    line_images: list,
    gemini_model,
    rate_limit_sleep: float = 2.0,
) -> str:
    """
    Transcribe a full page by sending each line crop to Gemini Vision.

    Args:
        line_images: List of PIL Images — one per detected text line.
        gemini_model: Gemini Vision GenerativeModel instance.
        rate_limit_sleep: Sleep between API calls (seconds) to respect rate limits.

    Returns:
        Full page transcription (lines joined with '\n').
    """
    line_texts = []
    for i, line_img in enumerate(line_images):
        text = transcribe_line_vision(line_img, gemini_model)
        line_texts.append(text)
        if i < len(line_images) - 1:
            time.sleep(rate_limit_sleep)

    return "\n".join(t for t in line_texts if t)


# ── Manuscript (Handwriting) Support ─────────────────────────────────────────

SYSTEM_PROMPT_MANUSCRIPT = """You are an expert palaeographer specialising in 17th–19th century Spanish manuscripts (documentos manuscritos históricos). Your task is to correct OCR errors in machine-extracted text from handwritten documents.

RULES — follow these exactly:
1. Fix ONLY clear recognition errors: misread letter forms, broken words, wrong characters caused by ambiguous handwriting.
2. PRESERVE all historical spelling. Do NOT modernise: u/v interchange, ç, long-s, archaic orthographic conventions.
3. Scribal abbreviations: expand only when the expansion is certain (e.g., nasal tilde ñ over a vowel). Leave uncertain forms as-is.
4. Do NOT add or remove punctuation beyond correcting obvious errors.
5. Do NOT re-order words or sentences.
6. Return ONLY the corrected text — no explanations, no preamble, no markdown formatting.

Common handwriting recognition errors to watch for:
- 'n' and 'u' confusion (scribal n/u ambiguity common in pre-18th-c. Spanish)
- 'i' and 'l' confusion in connected letterforms
- Nasal tildes (˜) over vowels — marks contraction of 'm' or 'n'
- Superscript letters (e.g., q̃ = que, xp̄o = Cristo) — leave abbreviated unless certain
- Long descenders on 'f' and long-s misread as 't' or 'j'
- Line-final flourishes misread as extra letters
- Connected letterforms split into wrong characters
"""

VISION_TRANSCRIPTION_PROMPT_MANUSCRIPT = """You are an expert palaeographer specialising in 17th–19th century Spanish manuscripts (documentos manuscritos históricos).

Your task is to TRANSCRIBE the handwritten text visible in the image exactly as it appears on the page.

RULES:
1. READ the image and output ONLY the text you see — nothing else.
2. PRESERVE historical orthography: u/v interchange, ç, long-s (ʃ), nasal tildes, superscript abbreviations, archaic spellings.
3. Expand scribal abbreviations ONLY when certain (e.g., a tilde clearly marking a dropped nasal). Leave uncertain abbreviations as-is.
4. Do NOT modernise, correct, interpret, or rephrase anything.
5. Do NOT add punctuation or formatting not visible in the image.
6. For illegible sections, use [...] — do NOT guess.
7. Transcribe ALL text including marginalia, headers, and catchwords.
8. Output ONLY the transcribed text — no preamble, no commentary."""


def setup_gemini_manuscript(api_key=None, model: str = "gemini-2.5-flash"):
    """
    Configure a Gemini client for manuscript OCR text correction.

    Uses SYSTEM_PROMPT_MANUSCRIPT (palaeography-focused) rather than the
    printed-document prompt. Intended for post-processing TrOCR or Surya
    output from handwritten pages.
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY env var or pass api_key.")
    genai.configure(api_key=key)

    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "top_p": 0.95,
    }

    return genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=SYSTEM_PROMPT_MANUSCRIPT,
    )


def setup_gemini_vision_manuscript(api_key=None, model: str = "gemini-2.5-flash"):
    """
    Configure a Gemini Vision client for full-page manuscript transcription.

    Uses VISION_TRANSCRIPTION_PROMPT_MANUSCRIPT as system instruction.
    Intended for handwritten pages where line segmentation is unreliable —
    the full page image is sent directly to the VLM.
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY env var or pass api_key.")
    genai.configure(api_key=key)

    generation_config = {
        "temperature": 0.05,
        "max_output_tokens": 4096,
        "top_p": 0.9,
    }

    return genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=VISION_TRANSCRIPTION_PROMPT_MANUSCRIPT,
    )


def transcribe_full_page_vision(
    page_image,
    gemini_model,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
) -> str:
    """
    Transcribe a full manuscript page image using Gemini Vision.

    Unlike transcribe_line_vision(), this sends the entire page rather than
    individual line crops. For handwritten documents, line segmentation via
    CRAFT is unreliable (connected letterforms, variable baseline), so
    bypassing detection and sending the full page to the VLM is more robust.

    The image is resized to at most 2048px on its longest side before sending
    to stay within Gemini's recommended input dimensions while preserving detail.

    Args:
        page_image: PIL Image of the full page.
        gemini_model: GenerativeModel configured with manuscript vision prompt.
        max_retries: Retries on API errors.
        retry_sleep: Sleep between retries (seconds).

    Returns:
        Full page transcription string. Returns '' on failure.
    """
    import io
    from PIL import Image as _Image

    # Resize to max 2048px on longest side
    max_dim = 2048
    w, h = page_image.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        page_image = page_image.resize((new_w, new_h), _Image.LANCZOS)

    if page_image.mode != "RGB":
        page_image = page_image.convert("RGB")

    buf = io.BytesIO()
    page_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    image_part = {
        "mime_type": "image/png",
        "data": img_bytes,
    }

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content([image_part])
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Vision API error (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(retry_sleep)
            else:
                print(f"[ERROR] Full-page vision transcription failed after {max_retries} attempts: {e}")
                return ""

    return ""


def batch_correct_pages(
    page_texts: Dict[str, Dict[int, str]],
    gemini_model,
    rate_limit_sleep: float = 2.0,
    cache_file: Optional[str | Path] = None,
) -> Dict[str, Dict[int, str]]:
    """
    Correct OCR text for all pages across all documents.

    Args:
        page_texts: {doc_name: {page_num: raw_text}}
        gemini_model: Configured GenerativeModel instance.
        rate_limit_sleep: Seconds between API calls.
        cache_file: JSON cache path.

    Returns:
        {doc_name: {page_num: corrected_text}}
    """
    # Flatten to key="{doc}|p{page}" for batch processing
    flat: Dict[str, str] = {}
    for doc, pages in page_texts.items():
        for page_num, text in pages.items():
            flat[f"{doc}|p{page_num}"] = text

    flat_corrected = batch_correct(
        flat, gemini_model,
        rate_limit_sleep=rate_limit_sleep,
        cache_file=cache_file,
    )

    # Unflatten
    corrected: Dict[str, Dict[int, str]] = {}
    for key, text in flat_corrected.items():
        doc, page_str = key.rsplit("|p", 1)
        page_num = int(page_str)
        if doc not in corrected:
            corrected[doc] = {}
        corrected[doc][page_num] = text

    return corrected

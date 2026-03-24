# RenAIssance OCR — GSoC 2026 Evaluation Task

Transformer-based OCR pipeline for 17th-century Spanish printed documents with LLM post-processing.

**Task:** Test I — Printed Documents (6 sources)
**Target metric:** Character Error Rate (CER) ≤ 10% (≥90% accuracy)

---

## Architecture

```
PDF → [PyMuPDF 300 DPI] → Page Images
   → [Preprocessing: CLAHE + deskew + NL-means denoise + Sauvola binarisation]
   → [CRAFT text detector] → Text line bounding boxes
   → [TrOCR (microsoft/trocr-large-printed) + LoRA fine-tuning] → Raw OCR text
   → [Gemini 2.0 Flash LLM] → Corrected OCR text
   → [jiwer / sacrebleu] → CER / WER / BLEU evaluation
```

### Component Justification

| Component | Choice | Why |
|-----------|--------|-----|
| OCR Model | TrOCR `trocr-large-printed` | ViT + transformer decoder — satisfies transformer architecture requirement; SOTA on printed OCR benchmarks (Li et al., 2021) |
| Text Detection | CRAFT (Baek et al., CVPR 2019) | Character-region heatmaps; robust to irregular spacing and historical layouts |
| Fine-tuning | LoRA (r=8, target q+v projections) | ~0.5% trainable params; fits T4 16 GB VRAM; prevents overfitting on small GT set |
| LLM post-processing | Gemini 2.0 Flash | Free API; strong historical Spanish coverage; contextual OCR error correction |
| Public dataset | HTR-United-CREMMA-Medieval | Historical script domain adaptation before task-specific fine-tuning |

---

## Repository Structure

```
evaluation-task/
├── notebook/
│   ├── renaissance_ocr_pipeline.ipynb     # Main notebook (Kaggle/Colab)
│   └── renaissance_ocr_pipeline.pdf       # PDF export with outputs
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # PDF loading, GT parsing, page alignment
│   ├── preprocessing.py    # CLAHE, deskew, denoise, Sauvola binarisation
│   ├── text_detection.py   # CRAFT text detection, marginalia filtering
│   ├── ocr_model.py        # TrOCR loading, LoRA fine-tuning, inference
│   ├── llm_postprocess.py  # Gemini API correction with caching
│   └── evaluation.py       # CER, WER, BLEU + visualisation
├── configs/
│   └── pipeline_config.yaml
├── results/                # Generated metrics (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### Local

```bash
conda create -n renaissance python=3.10
conda activate renaissance
pip install -r requirements.txt
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Kaggle (recommended — free T4 GPU)

1. Upload the `task-dataset/` directory and `src/` as a Kaggle dataset named `renaissance-ocr-data`.
2. Add your `GEMINI_API_KEY` as a Kaggle secret.
3. Open `notebook/renaissance_ocr_pipeline.ipynb`, attach the dataset, enable GPU, and run all cells.

### Google Colab (fallback)

1. Upload `task-dataset/` to Google Drive under `renaissance-ocr/`.
2. Add `GEMINI_API_KEY` as a Colab secret.
3. Open the notebook and run all cells (Runtime → Run all).

---

## Running the Pipeline

```python
# Run the full pipeline end-to-end from the notebook
# Or use the src modules directly:

from src.data_utils import load_all_pdfs, load_all_ground_truth, build_dataset_pairs
from src.preprocessing import preprocess_page
from src.text_detection import load_craft_model, detect_text_regions, filter_main_text, crop_line_images
from src.ocr_model import load_trocr, predict_page
from src.llm_postprocess import setup_gemini, correct_ocr_output
from src.evaluation import evaluate_predictions, aggregate_metrics

# Load data
pdf_map = load_all_pdfs('task-dataset/Test sources/Print', dpi=300)
gt_map  = load_all_ground_truth('task-dataset/Test transcriptions/Print')

# Run preprocessing + detection + OCR + LLM correction
# (see notebook for complete walkthrough)
```

---

## Results

| Stage | CER ↓ | WER ↓ | BLEU ↑ |
|-------|--------|--------|--------|
| Zero-shot TrOCR (baseline) | TBD | TBD | TBD |
| + Domain adaptation + Fine-tuning | TBD | TBD | TBD |
| + Gemini 2.0 Flash | TBD | TBD | TBD |

*Results populated after notebook execution on Kaggle T4.*

---

## References

1. Li, M., et al. (2021). TrOCR: Transformer-based optical character recognition with pre-trained models. *arXiv:2109.10282*.
2. Baek, Y., et al. (2019). Character region awareness for text detection. *CVPR 2019*. *arXiv:1904.01941*.
3. Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*.
4. Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization. *Pattern Recognition*, 33(2), 225–236.
5. Dosovitskiy, A., et al. (2020). An image is worth 16×16 words. *arXiv:2010.11929*.

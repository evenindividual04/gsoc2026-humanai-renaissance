# RenAIssance OCR — GSoC 2026 Evaluation Task

Transformer-based and VLM-based OCR pipeline for 17th–19th century Spanish historical documents, featuring two distinct tracks for printed and handwritten text.

## Tracks & Architecture

### Test I — Printed Documents (6 sources, 1600s)
**Pipeline:**
```text
PDF → [PyMuPDF 300 DPI]
   → [Preprocessing: CLAHE + deskew + NL-means denoise + Sauvola]
   → [CRAFT text detector] → Text line bounding boxes
   → [TrOCR (microsoft/trocr-large-printed) + LoRA] → Raw OCR text
   → [Gemini 2.5 Flash LLM] → Context-aware OCR correction
```

### Test II — Handwritten Documents (5 sources, 1600s–1800s)
**Pipeline:**
Due to CRAFT's limitations on connected historical handwriting, we employ a SOTA Vision-Language Model (VLM) approach bypassing line segmentation:
```text
PDF → [PyMuPDF 300 DPI] 
   → [Gemini 2.5 Flash Vision] → Full-page direct transcription
   → [Gemini 2.5 Flash LLM] → Manuscript abbreviation correction
```

---

## Results Summary

### Test I — Printed
| Stage | CER ↓ | WER ↓ | BLEU ↑ |
|-------|-------|-------|--------|
| Zero-shot TrOCR (baseline) | 1.001 | 1.091 | 20.1 |
| Zero-shot Surya | 0.507 | 0.726 | 74.0 |

### Test II — Handwritten
| Stage | CER ↓ | WER ↓ | BLEU ↑ |
|-------|-------|-------|--------|
| Zero-shot Surya | 0.845 | 1.265 | 41.1 |
| Gemini Vision (full-page) | 0.118 | 0.363 | 84.7 |
| Gemini Vision + Correction | 0.794 | 0.865 | 3.0  |

*(Note: LLM correction on Gemini Vision handwriting output sometimes over-normalised initially accurate historical spellings. See the enclosed PDF report for a detailed ablation study and full context).*

---

## Repository Structure

```text
evaluation-task/
├── notebook/
│   ├── RenAIssance_Evaluation_Task.ipynb        # Main notebook with all pipeline code and outputs
│   └── RenAIssance_Evaluation_Task.pdf # Exported PDF counterpart
├── src/                    # Source code modules (preprocessing, detection, etc.)
├── configs/                # Pipeline configurations
├── RenAIssance_Evaluation_Task_Report.pdf  # Comprehensive 6-page research report
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

### Local Environment
We recommend using Conda to manage the python environment:
```bash
conda create -n renaissance python=3.10
conda activate renaissance
pip install -r requirements.txt
```

Set your Gemini API key (required for LLM post-processing and Test II vision paths):
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### Kaggle (Recommended — Free T4 GPU)
1. Upload the evaluation dataset and `src/` as a Kaggle dataset named `renaissance-ocr-data`.
2. Add your `GEMINI_API_KEY` as a Kaggle secret.
3. Open `notebook/RenAIssance_Evaluation_Task.ipynb`, attach the dataset, enable the GPU accelerator, and run all cells.

### Google Colab (Fallback)
1. Upload the evaluation dataset to Google Drive under `renaissance-ocr/`.
2. Add `GEMINI_API_KEY` as a Colab secret.
3. Open the notebook and run all cells (Runtime → Run all).

---

## Running the Pipeline

The primary entry point is the heavily-commented Jupyter notebook in the `notebook/` directory, which provides an end-to-end execution of both tracks.

Alternatively, the pipeline can be run modularly via the Python source codebase:
```python
from src.data_utils import load_all_pdfs, load_all_ground_truth
from src.preprocessing import preprocess_page
from src.text_detection import load_craft_model, detect_text_regions
from src.ocr_model import load_trocr, predict_page
from src.llm_postprocess import setup_gemini, correct_ocr_output

# Load data
pdf_map = load_all_pdfs('task-dataset/Test sources/Print', dpi=300)
gt_map  = load_all_ground_truth('task-dataset/Test transcriptions/Print')

# Execute preprocessing, CRAFT detection, TrOCR inference, and Gemini LLM post-correction...
# See src/ modules and the notebook for detailed implementation.
```

## References
1. Li, M., et al. (2021). *TrOCR: Transformer-based optical character recognition with pre-trained models*.
2. Baek, Y., et al. (2019). *Character region awareness for text detection*.
3. Wang, P., et al. (2025). *Qwen2.5-VL technical report*. (Informing our VLM-first methodology for Test II).
4. Sauvola, J., & Pietikäinen, M. (2000). *Adaptive document image binarization*.

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
│   └── RenAIssance_Evaluation_Task_0_32013000...pdf # Exported PDF counterpart
├── src/                    # Source code modules (preprocessing, detection, etc.)
├── configs/                # Pipeline configurations
├── RenAIssance_Evaluation_Task_Report.pdf  # Comprehensive 6-page research report
├── requirements.txt        # Python dependencies
└── README.md
```

## Running the Pipeline

See the Jupyter notebook in the `notebook/` directory for the heavily-commented, end-to-end execution of both tracks. The pipeline relies on the Gemini API for post-correction and Test II vision processing:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
Install dependencies with `pip install -r requirements.txt`.

## References
1. Li, M., et al. (2021). *TrOCR: Transformer-based optical character recognition with pre-trained models*.
2. Baek, Y., et al. (2019). *Character region awareness for text detection*.
3. Wang, P., et al. (2025). *Qwen2.5-VL technical report*. (Informing our VLM-first methodology for Test II).
4. Sauvola, J., & Pietikäinen, M. (2000). *Adaptive document image binarization*.

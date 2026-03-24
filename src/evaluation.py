"""
Evaluation metrics for the RenAIssance OCR pipeline.

Metrics:
  - CER  (Character Error Rate) — primary metric, standard for OCR evaluation.
           CER = (S + D + I) / N  where S/D/I = substitutions/deletions/insertions
           at character level, N = reference character count.
  - WER  (Word Error Rate) — same formula at word level.
  - BLEU — n-gram precision score (1–4 grams). Less standard for OCR but
           captures fluency improvements from the LLM post-processing stage.

All metrics are computed using jiwer (CER/WER) and sacrebleu (BLEU) for
reproducibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate between reference and hypothesis strings.

    Args:
        reference: Ground-truth transcription.
        hypothesis: OCR-predicted text.

    Returns:
        CER in [0, ∞). Perfect match = 0.0, all-wrong ≈ 1.0.
    """
    import jiwer

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    return float(jiwer.cer(reference, hypothesis))


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate between reference and hypothesis strings.

    Args:
        reference: Ground-truth transcription.
        hypothesis: OCR-predicted text.

    Returns:
        WER in [0, ∞).
    """
    import jiwer

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    return float(jiwer.wer(reference, hypothesis))


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Corpus BLEU score (sacrebleu) for a single reference–hypothesis pair.

    Note: BLEU is designed for machine translation (multiple references) and
    is less sensitive than CER for OCR. Used here to measure n-gram fluency
    improvements from LLM post-processing.

    Args:
        reference: Ground-truth transcription.
        hypothesis: OCR-predicted text.

    Returns:
        BLEU score in [0, 100] (higher is better).
    """
    import sacrebleu

    if not reference.strip() or not hypothesis.strip():
        return 0.0

    result = sacrebleu.corpus_bleu(
        [hypothesis],
        [[reference]],
        lowercase=True,
        tokenize="char",  # char-level BLEU for morphologically rich language
    )
    return float(result.score)


# ── Pipeline Evaluation ───────────────────────────────────────────────────────

def evaluate_predictions(
    gt: Dict[str, str],
    predictions: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute CER, WER, BLEU for each document in the prediction set.

    Args:
        gt: {doc_key: reference_text}
        predictions: {doc_key: predicted_text}

    Returns:
        {doc_key: {"cer": float, "wer": float, "bleu": float}}
    """
    from data_utils import normalize_for_eval

    results = {}
    for key in gt:
        if key not in predictions:
            print(f"[WARN] No prediction for {key}")
            continue
        ref = normalize_for_eval(gt[key])
        hyp = normalize_for_eval(predictions[key])
        results[key] = {
            "cer": compute_cer(ref, hyp),
            "wer": compute_wer(ref, hyp),
            "bleu": compute_bleu(ref, hyp),
        }
    return results


def aggregate_metrics(per_doc_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute macro-averaged metrics across all documents.

    Args:
        per_doc_metrics: Output of evaluate_predictions().

    Returns:
        {"cer_mean": ..., "cer_std": ..., "wer_mean": ..., "bleu_mean": ...}
    """
    if not per_doc_metrics:
        return {}

    cers = [m["cer"] for m in per_doc_metrics.values()]
    wers = [m["wer"] for m in per_doc_metrics.values()]
    bleus = [m["bleu"] for m in per_doc_metrics.values()]

    return {
        "cer_mean": float(np.mean(cers)),
        "cer_std": float(np.std(cers)),
        "cer_min": float(np.min(cers)),
        "cer_max": float(np.max(cers)),
        "wer_mean": float(np.mean(wers)),
        "wer_std": float(np.std(wers)),
        "bleu_mean": float(np.mean(bleus)),
        "bleu_std": float(np.std(bleus)),
        "n_docs": len(cers),
    }


# ── Ablation Table ────────────────────────────────────────────────────────────

def build_ablation_table(
    gt: Dict[str, str],
    baseline_preds: Dict[str, str],
    finetuned_preds: Optional[Dict[str, str]] = None,
    llm_preds: Optional[Dict[str, str]] = None,
    extra_stages: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Build an ablation table comparing pipeline stages.

    Core stages (always evaluated in this order):
      1. Zero-shot TrOCR   — trocr-large-printed, no fine-tuning
      3. Fine-tuned TrOCR  — after domain adaptation + task fine-tuning
      4. Fine-tuned TrOCR + Gemini 2.5 Flash — fine-tuned + Gemini text correction

    Additional stages can be injected via extra_stages and are merged by their
    key prefix:
      "2. Zero-shot Surya" — Surya zero-shot parallel evaluation
      "5. Gemini Vision"   — Gemini Vision direct transcription (VLM-as-OCR)

    Args:
        gt: Ground truth dictionary.
        baseline_preds: Predictions from zero-shot TrOCR.
        finetuned_preds: Predictions after fine-tuning.
        llm_preds: Predictions after Gemini text post-processing.
        extra_stages: Optional {stage_name: preds_dict} entries to include.
                      Example: {"2. Zero-shot Surya": surya_preds,
                                "5. Gemini Vision": gv_preds}

    Returns:
        DataFrame with columns: [Stage, Document, CER, WER, BLEU].
        Each stage has one row per document plus an AGGREGATE summary row.
    """
    rows = []

    def _add_stage(preds, stage_name):
        if preds is None:
            return
        per_doc = evaluate_predictions(gt, preds)
        agg = aggregate_metrics(per_doc)
        for doc, metrics in per_doc.items():
            rows.append({
                "Stage": stage_name,
                "Document": doc,
                "CER": f"{metrics['cer']:.4f}",
                "WER": f"{metrics['wer']:.4f}",
                "BLEU": f"{metrics['bleu']:.2f}",
            })
        rows.append({
            "Stage": stage_name,
            "Document": "AGGREGATE",
            "CER": f"{agg.get('cer_mean', 0):.4f} ± {agg.get('cer_std', 0):.4f}",
            "WER": f"{agg.get('wer_mean', 0):.4f} ± {agg.get('wer_std', 0):.4f}",
            "BLEU": f"{agg.get('bleu_mean', 0):.2f} ± {agg.get('bleu_std', 0):.2f}",
        })

    # Merge all stages and sort by numeric prefix so order is deterministic
    all_stages: Dict[str, Optional[Dict[str, str]]] = {
        "1. Zero-shot TrOCR": baseline_preds,
        "3. Fine-tuned TrOCR": finetuned_preds,
        "4. Fine-tuned TrOCR + Gemini 2.5 Flash": llm_preds,
    }
    if extra_stages:
        all_stages.update(extra_stages)

    for stage_name in sorted(all_stages):
        _add_stage(all_stages[stage_name], stage_name)

    return pd.DataFrame(rows)


# ── Error Analysis ────────────────────────────────────────────────────────────

def error_analysis(
    reference: str,
    hypothesis: str,
    n_examples: int = 10,
) -> Dict[str, List]:
    """
    Categorise common OCR failure modes by aligning reference and hypothesis.

    Categories:
      - substitutions: wrong character
      - deletions: character missing from hypothesis
      - insertions: extra character in hypothesis

    Args:
        reference: Ground-truth text.
        hypothesis: OCR output.
        n_examples: Maximum examples per category.

    Returns:
        Dict with "substitutions", "deletions", "insertions" lists.
    """
    import jiwer

    ops = jiwer.process_characters(reference, hypothesis)

    substitutions = []
    deletions = []
    insertions = []

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    ref_i, hyp_i = 0, 0

    for chunk in ops.alignments[0]:
        if chunk.type == "equal":
            ref_i += chunk.ref_end_idx - chunk.ref_start_idx
            hyp_i += chunk.hyp_end_idx - chunk.hyp_start_idx
        elif chunk.type == "replace":
            r = "".join(ref_chars[chunk.ref_start_idx: chunk.ref_end_idx])
            h = "".join(hyp_chars[chunk.hyp_start_idx: chunk.hyp_end_idx])
            substitutions.append({"reference": r, "hypothesis": h})
        elif chunk.type == "delete":
            r = "".join(ref_chars[chunk.ref_start_idx: chunk.ref_end_idx])
            deletions.append({"reference": r})
        elif chunk.type == "insert":
            h = "".join(hyp_chars[chunk.hyp_start_idx: chunk.hyp_end_idx])
            insertions.append({"hypothesis": h})

    return {
        "substitutions": substitutions[:n_examples],
        "deletions": deletions[:n_examples],
        "insertions": insertions[:n_examples],
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_ablation(ablation_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot CER comparison across pipeline stages as a grouped bar chart.

    Args:
        ablation_df: Output of build_ablation_table().
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    # Filter aggregate rows only
    agg = ablation_df[ablation_df["Document"] == "AGGREGATE"].copy()
    agg["CER_val"] = agg["CER"].str.extract(r"^([\d.]+)").astype(float)
    agg["CER_err"] = agg["CER"].str.extract(r"± ([\d.]+)").astype(float).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    stages = agg["Stage"].tolist()
    x = np.arange(len(stages))

    import matplotlib.cm as cm
    n = len(stages)
    colors = [cm.tab10(i / max(n, 1)) for i in range(n)]

    bars = ax.bar(
        x, agg["CER_val"], yerr=agg["CER_err"],
        capsize=5, color=colors,
        alpha=0.85, width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Character Error Rate (lower is better)", fontsize=11)
    ax.set_title("OCR Pipeline Ablation — CER by Stage", fontsize=13)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, max(agg["CER_val"]) * 1.3 + 0.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ablation plot to {save_path}")

    plt.show()


def plot_per_document_cer(
    per_doc_metrics: Dict[str, Dict[str, float]],
    stage_label: str = "Fine-tuned + LLM",
    save_path: Optional[str] = None,
):
    """
    Bar chart of CER per document for a single pipeline stage.

    Args:
        per_doc_metrics: Output of evaluate_predictions().
        stage_label: Label for the plot title.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    docs = list(per_doc_metrics.keys())
    cers = [per_doc_metrics[d]["cer"] for d in docs]

    # Shorten document names for display
    short_names = [d[:30] + "..." if len(d) > 30 else d for d in docs]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(short_names, cers, color="#4C72B0", alpha=0.85)
    ax.set_ylabel("CER", fontsize=11)
    ax.set_title(f"Per-Document CER — {stage_label}", fontsize=12)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, max(cers) * 1.3 + 0.02)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ── Report Saving ─────────────────────────────────────────────────────────────

def save_evaluation_report(
    results: Dict,
    output_path: str | Path,
):
    """Save evaluation results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation report saved to {output_path}")

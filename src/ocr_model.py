"""
TrOCR-based OCR model for historical printed documents.

Architecture: Vision Transformer (ViT) encoder + autoregressive transformer
decoder (BEiT/RoBERTa decoder). Pre-trained on large synthetic and real OCR
datasets (SROIE, IAM, etc.) and fine-tuned here on 17th-century Spanish text.

Reference:
  Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., ... & Wei, F.
  (2021). TrOCR: Transformer-based optical character recognition with
  pre-trained models. arXiv:2109.10282.

Fine-tuning uses LoRA (Low-Rank Adaptation) to reduce memory requirements on
free-tier T4 GPUs:
  Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... &
  Chen, W. (2021). LoRA: Low-rank adaptation of large language models.
  arXiv:2106.09685.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    AutoProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_trocr(
    model_name: str = "microsoft/trocr-large-printed",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """
    Load TrOCR processor and model with optional LoRA adapters.

    The `trocr-large-printed` checkpoint is pre-trained on printed text
    and provides the best starting point for historical printed documents.

    Args:
        model_name: HuggingFace model identifier.
        use_lora: Whether to add LoRA adapters for memory-efficient fine-tuning.
        lora_r: LoRA rank (trade-off: higher r → more capacity, more params).
        lora_alpha: LoRA scaling factor (effective scale = alpha / r).
        lora_dropout: Dropout applied to LoRA layers.
        lora_target_modules: Attention projection layers to adapt.
        device: Target device ("cpu", "cuda", or None for auto).

    Returns:
        (processor, model) ready for training or inference.
    """
    if lora_target_modules is None:
        # TrOCR encoder (ViT/BEiT) uses "query"/"value"
        # TrOCR decoder (RoBERTa) uses "q_proj"/"v_proj"
        # Include both so LoRA adapts the full encoder-decoder stack
        lora_target_modules = ["query", "value", "q_proj", "v_proj"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading TrOCR: {model_name} on {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, low_cpu_mem_usage=False)

    # Configure decoder generation settings
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    # Generation parameters are passed directly to generate() in predict_page()

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("[WARN] peft not installed — training without LoRA.")

    model = model.to(device)
    return processor, model


# ── Dataset ───────────────────────────────────────────────────────────────────

class HistoricalOCRDataset(Dataset):
    """
    PyTorch Dataset for TrOCR fine-tuning on historical document line images.

    Each item is a (line_image, transcription) pair. The processor handles
    image encoding (ViT feature extraction) and label tokenisation.

    Args:
        line_images: List of PIL Images (one per text line).
        texts: Corresponding transcription strings.
        processor: TrOCRProcessor instance.
        augment: Whether to apply training augmentations.
        max_target_length: Maximum tokenised label length.
    """

    def __init__(
        self,
        line_images: List[Image.Image],
        texts: List[str],
        processor: TrOCRProcessor,
        augment: bool = False,
        max_target_length: int = 128,
    ):
        assert len(line_images) == len(texts), "Images and texts must be paired."
        self.images = line_images
        self.texts = texts
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment

        if augment:
            from preprocessing import get_train_augmentation
            self.transform = get_train_augmentation()
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image = self.images[idx].convert("RGB")
        text = self.texts[idx]

        if self.transform is not None:
            img_np = np.array(image)
            augmented = self.transform(image=img_np)
            image = Image.fromarray(augmented["image"])

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so cross-entropy ignores it
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune_trocr(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_dataset: HistoricalOCRDataset,
    val_dataset: HistoricalOCRDataset,
    output_dir: str = "./checkpoints",
    num_epochs: int = 10,
    learning_rate: float = 5e-6,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    fp16: bool = True,
    save_steps: int = 50,
    eval_steps: int = 50,
    early_stopping_patience: int = 3,
) -> VisionEncoderDecoderModel:
    """
    Fine-tune TrOCR using Seq2SeqTrainer with CER-based early stopping.

    Effective batch size = batch_size × gradient_accumulation_steps × num_GPUs.
    With batch_size=2 and accumulation=8: effective batch = 16 on a single T4.

    Args:
        model: TrOCR model (possibly with LoRA adapters).
        processor: TrOCRProcessor.
        train_dataset: Training set.
        val_dataset: Validation set.
        output_dir: Directory for checkpoints.
        num_epochs: Maximum training epochs.
        learning_rate: Peak learning rate for AdamW.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_steps: Linear warmup steps.
        weight_decay: L2 regularisation coefficient.
        fp16: Mixed precision training.
        save_steps: Checkpoint saving interval.
        eval_steps: Evaluation interval.
        early_stopping_patience: Stop if val CER does not improve for this many
                                  evaluations.

    Returns:
        Fine-tuned model (best checkpoint loaded).
    """
    import evaluate as hf_evaluate

    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 in labels (padding)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_strs, references=label_strs)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=10,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()
    return trainer.model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_line(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    line_image: Image.Image,
    device: Optional[str] = None,
) -> str:
    """
    Run TrOCR inference on a single line image.

    Args:
        model: TrOCR model in eval mode.
        processor: TrOCRProcessor.
        line_image: PIL Image of one text line.
        device: Inference device.

    Returns:
        Decoded text string.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    pixel_values = processor(
        line_image.convert("RGB"), return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def predict_page(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    line_images: List[Image.Image],
    batch_size: int = 16,
    device: Optional[str] = None,
) -> str:
    """
    Run TrOCR inference on a list of line images and assemble page text.

    Lines are processed in batches for efficiency. Results are joined with
    newlines to reconstruct the page layout.

    Args:
        model: TrOCR model in eval mode.
        processor: TrOCRProcessor.
        line_images: List of line crop PIL Images (in reading order).
        batch_size: Number of lines per inference batch.
        device: Inference device.

    Returns:
        Full page text with lines separated by newlines.
    """
    if device is None:
        device = next(model.parameters()).device

    # Unwrap DataParallel before calling generate() — DataParallel only wraps
    # forward(), so its gather step corrupts ModelOutput dicts and produces
    # outputs=None inside the transformers generation loop.
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base.eval()
    all_texts = []

    for i in range(0, len(line_images), batch_size):
        batch = [img.convert("RGB") for img in line_images[i: i + batch_size]]
        pixel_values = processor(
            batch, return_tensors="pt", padding=True
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = base.generate(pixel_values)

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_texts.extend(texts)

    return "\n".join(all_texts)


# ── Public Dataset Loading ────────────────────────────────────────────────────

def load_public_historical_dataset(
    dataset_name: str = "biglam/europeana_newspapers",
    max_samples: Optional[int] = 2000,
    processor: Optional[TrOCRProcessor] = None,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[HistoricalOCRDataset, HistoricalOCRDataset]:
    """
    Load a public historical OCR dataset from HuggingFace for domain adaptation.

    The Europeana Newspapers dataset contains historical printed newspaper lines
    from multiple European languages and centuries — the closest freely available
    proxy for 17th-century Spanish printed documents (similar typefaces, period
    orthography, paper degradation artefacts).

    Alternatives if unavailable: `bjoernp/trocr-lines-validation` (printed lines),
    or any ICDAR historical document dataset via HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier.
        max_samples: Limit samples for T4 time budget (None = all).
        processor: TrOCRProcessor (loaded from default model if None).
        val_split: Fraction reserved for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset) as HistoricalOCRDataset instances.
    """
    from datasets import load_dataset

    if processor is None:
        processor = AutoProcessor.from_pretrained("microsoft/trocr-large-printed")

    print(f"Loading public dataset: {dataset_name} (streaming, max {max_samples} samples)")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if max_samples:
        ds = ds.take(max_samples)

    # Materialise only the samples we need (avoids loading full parquet)
    img_col = txt_col = None

    def _get_image_and_text(example):
        nonlocal img_col, txt_col
        if img_col is None:
            img_col = next((c for c in ["image", "img", "scan"] if c in example), None)
            txt_col = next((c for c in ["text", "transcription", "gt", "label"] if c in example), None)
            if img_col is None or txt_col is None:
                print(f"  [WARN] Unknown columns: {list(example.keys())} — skipping")
                return None, None
        return example[img_col], example[txt_col]

    samples = []
    for ex in ds:
        img, txt = _get_image_and_text(ex)
        if img is not None and txt is not None and txt and txt.strip():
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            samples.append((img, txt.strip()))

    import random
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_split)
    val_pairs = samples[:n_val]
    train_pairs = samples[n_val:]

    train_images, train_texts = zip(*train_pairs) if train_pairs else ([], [])
    val_images, val_texts = zip(*val_pairs) if val_pairs else ([], [])

    print(f"Public dataset: {len(train_images)} train, {len(val_images)} val samples.")

    return (
        HistoricalOCRDataset(train_images, train_texts, processor, augment=True),
        HistoricalOCRDataset(val_images, val_texts, processor, augment=False),
    )


def save_model(
    model,
    processor: TrOCRProcessor,
    output_dir: str | Path,
):
    """
    Save model and processor to disk.

    If the model has LoRA adapters (PeftModel), merges them into the base
    weights before saving so the result is loadable as a standard
    VisionEncoderDecoderModel without requiring peft at inference time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge LoRA weights if present
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            print("LoRA adapters merged into base model for saving.")
    except ImportError:
        pass

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


def load_finetuned_model(
    model_dir: str | Path,
    device: Optional[str] = None,
) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load a fine-tuned TrOCR model from disk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = str(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()
    return processor, model

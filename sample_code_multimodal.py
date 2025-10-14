# sample_code_multimodal.py
import os
import argparse
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import ViltProcessor
from model_multimodal import ViltPriceRegressor

# -------------------------
# Configuration (edit if needed)
# -------------------------
OUTPUT_DIR = './outputs_multimodal'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
DEFAULT_TEST_CSV = './dataset/test.csv'
OUT_CSV = './dataset/test_out_multimodal.csv'
VILT_MODEL = 'dandelin/vilt-b32-mlm'
TEST_IMAGES_ROOT = './TestImages'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keep same text duplication used in training for consistency
DEFAULT_TEXT_DUPLICATION = 2

# -------------------------
# Helpers
# -------------------------
def clean_catalog_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = re.sub(r'Item\s*Name\s*:\s*', '', text, flags=re.IGNORECASE)
    s = re.sub(r'[,\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def load_image_for_sample(sample_id, images_root=TEST_IMAGES_ROOT):
    for ext in ('jpg', 'jpeg', 'png'):
        p = os.path.join(images_root, f"{sample_id}.{ext}")
        if os.path.exists(p):
            try:
                return Image.open(p).convert('RGB')
            except Exception:
                continue
    # fallback: white image
    return Image.new('RGB', (224, 224), (255, 255, 255))

def find_checkpoint(provided_path=None):
    """
    Return a checkpoint path to use. Priority:
      1) provided_path (if not None)
      2) outputs_multimodal/checkpoints/best.pt
      3) outputs_multimodal/checkpoints/latest.pt
      4) outputs_multimodal/vilt_price_model.pt
      5) None (no model found)
    """
    if provided_path:
        if os.path.exists(provided_path):
            return provided_path
        else:
            raise FileNotFoundError(f"Provided checkpoint {provided_path} not found.")
    # Try best -> latest -> model file
    candidates = [
        os.path.join(CHECKPOINT_DIR, 'best.pt'),
        os.path.join(CHECKPOINT_DIR, 'latest.pt'),
        os.path.join(OUTPUT_DIR, 'vilt_price_model.pt')
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_model_and_processor(checkpoint_path=None, vilt_model_name=VILT_MODEL, device=DEVICE):
    # Processor: prefer saved in OUTPUT_DIR
    if os.path.isdir(OUTPUT_DIR) and (os.path.exists(os.path.join(OUTPUT_DIR, 'processor_config.json')) or os.path.exists(os.path.join(OUTPUT_DIR, 'feature_extractor_config.json'))):
        print(f"[INFO] Loading processor from {OUTPUT_DIR}")
        processor = ViltProcessor.from_pretrained(OUTPUT_DIR)
    else:
        print(f"[INFO] Loading processor from hub '{vilt_model_name}'")
        processor = ViltProcessor.from_pretrained(vilt_model_name)

    # Model skeleton
    model = ViltPriceRegressor(vilt_model_name=vilt_model_name, dropout=0.2, hidden_dims=(512,128))
    # Find checkpoint
    ckpt = checkpoint_path or find_checkpoint(None)
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found. Train first or provide --checkpoint.")
    print(f"[INFO] Loading checkpoint from: {ckpt}")

    # torch.load may return either a state_dict or a full checkpoint dict
    loaded = torch.load(ckpt, map_location=device)
    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        state_dict = loaded['model_state_dict']
    else:
        # assume it's a raw state_dict or model saved directly
        state_dict = loaded

    # try to load state_dict to model (may need strict=False if shapes mismatch)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print("[WARNING] strict load failed, trying non-strict load:", e)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded to device: {device}")
    return model, processor

# -------------------------
# Predictor
# -------------------------
_model = None
_processor = None

def predictor(sample_id, catalog_content, checkpoint=None, use_log_target=True, text_duplication=DEFAULT_TEXT_DUPLICATION):
    global _model, _processor
    if _model is None or _processor is None:
        ckpt = checkpoint or find_checkpoint(None)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found. Train first or provide --checkpoint.")
        _model, _processor = load_model_and_processor(checkpoint_path=ckpt, device=DEVICE)

    text = clean_catalog_text(catalog_content)
    if text_duplication > 1:
        # duplicate text to bias textual dominance consistent with training
        text = (" " + text).join([""] * text_duplication).strip() or text

    image = load_image_for_sample(sample_id)
    enc = _processor(text=[text], images=[image], return_tensors='pt', padding='max_length', truncation=True)
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    pixel_values = enc['pixel_values'].to(DEVICE)

    with torch.no_grad():
        pred_log = _model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    pred_val = float(pred_log.cpu().item())
    if use_log_target:
        price = float(np.expm1(pred_val))
    else:
        price = pred_val
    if price < 0:
        price = 0.0
    return round(price, 2)

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default=DEFAULT_TEST_CSV, help='Path to test.csv with sample_id and catalog_content')
    parser.add_argument('--out_csv', type=str, default=OUT_CSV, help='Output CSV path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load (overrides default search)')
    parser.add_argument('--text_duplication', type=int, default=DEFAULT_TEXT_DUPLICATION, help='Text duplication used at inference (default same as training)')
    parser.add_argument('--use_log_target', action='store_true', help='If provided, invert log1p transform (default behavior). Use --no-log to disable')
    parser.add_argument('--no-log', dest='use_log_target', action='store_false')
    args = parser.parse_args()

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV {args.test_csv} not found")

    # Try to find checkpoint if not provided
    checkpoint_to_use = args.checkpoint or find_checkpoint(None)
    if checkpoint_to_use is None:
        raise FileNotFoundError("No checkpoint found in outputs_multimodal. Provide --checkpoint PATH or run training first.")
    print(f"[INFO] Using checkpoint: {checkpoint_to_use}")

    # We'll lazy-load model/processor in predictor when needed
    rows = []
    df = pd.read_csv(args.test_csv)
    if 'sample_id' not in df.columns or 'catalog_content' not in df.columns:
        raise ValueError("test.csv must contain 'sample_id' and 'catalog_content' columns")

    print(f"[INFO] Running inference for {len(df)} rows; outputs -> {args.out_csv}")
    results = []
    for i, row in df.iterrows():
        sid = row['sample_id']
        text = row['catalog_content']
        try:
            price = predictor(sid, text, checkpoint=checkpoint_to_use, use_log_target=args.use_log_target, text_duplication=args.text_duplication)
        except Exception as e:
            print(f"[WARN] Prediction failed for sample {sid}: {e}")
            price = 0.0
        results.append((sid, price))
        if (i + 1) % 100 == 0:
            print(f"[INFO] Predicted {i+1}/{len(df)}")

    out_df = pd.DataFrame(results, columns=['sample_id', 'price'])
    out_df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved predictions to {args.out_csv}")

if __name__ == '__main__':
    main()

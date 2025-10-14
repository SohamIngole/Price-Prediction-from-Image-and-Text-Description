import os
import random
import math
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, ViltProcessor
from torch.optim import AdamW

from model_multimodal import ViltPriceRegressor
from dataset_multimodal import MultimodalPriceDataset

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_PATH = './dataset/train.csv'             # CSV must be in working dir
TRAIN_IMAGES_ROOT = './TrainImages'   # folder with images named <sample_id>.jpg
OUTPUT_DIR = './outputs_multimodal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
VILT_MODEL = 'dandelin/vilt-b32-mlm'

# requested preference - will be capped to model/tokenizer max
PREFERRED_MAX_LEN = 64

BATCH_SIZE = 8            # tune based on GPU memory
EPOCHS = 10               # longer runs -> more checkpoint opportunities
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_LOG_TARGET = True
USE_AMP = True

# Choose num_workers: 0 for Windows-safe; otherwise increase
NUM_WORKERS = 0

# duplicate text to bias text dominance (1 = no duplication)
TEXT_DUPLICATION = 2

# ---------- Checkpoint settings ----------
# Save every N seconds (set to None to disable periodic saves)
SAVE_INTERVAL_SECONDS = 300  # 5 minutes; adjust to your preference
# Also save every N training steps (optional)
SAVE_EVERY_N_STEPS = None    # set to an int if you prefer step-based interval

# Name patterns
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pt")
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best.pt")

# ---------------------------
# Utilities
# ---------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(preds, targets):
    return math.sqrt(((preds - targets) ** 2).mean())

def smape(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    denom = (np.abs(targets) + np.abs(preds)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(preds[mask] - targets[mask]) / denom[mask]) * 100.0)

def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on most OSes

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_rmse, path):
    """
    Save a training checkpoint containing model + optimizer + scheduler + scaler + training state.
    """
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_rmse': best_val_rmse,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'timestamp': time.time()
    }
    try:
        atomic_save(state, path)
        print(f"[Checkpoint] Saved checkpoint to {path}")
    except Exception as e:
        print(f"[Checkpoint] Failed to save {path}: {e}")

def load_checkpoint_if_exists(path, model, optimizer=None, scheduler=None, scaler=None, device=DEVICE):
    """
    Load checkpoint if it exists and return (start_epoch, global_step, best_val_rmse).
    If not exists, returns (0, 0, inf).
    """
    if not os.path.exists(path):
        return 0, 0, float('inf')
    print(f"[Checkpoint] Loading checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = int(ckpt.get('epoch', 0))
    global_step = int(ckpt.get('global_step', 0))
    best_val_rmse = float(ckpt.get('best_val_rmse', float('inf')))
    print(f"[Checkpoint] Resuming from epoch={start_epoch}, global_step={global_step}, best_val_rmse={best_val_rmse:.4f}")
    return start_epoch, global_step, best_val_rmse

# ---------------------------
# Main
# ---------------------------
def main():
    seed_everything()

    # Load CSV
    df = pd.read_csv(DATA_PATH)
    if 'sample_id' not in df.columns:
        raise ValueError("train.csv must contain 'sample_id' column")
    if 'catalog_content' not in df.columns:
        raise ValueError("train.csv must contain 'catalog_content' column")

    target_col = 'price' if 'price' in df.columns else 'target'
    df[target_col] = pd.to_numeric(df.get(target_col, 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    if USE_LOG_TARGET:
        df[target_col] = np.log1p(df[target_col].values)

    # Shuffle + splits
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f"Samples: {n}, train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Processor
    processor = ViltProcessor.from_pretrained(VILT_MODEL)
    try:
        proc_max_len = int(processor.tokenizer.model_max_length)
    except Exception:
        proc_max_len = PREFERRED_MAX_LEN
    max_text_len = min(PREFERRED_MAX_LEN, proc_max_len)
    if max_text_len != PREFERRED_MAX_LEN:
        print(f"[Info] Adjusted MAX_LEN to processor limit: {max_text_len}")

    # Datasets (raw text + PIL images)
    train_ds = MultimodalPriceDataset(train_df,
                                      sample_id_col='sample_id',
                                      text_col='catalog_content',
                                      target_col=target_col,
                                      images_root=TRAIN_IMAGES_ROOT,
                                      is_train=True,
                                      text_duplication=TEXT_DUPLICATION)
    val_ds = MultimodalPriceDataset(val_df,
                                    sample_id_col='sample_id',
                                    text_col='catalog_content',
                                    target_col=target_col,
                                    images_root=TRAIN_IMAGES_ROOT,
                                    is_train=True,
                                    text_duplication=TEXT_DUPLICATION)
    test_ds = MultimodalPriceDataset(test_df,
                                     sample_id_col='sample_id',
                                     text_col='catalog_content',
                                     target_col=target_col,
                                     images_root=TRAIN_IMAGES_ROOT,
                                     is_train=True,
                                     text_duplication=TEXT_DUPLICATION)

    # collate_fn (uses processor to produce batched tensors)
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        enc = processor(text=texts, images=images, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=max_text_len)
        out = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'pixel_values': enc['pixel_values']
        }
        if 'labels' in batch[0] and batch[0].get('labels') is not None:
            out['labels'] = torch.stack([item['labels'] for item in batch], dim=0)
        return out

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=RandomSampler(train_ds),
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_ds),
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_ds),
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = ViltPriceRegressor(vilt_model_name=VILT_MODEL, dropout=0.2, hidden_dims=(512,128))
    model.to(DEVICE)

    # Optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and torch.cuda.is_available()) else None

    # Resume from latest checkpoint if present
    start_epoch, global_step, best_val_rmse = load_checkpoint_if_exists(LATEST_CKPT, model, optimizer, scheduler, scaler, device=DEVICE)

    last_save_time = time.time()
    # If checkpoint present, set last_save_time to now so we don't immediately re-save
    if os.path.exists(LATEST_CKPT):
        last_save_time = time.time()

    print(f"[Start] Starting training from epoch {start_epoch}, global_step {global_step}, best_val_rmse {best_val_rmse}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                    loss = loss_fn(preds, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_losses.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))
            global_step += 1
            pbar.set_postfix(train_loss=np.mean(train_losses), global_step=global_step)

            # --- Periodic checkpoint by time or steps ---
            now = time.time()
            do_save = False
            if SAVE_INTERVAL_SECONDS is not None and (now - last_save_time) >= SAVE_INTERVAL_SECONDS:
                do_save = True
            if SAVE_EVERY_N_STEPS is not None and SAVE_EVERY_N_STEPS > 0 and (global_step % SAVE_EVERY_N_STEPS == 0):
                do_save = True

            if do_save:
                ckpt_path = LATEST_CKPT
                try:
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_rmse, ckpt_path)
                    # also save processor files so inference can load from OUTPUT_DIR
                    try:
                        processor.save_pretrained(OUTPUT_DIR)
                    except Exception as e:
                        print(f"[Checkpoint] Warning: failed to save processor to {OUTPUT_DIR}: {e}")
                    last_save_time = now
                except Exception as e:
                    print(f"[Checkpoint] Exception during periodic save: {e}")

        # End epoch validation
        model.eval()
        val_preds_log = []
        val_trues_log = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                else:
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

                val_preds_log.append(preds.detach().cpu().numpy())
                val_trues_log.append(labels.detach().cpu().numpy())

        if len(val_preds_log) == 0:
            print("[Warning] Validation produced no predictions; skipping metrics")
            continue

        val_preds_log = np.concatenate(val_preds_log)
        val_trues_log = np.concatenate(val_trues_log)

        if USE_LOG_TARGET:
            val_preds = np.expm1(val_preds_log)
            val_trues = np.expm1(val_trues_log)
        else:
            val_preds = val_preds_log
            val_trues = val_trues_log

        cur_rmse = rmse(val_preds, val_trues)
        cur_mae = np.mean(np.abs(val_preds - val_trues))
        cur_smape = smape(val_preds, val_trues)
        print(f"Epoch {epoch+1} validation RMSE: {cur_rmse:.4f}  MAE: {cur_mae:.4f}  SMAPE: {cur_smape:.2f}%")

        # Save epoch-level latest checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_rmse, LATEST_CKPT)
        processor.save_pretrained(OUTPUT_DIR)

        # Save best model if improved
        if cur_rmse < best_val_rmse:
            best_val_rmse = cur_rmse
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_rmse, BEST_CKPT)
            # also save a user-friendly model file at OUTPUT_DIR root
            try:
                model_path = os.path.join(OUTPUT_DIR, 'vilt_price_model.pt')
                torch.save(model.state_dict(), model_path)
                print(f"[Best Model] Saved best model state to {model_path}")
            except Exception as e:
                print(f"[Best Model] failed to save model.pt: {e}")

    # Final evaluation on test split
    print("Evaluating on held-out test split...")
    # load best or latest? we'll evaluate using latest saved weights
    # If you want to evaluate best, change path to BEST_CKPT and load accordingly.
    # For now use model in memory (trained to completion)
    model.eval()
    test_preds_log = []
    test_trues_log = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            else:
                preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            test_preds_log.append(preds.detach().cpu().numpy())
            test_trues_log.append(labels.detach().cpu().numpy())

    if len(test_preds_log) == 0:
        print("[Warning] Test produced no predictions; exiting")
        return

    test_preds_log = np.concatenate(test_preds_log)
    test_trues_log = np.concatenate(test_trues_log)

    if USE_LOG_TARGET:
        test_preds = np.expm1(test_preds_log)
        test_trues = np.expm1(test_trues_log)
    else:
        test_preds = test_preds_log
        test_trues = test_trues_log

    test_rmse = rmse(test_preds, test_trues)
    test_mae = np.mean(np.abs(test_preds - test_trues))
    test_smape = smape(test_preds, test_trues)
    print(f"Test RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  SMAPE: {test_smape:.2f}%")

if __name__ == '__main__':
    main()
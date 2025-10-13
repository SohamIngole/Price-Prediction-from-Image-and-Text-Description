import os
import random
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW

from BERTmodel import BertPriceRegressor
from dataset import PriceCatalogDataset

DATA_PATH = './dataset/train.csv'
OUTPUT_DIR = './outputs_price_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BERT_MODEL = 'bert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_LOG_TARGET = True
USE_AMP = True

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
    denominator = (np.abs(targets) + np.abs(preds)) / 2.0
    # To avoid division by zero
    mask = denominator != 0
    smape_val = np.mean(np.abs(preds[mask] - targets[mask]) / denominator[mask]) * 100
    return smape_val

def main():
    seed_everything()

    # 1) Load data
    df = pd.read_csv(DATA_PATH)
    assert 'catalog_content' in df.columns, "catalog_content column missing"
    target_col = 'price' if 'price' in df.columns else 'target'
    if target_col not in df.columns:
        raise ValueError("No price/target column found in train.csv")

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0.0).clip(lower=0.0)

    if USE_LOG_TARGET:
        df[target_col] = np.log1p(df[target_col].values)

    # 2) Train/Val/Test split (80/10/10)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"Total samples: {n}, train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # 3) Tokenizer + Datasets
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    train_ds = PriceCatalogDataset(train_df, tokenizer, target_col=target_col, max_length=MAX_LEN, is_train=True)
    val_ds = PriceCatalogDataset(val_df, tokenizer, target_col=target_col, max_length=MAX_LEN, is_train=True)
    test_ds = PriceCatalogDataset(test_df, tokenizer, target_col=target_col, max_length=MAX_LEN, is_train=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=RandomSampler(train_ds), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_ds))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(test_ds))

    # 4) Model
    model = BertPriceRegressor(bert_model_name=BERT_MODEL, dropout=0.2, hidden_dims=(512,128))
    model.to(DEVICE)

    # 5) Optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and torch.cuda.is_available()) else None

    # 6) Training loop
    best_val_rmse = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(preds, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            pbar.set_postfix(loss=sum(train_losses)/len(train_losses))

        # Validation
        model.eval()
        val_preds_log = []
        val_trues_log = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        preds = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    preds = model(input_ids=input_ids, attention_mask=attention_mask)

                val_preds_log.append(preds.detach().cpu().numpy())
                val_trues_log.append(labels.detach().cpu().numpy())

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

        # Save best model
        if cur_rmse < best_val_rmse:
            best_val_rmse = cur_rmse
            model_path = os.path.join(OUTPUT_DIR, 'bert_price_model.pt')
            torch.save(model.state_dict(), model_path)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Saved best model to {model_path} (RMSE {best_val_rmse:.4f})")

    print("Evaluating on held-out test split...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'bert_price_model.pt'), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    test_preds_log = []
    test_trues_log = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds_log.append(preds.detach().cpu().numpy())
            test_trues_log.append(labels.detach().cpu().numpy())

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

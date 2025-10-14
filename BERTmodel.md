# 🧠 Amazon ML Challenge 2025 — NLP-Based Price Prediction

**Team Name:** Code Walkers  
**Team Members:** Sankalp Jain, Soham Ingole, Sparsh Goyal  
**Submission Date:** 12/10/15, 22:21

## 📘 Overview
This project predicts the **target price** of products using only the textual feature **`catalog_content`**, which contains the item name, attributes, and size information.  
A fine-tuned **BERT-based transformer model** (with a feed-forward regression head) learns contextual representations from text and predicts product prices.

---

## 📂 Dataset

**File:** `train.csv`  
**Rows:** ~75,000  

| Column | Description |
|---------|-------------|
| `ID` | Unique identifier |
| `image_link` | Product image URL *(not used in this task)* |
| `catalog_content` | Product name + descriptive features + size |
| `price` | Target price (float) |

---

## ⚙️ Data Preprocessing

Performed via `clean_catalog_text()` in `dataset.py`:
- Removes `"Item Name:"` prefix  
- Normalizes punctuation & whitespace  
- Converts to lowercase  
- Retains numeric values and units (`8oz`, `30 Ounce`, etc.)

---

## 🧠 Model Architecture

### 🔹 1. Transformer Backbone
Pretrained **BERT** model (`bert-base-uncased`), optionally replaceable by smaller or newer models like:
- `distilbert-base-uncased`
- `microsoft/deberta-v3-base`
- `albert-base-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

### 🔹 2. Regression Head (Fully Connected Network)

- Linear(768 → 512) → ReLU → Dropout(0.2)
- Linear(512 → 128) → ReLU → Dropout(0.2)
- Linear(128 → 1)
- Defined dynamically using `hidden_dims=(512,128)` in **`model.py`**.

### 🔹 3. End-to-End Fine-Tuning
- All BERT and regressor weights are **trainable**
- Gradients propagate through the full model
- No freezing — **true fine-tuning**

---

## ⚙️ Training Configuration

| Setting | Value |
|----------|--------|
| Batch size | 32 |
| Max sequence length | 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Optimizer | `torch.optim.AdamW` |
| Scheduler | Linear warmup & decay |
| Epochs | 4 |
| Loss | MSE on log-transformed price |
| Log target transform | `log1p(price)` |
| Mixed precision (AMP) | ✅ Enabled |
| Validation split | 10% |
| Test split | 10% |

---
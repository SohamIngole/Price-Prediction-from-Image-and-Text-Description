# 🧠 Multimodal Price Prediction using ViLT (Vision-and-Language Transformer)

## 📘 Overview
This project implements an **end-to-end multimodal deep learning system** for predicting the **price of catalog items** based on both:

- **Textual descriptions** (the `catalog_content` field)
- **Visual features** (product images)

We use a **Vision-and-Language Transformer (ViLT)** backbone that jointly learns representations of text and image features through **cross-attention**, enabling the model to capture the semantic relationships between what an item looks like and how it’s described.

The model is fine-tuned for **regression** (predicting price) with additional preprocessing, a custom loss pipeline, and robust checkpointing.

---

## 🔍 Key Concepts

### 🧩 What is Cross-Attention?
In multimodal learning, **cross-attention** allows one modality (e.g., text) to attend to information from another modality (e.g., image).  
This enables contextual fusion of representations — the model learns *how words describe parts of the image*.

For example:
> “Lavazza Gran Riserva Filtro Ground Coffee, Dark Roast 8oz Brick”  
> Text tokens like “Dark Roast” or “8oz” can attend to visual features corresponding to dark color and packaging size.

### 💡 How ViLT Works
**ViLT (Vision-and-Language Transformer)** integrates text and image features using a single Transformer encoder — it’s a more efficient modern alternative to dual-stream models like ViLBERT or LXMERT.

- **Text tokens** (from BERT-style embeddings)
- **Image patches** (from a linear projection of visual features)
- Both are concatenated and processed jointly through multi-head attention.
- **Cross-attention emerges** naturally between text and image tokens inside self-attention layers.

This architecture:
- Removes the need for heavy image encoders (like ResNet/CNN)
- Uses *less than 120M parameters* — far smaller than 8B+ models
- Has native support via Hugging Face Transformers (`dandelin/vilt-b32-mlm`)

---

## 🧠 Model Architecture

### 1. **Base Encoder**
- Pretrained ViLT backbone (`dandelin/vilt-b32-mlm`)
- Outputs contextualized multimodal embeddings

### 2. **Regression Head**
- Fully connected feed-forward layers on top of the [CLS] token
- Example configuration:
  ```python
  hidden_dims = (512, 128)
  dropout = 0.2
  output_dim = 1  # price prediction
- implements log-target regression (log(price + 1)) for stable training

### 3. Loss Function
- MSELoss (Mean Squared Error) on log-transformed targets
- Evaluation metrics:
- - RMSE (Root Mean Square Error)
- - MAE (Mean Absolute Error)
- - SMAPE (Symmetric Mean Absolute Percentage Error)

---

## ⚙️ Training Pipeline (`train_multimodal.py`)

### ✅ Features

- Fine-tunes **ViLT** on text–image pairs  
- Uses **Automatic Mixed Precision (AMP)** for GPU efficiency  
- Applies **log-transform** on target prices (`log1p`)  
- Implements **periodic checkpointing**  
- Automatically resumes from `latest.pt` if interrupted  
- Evaluates and saves **best model based on RMSE**

---

### 🧾 Checkpointing System

| File | Purpose |
|------|----------|
| `outputs_multimodal/checkpoints/latest.pt` | Latest checkpoint (autosaved every few minutes) |
| `outputs_multimodal/checkpoints/best.pt` | Best-performing model by validation RMSE |
| `outputs_multimodal/vilt_price_model.pt` | Simplified model weights for inference |
| `outputs_multimodal/` | Stores tokenizer/processor configs for easy reloading |

---

### 🚀 Example CLI Run

- ```bash
- python train_multimodal.py

---
## 🔮 Inference Pipeline (`sample_code_multimodal.py`)

### ✅ Features

- Auto-detects **best / latest / model weights** from `outputs_multimodal`
- Loads product **images** from `TestImages/<sample_id>.jpg`
- Cleans and duplicates **catalog text** for stronger textual dominance
- Automatically inverts **log-transform** (`expm1`) for final price predictions
- Generates output file **`test_out_multimodal.csv`** with predicted prices

---

### 🧭 Usage

- Run inference with default configuration (auto-selects best available model):

- ```bash
- python sample_code_multimodal.py

---
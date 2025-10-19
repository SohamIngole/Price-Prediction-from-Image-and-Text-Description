import os
import pandas as pd
import torch
from transformers import BertTokenizerFast
from BERTmodel import BertPriceRegressor
import numpy as np

OUTPUT_DIR = './outputs_price_model'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'bert_price_model.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path=MODEL_PATH, bert_model_name='bert-base-uncased'):
    tokenizer = BertTokenizerFast.from_pretrained(OUTPUT_DIR)
    model = BertPriceRegressor(bert_model_name=bert_model_name, dropout=0.2, hidden_dims=(512,128))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

# Preprocessing same as training
import re
def clean_catalog_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = re.sub(r'Item\s*Name\s*:\s*', '', text, flags=re.IGNORECASE)
    s = re.sub(r'[,\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

_model = None
_tokenizer = None

def predictor(sample_id, catalog_content, image_link=None, use_log_target=True):

    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model()

    text = clean_catalog_text(catalog_content)
    enc = _tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        pred_log = _model(input_ids=input_ids, attention_mask=attention_mask)

    pred_log_val = float(pred_log.cpu().item())
    if use_log_target:
        price = float(np.expm1(pred_log_val))  # invert log1p
    else:
        price = pred_log_val

    if price < 0:
        price = 0.0
    return round(price, 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row.get('image_link', None), use_log_target=True),
        axis=1
    )
    output_df = test[['sample_id', 'price']]
    output_filename = os.path.join(DATASET_FOLDER, 'test_out_BERT.csv')
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")

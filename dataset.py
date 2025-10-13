import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

def clean_catalog_text(text: str) -> str:
    """
    Basic cleaning for catalog_content:
    - remove "Item Name:" token if present
    - normalize whitespace and punctuation
    - lowercase (we use uncased BERT, so lowercase is fine)
    - keep numeric tokens (useful for size/weight)
    """
    if not isinstance(text, str):
        return ""
    s = text
    # Remove common label
    s = re.sub(r'Item\s*Name\s*:\s*', '', s, flags=re.IGNORECASE)
    # Replace multiple commas/whitespace with single space
    s = re.sub(r'[,\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

class PriceCatalogDataset(Dataset):
    def __init__(self, df, tokenizer: BertTokenizerFast, text_col='catalog_content', target_col='price',
                 max_length=64, is_train=True):
        """
        df: pandas DataFrame containing at least 'catalog_content' and target
        tokenizer: HuggingFace tokenizer
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.target_col = target_col
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = row.get(self.text_col, "")
        text = clean_catalog_text(raw)

        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }

        if self.is_train:
            target = float(row[self.target_col])
            item['labels'] = torch.tensor(target, dtype=torch.float)
        return item

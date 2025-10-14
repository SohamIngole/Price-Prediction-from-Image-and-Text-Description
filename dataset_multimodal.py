# dataset_multimodal.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re

def default_image(size=(224,224)):
    return Image.new('RGB', size, (255,255,255))

def clean_catalog_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = re.sub(r'Item\s*Name\s*:\s*', '', text, flags=re.IGNORECASE)
    s = re.sub(r'[,\t]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

class MultimodalPriceDataset(Dataset):
    """
    Now returns raw items:
      - 'text' : cleaned text (str)
      - 'image': PIL.Image
      - 'labels': torch.tensor (if is_train)
    Collation and processor batching happens in the DataLoader's collate_fn.
    """
    def __init__(self, df: pd.DataFrame,
                 sample_id_col='sample_id', text_col='catalog_content', target_col='price',
                 images_root='TrainImages', is_train=True, text_duplication=1):
        self.df = df.reset_index(drop=True)
        self.sample_id_col = sample_id_col
        self.text_col = text_col
        self.target_col = target_col
        self.images_root = images_root
        self.is_train = is_train
        self.text_duplication = max(1, int(text_duplication))

    def __len__(self):
        return len(self.df)

    def _image_path_for_id(self, sample_id):
        fname = f"{sample_id}.jpg"
        p = os.path.join(self.images_root, fname)
        if os.path.exists(p):
            return p
        p2 = os.path.join(self.images_root, f"{sample_id}.png")
        if os.path.exists(p2):
            return p2
        return None

    def _load_image(self, sample_id):
        p = self._image_path_for_id(sample_id)
        if not p:
            return default_image()
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = default_image()
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row[self.sample_id_col]
        raw_text = row.get(self.text_col, "")
        text = clean_catalog_text(raw_text)
        if self.text_duplication > 1:
            # duplicate text to bias text dominance
            text = (" " + text).join([""] * self.text_duplication).strip() or text

        image = self._load_image(sample_id)

        item = {
            'sample_id': sample_id,
            'text': text,
            'image': image
        }

        if self.is_train:
            target = float(row[self.target_col]) if (self.target_col in row and pd.notna(row[self.target_col])) else 0.0
            item['labels'] = torch.tensor(target, dtype=torch.float)

        return item

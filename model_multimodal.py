import torch
import torch.nn as nn
from transformers import ViltModel

class ViltPriceRegressor(nn.Module):
    """
    ViLT-based multimodal regressor:
      - ViltModel accepts raw images + text via ViltProcessor
      - small FCC regressor on top of pooled CLS embedding
    """
    def __init__(self, vilt_model_name='dandelin/vilt-b32-mlm', dropout=0.2, hidden_dims=(512,128)):
        super().__init__()
        self.vilt = ViltModel.from_pretrained(vilt_model_name)
        hidden_size = self.vilt.config.hidden_size  # typically 768

        layers = []
        in_dim = hidden_size
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        """
        Forward pass accepts:
          - input_ids: (batch, seq_len)
          - attention_mask: (batch, seq_len)
          - pixel_values: (batch, 3, H, W)
        """
        outputs = self.vilt(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_dict=True)

        # use pooler_output when available else use first token embedding
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS-like token

        out = self.regressor(pooled).squeeze(-1)
        return out
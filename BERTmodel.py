import torch
import torch.nn as nn
from transformers import BertModel

class BertPriceRegressor(nn.Module):
    """
    BERT encoder + small feed-forward head for regression (price prediction).
    """
    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.2, hidden_dims=(512, 128)):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        layers = []
        in_dim = bert_hidden
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # final reg output

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.pooler_output  # shape (batch, hidden)
        out = self.regressor(pooled).squeeze(-1)  # shape (batch,)
        return out

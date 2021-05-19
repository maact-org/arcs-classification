import torch.nn
from torch import nn
from transformers import BertModel
import numpy as np


class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, name):
        super(SentimentClassifier, self).__init__()
        self.name = name
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        self.probability = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        dect = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = dect['pooler_output']
        dropped = self.drop(pooled_output)
        output = self.out(dropped)
        return self.probability(output).squeeze(-1)

    def get_sentiment_distribution(self, data_loader, device):
        self.eval()
        predictions = []
        preds = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                predictions += [outputs]
        for pre in predictions:
            preds += [pre.cpu().numpy()]
        return np.concatenate(preds)

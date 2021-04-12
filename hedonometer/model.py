import torch.nn
from torch import nn
from transformers import BertModel
import numpy as np


class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        dect = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = dect['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)

    def get_sentiment_distribution(self, data_loader, device):
        self.eval()
        predictions = []
        distribution = nn.Sigmoid()
        preds = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                dist = distribution(outputs)
                predictions += [dist]
        for pre in predictions:
            preds += [pre.cpu().numpy()]
        preds = np.asarray(preds)
        data = np.vstack((pred for pred in preds))
        return data

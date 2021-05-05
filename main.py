from hedonometer.model import SentimentClassifier
from transformers import BertTokenizer
import settings as st
import torch

sc = SentimentClassifier(st.BERT_MODEL_PATH, "sentimentis")
sc.load_state_dict(torch.load(st.SC_MODEL_PATH, map_location=torch.device("cpu")))

# Creating a tokenizer
tokenizer = BertTokenizer.from_pretrained(st.BERT_MODEL_PATH, do_lower_case=False)
tokenized = tokenizer("Estou muito feliz com meu gatinho")

input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

tokenized = tokenizer("o monstro no meu armário é horrível")
input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

from hedonometer.model import SentimentClassifier
from transformers import BertTokenizer
import settings as st
import torch
import utils

sc = SentimentClassifier(st.BERT_MODEL_PATH, "sentimentis")
sc.load_state_dict(torch.load(st.SC_MODEL_PATH, map_location=torch.device("cpu")))

# Creating a tokenizer
tokenizer = BertTokenizer.from_pretrained(st.BERT_MODEL_PATH, do_lower_case=False)
tokenized = tokenizer("Estou muito feliz com meu gatinho")

books = utils.build_dataset_from_folders(st.DATA_SET_EVAL_PATH)
for book in books:
    print(utils.get_score_from_sentiment_model(book, tokenizer))

input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

tokenized = tokenizer("o monstro no meu armário é horrível")
input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

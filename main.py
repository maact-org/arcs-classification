from hedonometer.model import SentimentClassifier
from transformers import BertTokenizer
import settings as st
import torch
import utils
from os import path
import logging

sc = SentimentClassifier(st.BERT_MODEL_PATH, "sentimentis")
sc.load_state_dict(torch.load(st.SC_MODEL_PATH, map_location=torch.device("cpu")))

# Creating a tokenizer
tokenizer = BertTokenizer.from_pretrained(st.BERT_MODEL_PATH, do_lower_case=False)

books_df = utils.build_dataset_from_folders(st.DATA_SET_EVAL_PATH)

if path.exists(st.DATALOADER_TOKENIZED_PATH):
    logging.info("Loading tokenized dataloader...")
    tokenized_dl = torch.load(st.DATALOADER_TOKENIZED_PATH)
else:
    logging.info("Generating a new dataloader...")
    tokenized_dl = utils.get_prepared_dataset(books_df, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
    logging.info("Saving tokenized dataloader...")
    torch.save(tokenized_dl, st.DATALOADER_TOKENIZED_PATH)

for (idx, row) in enumerate(tokenized_dl):
    print(row)
    # print(utils.get_score_from_sentiment_model(book, tokenizer))
    break

input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

tokenized = tokenizer("o monstro no meu armário é horrível")
input_ids = torch.Tensor(tokenized.input_ids).view(1, -1).int()
attention_mask = torch.Tensor(tokenized.attention_mask).view(1, -1).int()
sentiment = sc(input_ids, attention_mask)
print(sentiment)

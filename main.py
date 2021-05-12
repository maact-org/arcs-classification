from hedonometer.model import SentimentClassifier
from transformers import BertTokenizer
import settings as st
import torch
import utils
from os import path
import logging
from arcs import ArcModel
from os import walk
import pandas as pd
import numpy as np
import tensorflow as tf
import training as trn
import evaluation as evl


logging.basicConfig(encoding='utf-8', level=logging.INFO)

# sc = SentimentClassifier(st.BERT_MODEL_PATH, "sentimentis")
# sc.load_state_dict(torch.load(st.SC_MODEL_PATH, map_location=torch.device("cpu")))
#
# # Creating a tokenizer
# tokenizer = BertTokenizer.from_pretrained(st.BERT_MODEL_PATH, do_lower_case=False)
#
books_df = utils.build_dataset_from_folders(st.DATA_SET_EVAL_PATH)
books_df.to_csv("books_arcs.csv")

#
# if path.exists(st.DATALOADER_TOKENIZED_PATH):
#     logging.info("Loading tokenized dataloader...")
#     tokenized_dl = torch.load(st.DATALOADER_TOKENIZED_PATH)
# else:
#     logging.info("Generating a new dataloader...")
#     tokenized_dl = utils.get_prepared_dataset(books_df, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
#     logging.info("Saving tokenized dataloader...")
#     torch.save(tokenized_dl, st.DATALOADER_TOKENIZED_PATH)


model = ArcModel(input_shape=(33, 1))
# model.load_model(file='data/arcs_classifier.h5')
model.model = trn.train_model_with_csv(
                        model.model,
                        file_path="training/arcs.csv",
                        epochs=500)

model.save_model(file='data/arcs_classifier.h5')


f = []
for (dirpath, dirnames, filenames) in walk('series'):
    f.extend(filenames)
    break

tags_df = pd.read_csv('data/books.csv')[['book', 'tag']]
tags_df = tags_df.drop_duplicates().reset_index(drop=True)
prepared_df = pd.DataFrame()

for i in range(len(tags_df)):
    df = pd.read_csv('data/series/book_{}.csv'.format(i))
    serie = df['value'].to_numpy()
    prepared_df = prepared_df.append({"book": i, "serie": serie, "tag": tags_df['tag'][i]}, ignore_index=True)

# outputs_df.to_csv("outputs.csv")

# Evaluate model
evl_df = evl.evaluate_df(model, prepared_df)
logging.info(evl_df.describe())
evl_df.to_csv("evaluation.csv")

print(evl_df['accuracy'].mean())
print(evl_df['loss'].mean())

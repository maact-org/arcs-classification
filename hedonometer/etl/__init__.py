
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import string
import warnings

warnings.filterwarnings("ignore")


def get_label_encoder():
    """
    Returns a label encoder that maps 'positive' to 1 and 'negative to 0.
    :return: A label encoder for the classes in the metabooks dataset
    """
    le = LabelEncoder()
    le.classes_ = np.array(['negative', 'positive', 'F', 'FRF', 'FR', 'RFR', 'RF', 'R'])
    return le


def get_prepared_dataset(df: pd.DataFrame, tokenizer, max_len, batch_size):
    """
    Creates a data loader with a dataset of tokenized data, labeled and with the original text
    :param df: Data Frame containing fields MainDescription and class_id
    :param tokenizer: tokenizer for preprocessing
    :param max_len: Max sequence length
    :param batch_size: Batch size
    :return: A data loader for training or validation
    """

    le = get_label_encoder()
    df['tag'] = le.transform(df.tag)

    ds = BooksDataSet(
        df=df,
        label_encoder=le,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
    )


class BooksDataSet(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len, label_encoder=None):
        self.text = df.text
        self.tags = df.tag
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = label_encoder
        self.n_classes = len(self.label_encoder.classes_)
        oh_enc = OneHotEncoder()
        self.one_hot_encoder = oh_enc.fit([[i] for i in range(self.n_classes)])
        self._clean()

    def __len__(self):
        return len(self.tokenized_df)

    def __getitem__(self, item):
        input_ids = self.tokenized_df.input_ids[item]
        attention_mask = self.tokenized_df.attention_mask[item]
        tag = self.tokenized_df.tag[item]
        one_hot_tag = self.tokenized_df.one_hot_tag[item]
        result = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "tag": [tag],
            "one_hot_tag": [one_hot_tag],
        }
        return result

    def _clean(self):
        encoding_function = lambda text: self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        tokenized_df = pd.DataFrame(columns=['input_ids', 'attention_mask', 'tag'])
        for index, row in self.df.iterrows():
            tokenized_sentences = [encoding_function(i) for i in row['text'].split('\n') if i]
            input_ids = [i.input_ids for i in tokenized_sentences]
            attention_mask = [i.attention_mask for i in tokenized_sentences]
            tag = row['tag']
            tag_to_encode = np.array(tag).reshape(-1, 1)
            one_hot_tag = self.one_hot_encoder.transform(tag_to_encode)
            current_df = pd.DataFrame({'tag': [tag],
                                       'one_hot_tag': [one_hot_tag],
                                       'text': [row['text']]
                                       })
            tokenized_df = pd.concat([tokenized_df, current_df])
        tokenized_df = tokenized_df.reset_index(drop=True)

        self.tokenized_df = tokenized_df

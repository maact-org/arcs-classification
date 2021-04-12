from os.path import exists

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

import warnings

warnings.filterwarnings("ignore")


def get_label_encoder(df: pd.DataFrame = None, le_path=""):
    """
    If the file LABEL_ENCODER_PATH exists it loads this file as the label encoder and returns it, otherwise it fit a
    label encoder to the given data frame, saves it and returns it.
    :param df: metabooks dataset
    :param le_path: Path of the label encoder
    :return: A label encoder for the classes in the metabooks dataset
    """
    le = LabelEncoder()
    if exists(le_path):
        print("Using local le")
        le.classes_ = np.load(le_path, allow_pickle=True)
    else:
        print("New label encoder")
        le.fit(df.code)
        np.save(le_path, le.classes_)
    return le


def get_prepared_dataset(df: pd.DataFrame, tokenizer, max_len, batch_size, le_path=""):
    """
    Creates a data loader with a dataset of tokenized data, labeled and with the original text
    :param df: Data Frame containing fields MainDescription and class_id
    :param tokenizer: tokenizer for preprocessing
    :param max_len: Max sequence length
    :param batch_size: Batch size
    :return: A data loader for training or validation
    """

    le = get_label_encoder(df, le_path=le_path)
    df['class_id'] = le.transform(df.code)

    ds = TweeterDataSet(
        texts=df.text,
        targets=df.sentiment_id,
        label_encoder=le,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class TweeterDataSet(Dataset):
    def __init__(self, texts: pd.DataFrame, tokenizer, max_len, targets=None, label_encoder=None, n_classes=12):
        self.texts = texts
        self.targets = targets
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_classes = n_classes
        self._clean()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoded = self.encoded[item]
        result = {
            'text': text,
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
        }
        if self.targets is not None:
            target = self.targets[item]
            one_hot = self.one_hot[item]
            result['targets'] = torch.tensor(target, dtype=torch.long)
            result['one_hot'] = torch.tensor(one_hot, dtype=torch.int)
        return result

    def _clean(self):

        oh_enc = OneHotEncoder()

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
        encoded = self.texts.apply(encoding_function)
        self.encoded = encoded.values
        self.texts = self.texts.values
        self.targets = self.targets.values

        targets_for_one_hot = self.targets.reshape(-1, 1)
        oh_enc.fit([[i] for i in range(self.n_classes)])
        self.one_hot = oh_enc.transform(targets_for_one_hot).toarray()
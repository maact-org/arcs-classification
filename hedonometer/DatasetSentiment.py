from torch.utils.data import Dataset, DataLoader
import hedonometer.settings as st
from torch import nn, tensor
import torch

class DatasetSentiments(Dataset):
    def __init__(self, tokenizer, df):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            tokenizer (huggingface Tokenizer): Will encode str sentences
        """
        self.df = df
        self.n_labels = self.df["arc"].astype("category").cat.categories.size
        self.size = len(self.df)
        self.tokenizer = tokenizer
        self._clean()
    def __len__(self):
        return int(self.size)
    def __df__(self):
        return self.df
    def __getitem__(self, idx):
        return {
        "input_ids":self.input_ids[idx],
        "token_type_id":self.token_type_ids[idx],
        "attention_mask": self.attention_mask[idx],
        }
    def _clean(self):
        tokenizer = self.tokenizer
        tokenized = self.tokenizer(self.df["text"].tolist(), return_tensors='pt', max_length=st.MAX_LEN, pad_to_max_length=True)
        self.input_ids = tensor(tokenized['input_ids'], dtype=torch.int64)
        self.input_ids = torch.split(self.input_ids, 33)
        self.token_type_ids = tokenized['token_type_ids']
        self.attention_mask = tokenized['attention_mask']
        self.attention_mask = torch.split(self.attention_mask, 33)
        print(self.input_ids)
        print("attention mask: ",self.attention_mask)
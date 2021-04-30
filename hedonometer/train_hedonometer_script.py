import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer, AdamW

import settings as st
from hedonometer.etl import get_prepared_dataset
from hedonometer.model import SentimentClassifier
from hedonometer.train import train_model

if __name__ == "__main__":
    df = pd.read_csv(st.DATA_SET_PATH)

    df_raw_train, df_raw_test = train_test_split(
        df,
        test_size=0.1,
        random_state=st.RANDOM_SEED
    )

    # Creating a tokenizer
    tokenizer = BertTokenizer.from_pretrained(st.PRE_TRAINED_MODEL, do_lower_case=False)

    # Splitting the tests
    train_data_loader = get_prepared_dataset(df_raw_train, tokenizer, st.MAX_LEN, st.BATCH_SIZE)
    test_data_loader = get_prepared_dataset(df_raw_test, tokenizer, st.MAX_LEN, st.BATCH_SIZE)

    # Creating a model
    model = SentimentClassifier(st.PRE_TRAINED_MODEL, 'sentiment_m_cleaned_double_out')
    model.to(st.DEVICE)

    # Defining an optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    # Defining a loss function
    loss_fn = nn.BCELoss().to(st.DEVICE)

    # Training a model
    train_model(model, loss_fn, optimizer, train_data_loader, test_data_loader)

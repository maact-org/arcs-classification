import numpy as np
import pandas as pd
from transformers import BertTokenizer

from arcs import ArcModel
from hedonometer.model import SentimentClassifier
import hedonometer.settings as st
from hedonometer.etl import get_prepared_dataset
import tensorflow as tf

SPLIT_SIZE = 6

tf.config.set_visible_devices([], 'GPU')


class Analyzer:

    def __init__(self, hedonometer_path: str, tokenizer_path: str):
        self.hedonometer = SentimentClassifier(st.PRE_TRAINED_MODEL, 'hedonometer')
        self.hedonometer.to(st.DEVICE)
        self.arc_model = ArcModel(input_shape=(33, 1))
        self.arc_model.load_model()
        self.tokenizer = BertTokenizer.from_pretrained(st.PRE_TRAINED_MODEL, do_lower_case=False)

    def __split_book__(self, book: str):
        book_words = book.split(' ')
        book_sentences_words = [
            book_words[i:i + SPLIT_SIZE] for i in
            range(0, len(book_words) - SPLIT_SIZE, SPLIT_SIZE)
        ]
        book_sentences = [
            ' '.join(sentence) for sentence in
            book_sentences_words
        ]
        return book_sentences

    def __calculate_book_arc__(self, book: str):
        book_sentences = self.__split_book__(book)
        df = pd.DataFrame({
            'text': book_sentences
        })
        data_loader = get_prepared_dataset(
            df,
            tokenizer=self.tokenizer,
            max_len=st.MAX_LEN,
            batch_size=st.BATCH_SIZE
        )

        sentiment_series = self.hedonometer.get_sentiment_distribution(
            data_loader=data_loader,
            device=st.DEVICE
        )
        series_chunks = np.array_split(sentiment_series, 33)
        chunks_average = [s.mean() for s in series_chunks]
        return np.array(chunks_average)

    def predict_book_arc(self, book: str):
        time_series = self.__calculate_book_arc__(book)
        prediction = self.arc_model.model.predict(
            time_series[np.newaxis, ..., np.newaxis]
        )
        return prediction.argmax()

    def predict_multiple_books(self, books: list):
        list_of_series = [
            self.__calculate_book_arc__(book)[...,np.newaxis]
            for book in books
        ]
        predictions=self.arc_model.model.predict(
            np.stack(list_of_series)
        )

        return predictions.argmax(axis=1)

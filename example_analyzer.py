import os

from analyzer import Analyzer
import hedonometer.settings as st

HEDONOMETER = 'hedonometer/data/best_sentiment_m_cleaned.bin'
EXAMPLE_PATH = 'texts/pt/'

if __name__ == '__main__':
    file_names = os.listdir(EXAMPLE_PATH)
    files = [open(EXAMPLE_PATH + name) for name in file_names]
    books = [file.read() for file in files]
    anlzr = Analyzer(
        hedonometer_path=HEDONOMETER,
        tokenizer_path=st.PRE_TRAINED_MODEL
    )
    arc = anlzr.predict_book_arc(books[0])
    print(arc)
    arcs = anlzr.predict_multiple_books(books)
    print(arcs)

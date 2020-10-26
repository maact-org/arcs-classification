import pandas as pd
import numpy as np
import logging
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_sentences_from_text(text):
    """Loads a list of sentences from a text"""
    def get_sentences():
        f = open(text, 'r+')
        for line in f.readlines():
            sentences = line.split('.')
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = ''.join([c for c in sentence if c in string.ascii_lowercase or ' '])
                yield sentence
    return get_sentences


def get_score_from_hd():
    """Get score of sentence given an hedonometer"""
    hd = pd.read_csv("hedonometer/hedonometer.csv")
    hd_dict = {}
    for index, row in hd.iterrows():
        hd_dict[row['Word'].lower()] = row['Happiness Score']

    def get_score(sentence):
        return hd_dict.get(sentence, None)
    return get_score


def get_score_from_vader():
    """Get score of sentence given the vader pre-trained model."""
    analyzer = SentimentIntensityAnalyzer()

    def get_score(sentence):
        d = analyzer.polarity_scores(sentence)
        score = -d['neg'] + d['pos'] or None
        return score
    return get_score


def calculate_story_scores(get_score, sentences):
    """Returns an array of scores"""
    scores = np.array([])

    for sentence in sentences():
        score = get_score(sentence)
        if score:
            scores = np.append(scores, score)

    logging.info("%d sentences scored" % len(scores))
    return scores


def get_df_for_plotting(scores: np.ndarray, window=0.1):
    """Returns a Series DataFrame"""
    n = scores.size
    s = np.array_split(scores, 1/window)
    averaged = np.array([np.average(i) for i in s])
    return pd.Series(averaged)

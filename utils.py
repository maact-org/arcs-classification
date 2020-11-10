import logging
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import math

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


def get_score_from_hd(hd_file="hedonometer/hedonometer.csv"):
    """Get score of sentence given an hedonometer"""
    hd = pd.read_csv(hd_file)
    hd_dict = {}
    for index, row in hd.iterrows():
        hd_dict[row['Word'].lower()] = row['Happiness Score']

    def get_score(sentence):
        scores = []
        for word in sentence.split():
            scores.append(hd_dict.get(word, 0))
        averaged = np.average(scores)
        return averaged
    return get_score


def get_score_from_vader():
    """Get score of sentence given the vader pre-trained model."""
    analyzer = SentimentIntensityAnalyzer()

    def get_score(sentence):
        d = analyzer.polarity_scores(sentence)
        score = -d['neg'] if abs(d['neg']) > d['pos'] else d['pos']
        return score
    return get_score


def get_story_scores(get_score, sentences, window=33):
    """Returns a DataFrame of scores"""
    scores = np.array([])

    for sentence in sentences():
        score = get_score(sentence)
        if score and not math.isnan(score):
            scores = np.append(scores, score)

    logging.info("%d sentences scored" % len(scores))
    n = scores.size
    s = np.array_split(scores, 33)
    averaged = np.array([np.average(i) for i in s])
    df = pd.Series(averaged)
    df = df.dropna()
    # Normalize the data
    max = df.max()
    min = df.min()
    df = df.apply(lambda x: 2*((x-min)/(max-min))-1)
    return df
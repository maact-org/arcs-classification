import utils
import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Shows the arc of story of given text
text = "texts/tokillamockingbird.txt"
method="hedo"

sentences = utils.get_sentences_from_text(text)
if method == "hedo":
    get_score = utils.get_score_from_hd()
if method == "vader":
    get_score = utils.get_score_from_vader()

df = utils.get_story_scores(get_score, sentences, window=100)
df = df.rolling(8, win_type='blackman').mean()
sns.lineplot(x=df.index, y=df.values, data=df)
plt.show()
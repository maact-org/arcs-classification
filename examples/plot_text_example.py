import utils
import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


text = "data/janeeyre.txt"
sentences = utils.get_sentences_from_text(text)
get_score = utils.get_score_from_hd()
df = utils.get_story_scores(get_score, sentences, window=0.03)
a = df.values.reshape(1, 33, 1)
sns.lineplot(x=df.index, y=df.values, data=df)
plt.show()
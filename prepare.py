import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def prepare_inputs(text, model='hedo'):
    """Create a Series DataFrame with normalized scores given the name of an existing text."""
    text = text
    sentences = arcs.get_sentences_from_text('data/%s.txt' % text)

    # Optional models for getting scores (Temporal)
    models_dict = {
        'hedo': arcs.get_score_from_hd(),
        'vader': arcs.get_score_from_vader()
    }
    get_score = models_dict[model]

    # Get the time serie as raw array
    a = arcs.calculate_story_scores(get_score, sentences)
    # Get as dataframe averaging windows of data
    df = arcs.get_df_for_plotting(a, window=0.03)
    # Rolling
    # df = df.rolling(100, win_type='blackman').sum()
    df = df.dropna()

    # Normalize the data
    max = df.max()
    min = df.min()
    df = df.apply(lambda x: 2*((x-min)/(max-min))-1)
    return df


if __name__ == "__main__":
    model = 'hedo'
    text = 'thecountofmontecristo'
    df = prepare_inputs(text, model=model)

    # Plotting the data
    sns.lineplot(x=df.index, y=df.values, data=df)
    plt.savefig(model+'_'+text)
    plt.show()



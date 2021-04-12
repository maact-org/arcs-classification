import utils
import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def train_model_with_csv(model):
    df = pd.read_csv("training/arcs.csv")
    data = np.array([])
    labels = np.array([])
    for index, row in df.iterrows():
        data = np.append(data, [[row[i] for i in range(1, 34)]])
        labels = np.append(labels, [row['category']])
    data = data.reshape(df.shape[0], 33, 1)
    history = model.fit(data, labels, epochs=500)
    print(history.history)

def get_timeseries(text, method="hedo", plot_as=False):
    # Get scores from text
    sentences = utils.get_sentences_from_text(text)
    if method == "hedo":
        get_score = utils.get_score_from_hd()
    if method == "vader":
        get_score = utils.get_score_from_vader()
    df = utils.get_story_scores(get_score, sentences, window=33)
    if plot_as:
        sns.lineplot(x=df.index, y=df.values, data=df)
        plt.savefig(plot_as+".png")
        plt.clf()
    return df

def get_class(df):
    a = df.values.reshape(1, 33, 1)
    result = model.predict(a)
    i = np.argmax(result)
    return result[0], i

if __name__ == "__main__":
    classes = ['rise', 'fall', 'rise-fall-rise', 'fall-rise-fall', 'fall-rise', 'rise-fall']

    # If already trained
    arcs = arcs.ArcModel(input_shape=(33, 1))

    try:
        arcs.load_model()
        model = arcs.model
    except Exception as e:
        model = arcs.model
        train_model_with_csv(model)
        arcs.save_model()

    files = os.listdir("texts")
    for file in files:
        try:
            df = get_timeseries("texts/"+file, method="vader", plot_as=file)
            result, i = get_class(df)
            print("{} was classified as {}".format(file, classes[i]))
        except:
            raise Exception("An error ocurred when trying to read {}".format(file))

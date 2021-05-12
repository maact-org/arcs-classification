import utils
import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging


def train_model_with_csv(model, file_path="arcs.csv", epochs=500):
    df = pd.read_csv(file_path).sample(frac=1)
    data = np.array([])
    labels = np.array([])

    le = LabelEncoder()
    le.fit(df['tag'])

    logging.info("Training into {} classes".format(le.classes_))
    for index, row in df.iterrows():
        data = np.append(data, [[row[i] for i in range(1, 34)]])
        label = le.transform([row['tag']])
        labels = np.append(labels, label)

    data = data.reshape(df.shape[0], 33, 1)
    history = model.fit(data, labels, epochs=epochs)
    print(history)
    return model

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

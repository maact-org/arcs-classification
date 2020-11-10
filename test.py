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
    model_json = model.to_json()
    with open('model-arcs.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model-arcs.h5')
    print(history.history)

def get_class(text, plot=False):
    # Get scores from text
    sentences = utils.get_sentences_from_text(text)
    get_score = utils.get_score_from_hd()
    df = utils.get_story_scores(get_score, sentences, window=0.03)
    a = df.values.reshape(1, 33, 1)
    result = model.predict(a)
    if plot:
        sns.lineplot(x=df.index, y=df.values, data=df)
        plt.savefig(text)
    i = np.argmax(result)
    return result[0], i

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

files = os.listdir("/home/valentina/Downloads/txts")
for file in files:
    try:
        result, i = get_class("/home/valentina/Downloads/txts/"+file, plot=True)
    except:
        pass
    else:
        print(file)
import utils
import arcs
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_model_with_csv(model):
    df = pd.read_csv("training/arcs.csv")
    data = np.array([])
    labels = np.array([])
    for index, row in df.iterrows():
        data = np.append(data, [[row[i] for i in range(1, 34)]])
        labels = np.append(labels, [row['category']])
    data = data.reshape(df.shape[0], 33, 1)
    model.fit(data, labels, epochs=300)
    model.save_weights("arcs-model")

classes = ['none', 'fall', 'rise-fall-rise', 'none', 'none', 'none']

# If already trained
arcs = arcs.ArcModel(input_shape=(33,1))

try:
    arcs.load_model()
    model = arcs.model
except:
    model = arcs.model
    train_model_with_csv(model)

# Get scores from text
sentences = utils.get_sentences_from_text("data/madamebovary.txt")
get_score = utils.get_score_from_hd()
df = utils.get_story_scores(get_score, sentences, window=0.03)
sns.lineplot(x=df.index, y=df.values, data=df)
a = df.values.reshape(1, 33, 1)
result = model.predict(a)
i = np.argmax(result)
# Print predicted class
print(classes[i])
plt.show()

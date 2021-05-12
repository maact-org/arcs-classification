from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import evaluation.metrics as mt


def evaluate_df(model, df):
    enc = OneHotEncoder()
    le = LabelEncoder()
    le.fit(df['tag'])
    enc = enc.fit(df['tag'].to_numpy().reshape(-1, 1))
    losses = []
    accuracies = []
    for index, row in df.iterrows():
        input = row['serie'].reshape(1, -1, 1)
        tag = row['tag']
        target = le.transform([[tag]]).reshape(1, -1, 1)
        output = model.model.evaluate(input, target)
        losses.append(output[0])
        accuracies.append(output[1])
    df_to_eval = pd.DataFrame({"loss": losses, "accuracy": accuracies})
    return df_to_eval

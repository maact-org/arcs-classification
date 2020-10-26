import numpy as np
import tensorflow as tf
import pandas as pd


def prepare_and_train_model(input_shape, train_data, train_labels):

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(33, 1)),
        tf.keras.layers.Conv1D(10, (5,)),
        tf.keras.layers.AveragePooling1D(pool_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10)
    return model


def load_example_inputs():
    df = pd.read_csv("training/arcs.csv")
    data = np.array([])
    labels = np.array([])
    for index, row in df.iterrows():
        data = np.append(data, [[row[i] for i in range(1, 34)]])
        labels = np.append(labels, [row['category']])
    data = data.reshape(df.shape[0], 33, 1)
    return data, labels

import tensorflow as tf
from keras.models import model_from_json


class ArcModel:
    def __init__(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(100, (3,)),
            tf.keras.layers.AveragePooling1D(pool_size=3),
            tf.keras.layers.Conv1D(5, (3,)),
            tf.keras.layers.AveragePooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation='sigmoid'),
        ])

        model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        self.model = model

    def load_model(self, file='model-arcs.json'):
        """
        Loads the model from json file
        :param file: The name of the file to load
        :return: None
        """
        self.model = model_from_json(file)

    def save_model(self, file='model-arcs.json'):
        """
        Saves the model to json file
        :param file: The name of the file for saving
        :return: None
        """
        with open(file, "w") as json_file:
            json_file.write(self.model)

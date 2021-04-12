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

    def load_model(self, file="model/model.h5"):
        """
        Loads the model from json file
        :param file: The name of the file to load
        :return: None
        """
        self.model = tf.keras.models.load_model(file)

    def save_model(self, file="model/model.h5"):
        """
        Saves the model to json file
        :param file: The name of the file for saving
        :return: None
        """
        self.model.save(file)

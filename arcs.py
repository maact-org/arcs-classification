import tensorflow as tf

class ArcModel:

    def __init__(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(10, (5,)),
            tf.keras.layers.AveragePooling1D(pool_size=3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation='sigmoid'),
        ])

        model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        self.model = model

    def load_model(self, file='arcs-model'):
        self.model.load_weights('arcs-model')
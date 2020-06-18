import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization


class Actor:
    def __init__(self, state_space_dim, action_space_dim, action_max_val, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.action_max_val = action_max_val
        self.model = self.build_nn()

    def build_nn(self):
        inputs = Input(shape=(self.state_space_dim,))
        out = Dense(self.hidden_layer_size, activation="relu")(inputs)
        out = BatchNormalization()(out)
        out = Dense(self.hidden_layer_size, activation="relu")(out)
        out = BatchNormalization()(out)
        outputs = Dense(self.action_space_dim, activation="tanh")(out)

        outputs = outputs * self.action_max_val
        model = tf.keras.Model(inputs, outputs)
        return model

    def forward(self, inputs):
        return self.model(inputs)

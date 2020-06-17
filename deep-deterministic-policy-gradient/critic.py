import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate


class Critic:
    def __init__(self, state_space_dim, action_space_dim, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.model = self.build_nn()

    def build_nn(self):
        state_input = Input(shape=self.state_space_dim)
        state_out = Dense(16, activation="relu")(state_input)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation="relu")(state_out)
        state_out = BatchNormalization()(state_out)

        action_input = Input(shape=self.action_space_dim)
        action_out = Dense(32, activation="relu")(action_input)
        action_out = BatchNormalization()(action_out)

        concat = Concatenate()([state_out, action_out])

        out = Dense(self.hidden_layer_size, activation="relu")(concat)
        out = BatchNormalization()(out)
        out = Dense(self.hidden_layer_size, activation="relu")(out)
        out = BatchNormalization()(out)
        outputs = Dense(1)(out)

        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def forward(self, inputs):
        return self.model(inputs)

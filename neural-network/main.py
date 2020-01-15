import math
from typing import List

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


array_sigmoid = np.vectorize(sigmoid)


class Layer:
    def __init__(self, array, dropout_probability):
        self.array = array
        self.shape = array.shape
        self.dropout_probability = dropout_probability
        self.last_zero_indexes = []

    def zero_random_rows(self):
        zero_indexes = [x < self.dropout_probability for x in np.random.random(self.array.shape[0])]
        self.array[zero_indexes] = 0
        return zero_indexes

    def copy(self):
        return Layer(self.array.copy(), self.dropout_probability)

    def __repr__(self):
        return f"Layer(array={self.array}, dop={self.dropout_probability}, lzi={self.last_zero_indexes})"


class NeuralNetwork:
    def __init__(self, layer_sizes=None, dropout_probability=0.):
        self.layers: List[Layer] = []
        if layer_sizes is None:
            return
        for i in range(len(layer_sizes) - 1):
            self.add_layer(Layer(np.random.rand(layer_sizes[i], layer_sizes[i + 1]), dropout_probability))

    def add_layer(self, layer: Layer):
        if self.layers and self.layers[-1].shape[1] != layer.shape[0]:
            raise Exception(f"Layers {self.layers[-1]} and {layer} have incompatible sizes")
        self.layers.append(layer)

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def predict(self, inp):
        out = inp
        for layer in self.layers:
            out = array_sigmoid(np.dot(out, layer.array))
        return out

    def layers_count(self):
        return len(self.layers)

    def get_gradients(self, train_input, train_output):
        outs = [train_input]
        for layer in self.layers:
            if layer.dropout_probability > 0.:
                final_layer = layer.copy()
                layer.last_zero_indexes = final_layer.zero_random_rows()
            else:
                final_layer = layer
                layer.last_zero_indexes = []
            outs.append(array_sigmoid(np.dot(outs[-1], final_layer.array)))
        deltas = [-outs[-1] * (1 - outs[-1]) * (train_output - outs[-1])]
        if self.layers[0].dropout_probability > 0.:
            deltas[0][:, self.layers[0].last_zero_indexes] = 0

        for i in range(self.layers_count() - 1):
            # print("i:", i)
            new_delta = outs[-i - 2] * (1 - outs[-i - 2]) * np.dot(self.layers[-i - 1].array,
                                                                   deltas[-1].swapaxes(0, 1)).swapaxes(0, 1)
            # print([(x.shape, x.last_zero_indexes) for x in self.layers])
            if self.layers[i].dropout_probability > 0.:
                # print(new_delta.shape[1])
                # print(len(self.layers[-i - 1].last_zero_indexes))
                new_delta[:, self.layers[-i - 1].last_zero_indexes] = 0
            deltas.append(new_delta)
        grads = []
        for i in range(self.layers_count()):
            grads.append(np.dot(outs[i].swapaxes(0, 1), deltas[-i - 1]))
        return grads

    def do_step(self, train_input, train_output, learning_rate):
        grads = self.get_gradients(train_input, train_output)
        for i in range(self.layers_count()):
            self.layers[i].array += -learning_rate * grads[i]

    def mse(self, test_input, test_output):
        return np.square(test_output - self.predict(test_input)).sum() / test_input.shape[0]

    def learn(self, train_input, train_output, learning_rate, accuracy):
        mse_val = self.mse(train_input, train_output)
        while mse_val > accuracy:
            try:
                self.do_step(train_input, train_output, learning_rate)
                mse_val = self.mse(train_input, train_output)
                print(mse_val)
            except KeyboardInterrupt:
                print("Stopped by user")
                break


ARRAY_1 = np.array([
    [0.2, 0.3, 0.4],
    [0.7, -0.2, -0.1],
])

ARRAY_2 = np.array([
    [0.37431857, -0.4050446,  -2.757726,  0.234686],
    [10, -0.4266498,   0.45890669, -1.0897836],
    [-0.14667352, -2.1691296, -0.02718759,  0.14773357]
])

ARRAY_3 = np.array([
    [0.3, 0.1],
    [-2., 0.2],
    [0.1, -3.1],
    [-0.9, 1.1],
])


def gen_out_by_arrays(arrays, inp):
    nn = NeuralNetwork()
    nn.add_layers([Layer(array, 0.) for array in arrays])
    return nn.predict(inp)


def main():
    train_input = np.random.rand(1000, 2)
    train_output = gen_out_by_arrays([ARRAY_1, ARRAY_2, ARRAY_3], train_input)
    nn = NeuralNetwork([2, 3, 4, 2])
    nn.layers[2].dropout_probability = 0.3
    nn.learn(train_input, train_output, 0.1, 10e-5)

    test_input = np.random.rand(100, 2)
    test_output = gen_out_by_arrays([ARRAY_1, ARRAY_2, ARRAY_3], test_input)
    print(nn.mse(test_input, test_output))


if __name__ == '__main__':
    main()

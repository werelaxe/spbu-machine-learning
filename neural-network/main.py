import math
import random
from time import time, sleep

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


array_sigmoid = np.vectorize(sigmoid)


class NeuralNetwork:
    def __init__(self, layer_sizes=None):
        self.layers = []
        if layer_sizes is None:
            return
        for i in range(len(layer_sizes) - 1):
            self.add_layer(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))

    def add_layer(self, layer):
        if self.layers and self.layers[-1].shape[1] != layer.shape[0]:
            raise Exception(f"Layers {self.layers[-1]} and {layer} have incompatible sizes")
        self.layers.append(layer)

    def predict(self, inp):
        out = inp
        for layer in self.layers:
            out = array_sigmoid(np.dot(out, layer))
        return out

    def layers_count(self):
        return len(self.layers)

    def get_gradients(self, train_input, train_output):
        outs = [train_input]
        for layer in self.layers:
            outs.append(array_sigmoid(np.dot(outs[-1], layer)))
        deltas = [-outs[-1] * (1 - outs[-1]) * (train_output - outs[-1])]
        for i in range(self.layers_count() - 1):
            new_delta = outs[-i - 2] * (1 - outs[-i - 2]) * np.dot(self.layers[-i - 1], deltas[-1].swapaxes(0, 1)).swapaxes(0, 1)
            deltas.append(new_delta)
        grads = []
        for i in range(self.layers_count()):
            grads.append(np.dot(outs[i].swapaxes(0, 1), deltas[-i - 1]))
        return grads

    def do_step(self, train_input, train_output, learning_rate):
        grads = self.get_gradients(train_input, train_output)
        for i in range(self.layers_count()):
            self.layers[i] += -learning_rate * grads[i]

    def mse(self, test_input, test_output):
        return np.square(test_output - self.predict(test_input)).sum() / test_input.shape[0]


LAYER_1 = np.array([
    [0.2, 0.3, 0.4],
    [0.7, -0.2, -0.1],
])


LAYER_2 = np.array([
    [0.37431857, -0.4050446,  -2.757726,  0.234686],
    [10, -0.4266498,   0.45890669, -1.0897836],
    [-0.14667352, -2.1691296, -0.02718759,  0.14773357]
])


LAYER_3 = np.array([
    [0.3, 0.1],
    [-2., 0.2],
    [0.1, -3.1],
    [-0.9, 1.1],
])


def main():
    # nn = NeuralNetwork([2, 3, 4, 2])
    nn = NeuralNetwork()

    inp = np.array([[0.1, 0.9]])
    out = np.array([[0.1, 0.2]])
    print(nn.predict(inp))

    exit(1)
    # layer_1 = np.random.rand(2, 3)
    # layer_2 = np.random.rand(3, 4)
    # layer_3 = np.random.rand(4, 2)
    mse_val = nn.mse(inp, out)

    i = 0
    while mse_val > 10e-6:
        try:
            nn.do_step(inp, out, 0.01)
            mse_val = nn.mse(inp, out)
            i += 1
            print(mse_val)
        except KeyboardInterrupt:
            break
    print(nn.predict(inp))


if __name__ == '__main__':
    main()

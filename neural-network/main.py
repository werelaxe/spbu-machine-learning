import math
from enum import Enum
from typing import List, Tuple

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.e ** (np.dot(x, -1)))


def sigmoid_derivative(sigmoid_value, _):
    return sigmoid_value * (1 - sigmoid_value)


def softmax(x):
    exp = math.e ** x
    return np.transpose((np.transpose(exp) / np.transpose(exp.sum(axis=1))))


def softmax_derivative(softmax_value, _):
    return softmax_value * (1 - softmax_value)


def relu(x):
    return np.where(x > 0., x, 0.)


def relu_derivative(_, x):
    return np.where(x <= 0., 0., 1.)


class ActivationFunction(Enum):
    SIGMOID = 1
    SOFTMAX = 2
    RELU = 3


ACTIVATION_FUNCTIONS = {
    ActivationFunction.SIGMOID: (sigmoid, sigmoid_derivative),
    ActivationFunction.SOFTMAX: (softmax, softmax_derivative),
    ActivationFunction.RELU: (relu, relu_derivative),
}


class Layer:
    def __init__(
            self,
            array: np.ndarray = None,
            shape: Tuple[int, int] = None,
            dropout_probability: float = 0,
            activation_function: ActivationFunction = ActivationFunction.SIGMOID
    ):
        self.array = array
        if array is None:
            self.array = np.random.rand(*shape)
            self.shape = shape
        else:
            self.shape = array.shape
        self.dropout_probability = dropout_probability
        self.last_zero_indexes = []
        self.activation_function = activation_function
        self.activate, self.derivative_activate = ACTIVATION_FUNCTIONS[activation_function]

    def zero_random_rows(self):
        zero_indexes = [x < self.dropout_probability for x in np.random.random(self.array.shape[0])]
        self.array[zero_indexes] = 0
        return zero_indexes

    def copy(self):
        return Layer(self.array.copy(), None, self.dropout_probability, self.activation_function)

    def __repr__(self):
        return f"Layer(array={self.array}, " \
               f"dop={self.dropout_probability}, " \
               f"lzi={self.last_zero_indexes}, " \
               f"af={self.activation_function})"


class NeuralNetwork:
    def __init__(
            self,
            layer_sizes: List[int] = None,
            dropout_probability: float = 0.,
            activation_function: ActivationFunction = ActivationFunction.SIGMOID
    ):
        self.layers: List[Layer] = []
        if layer_sizes is None:
            return
        for i in range(len(layer_sizes) - 1):
            new_later = Layer(
                np.random.rand(layer_sizes[i], layer_sizes[i + 1]),
                None,
                dropout_probability,
                activation_function
            )
            self.add_layer(new_later)

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
            out = layer.activate(np.dot(out, layer.array))
        return out

    def layers_count(self):
        return len(self.layers)

    def get_gradients(self, train_input, train_output):
        outs = [train_input]
        sums = [train_input]
        for layer in self.layers:
            if layer.dropout_probability > 0.:
                final_layer = layer.copy()
                layer.last_zero_indexes = final_layer.zero_random_rows()
            else:
                final_layer = layer
                layer.last_zero_indexes = []
            new_sum = np.dot(outs[-1], final_layer.array)
            sums.append(new_sum)
            outs.append(layer.activate(new_sum))

        deltas = [-self.layers[-1].derivative_activate(outs[-1], sums[-1]) * (train_output - outs[-1])]
        if self.layers[0].dropout_probability > 0.:
            deltas[0][:, self.layers[0].last_zero_indexes] = 0

        for i in range(self.layers_count() - 1):
            new_delta = \
                self.layers[-i - 1].derivative_activate(
                    outs[-i - 2],
                    sums[-i - 2]
                ) * np.dot(
                    self.layers[-i - 1].array,
                    deltas[-1].swapaxes(0, 1)
                ).swapaxes(0, 1)

            if self.layers[i].dropout_probability > 0.:
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
            except KeyboardInterrupt:
                print("Stopped by user")
                break


def main():
    nn = NeuralNetwork([4, 4, 4], activation_function=ActivationFunction.SIGMOID)
    nn.add_layer(Layer(shape=(4, 4), activation_function=ActivationFunction.SOFTMAX))


if __name__ == '__main__':
    main()

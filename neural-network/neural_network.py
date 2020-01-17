from operator import sub
from typing import List

import numpy as np

from activation_functions import ActivationFunction
from dataset_ops import transform_to_digit
from layer import Layer


class NeuralNetwork:
    def __init__(
            self,
            layer_sizes: List[int] = None,
            dropout_probability: float = 0.,
            activation_function: ActivationFunction = ActivationFunction.SIGMOID,
            random_weights: bool = False
    ):
        self.layers: List[Layer] = []
        if layer_sizes is None:
            return
        for i in range(len(layer_sizes) - 1):
            new_array = \
                np.random.rand(layer_sizes[i], layer_sizes[i + 1]) \
                    if random_weights \
                    else np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            new_later = Layer(
                array=new_array,
                dropout_probability=dropout_probability,
                activation_function=activation_function
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
            new_sum = np.dot(out, layer.array)
            out = layer.activate(new_sum)
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

        if self.layers[-1].dropout_probability > 0.:
            deltas[0][:, self.layers[-1].last_zero_indexes] = 0

        grads = []
        for i in range(self.layers_count()):
            grads.append(np.dot(outs[i].swapaxes(0, 1), deltas[-i - 1]))
        return grads

    def do_step(self, train_input, train_output, learning_rate):
        grads = self.get_gradients(train_input, train_output)
        for i in range(self.layers_count()):
            self.layers[i].array += -learning_rate * grads[i]

    def do_epoch(self, train_input, train_output, learning_rate, batch_size):
        for i in range(train_input.shape[0] // batch_size):
            batch_input = train_input[i * batch_size:i * batch_size + batch_size]
            batch_output = train_output[i * batch_size:i * batch_size + batch_size]
            self.do_step(batch_input, batch_output, learning_rate)

    def mse(self, test_input, test_output):
        return np.square(test_output - self.predict(test_input)).sum() / test_input.shape[0]

    def accuracy(self, test_input, test_output):
        predicted_output = self.predict(test_input)
        transformed_arrays = [np.array(transform_to_digit(array)) for array in [predicted_output, test_output]]
        return 1 - sub(*transformed_arrays).nonzero()[0].shape[0] / predicted_output.shape[0]

    def learn(self, train_input, train_output, learning_rate, required_accuracy, batch_size):
        accuracy_val = self.accuracy(train_input, train_output)
        while required_accuracy > accuracy_val:
            self.do_epoch(train_input, train_output, learning_rate, batch_size)
            accuracy_val = self.accuracy(train_input, train_output)
            mse_val = self.mse(train_input, train_output)
            print("{:10.6f} {:10.3f} {:10.3f}".format(mse_val, accuracy_val, learning_rate))
            if np.isnan(mse_val):
                print("Escape due to nan!")
                raise Exception("nan!")

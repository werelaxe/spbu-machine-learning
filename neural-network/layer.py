from typing import Tuple

import numpy as np

from activation_functions import ActivationFunction, ACTIVATION_FUNCTIONS


class Layer:
    def __init__(
            self,
            array: np.ndarray = None,
            shape: Tuple[int, int] = None,
            dropout_probability: float = 0,
            activation_function: ActivationFunction = ActivationFunction.SIGMOID,
            random_weights: bool = False
    ):
        self.array = array
        if array is None:
            self.array = np.random.rand(*shape) if random_weights else np.full(shape, 0.)
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

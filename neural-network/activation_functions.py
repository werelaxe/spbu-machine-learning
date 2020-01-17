import math
from enum import Enum

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

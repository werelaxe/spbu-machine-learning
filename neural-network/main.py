import math
import random
from time import time, sleep

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


array_sigmoid = np.vectorize(sigmoid)


def get_grad(inp, layer, out, k):
    o_1 = inp
    o_2 = array_sigmoid(np.dot(inp, layer))
    delta = o_2 * (1 - o_2) * (out - o_2)
    return -k * np.dot(o_1.swapaxes(0, 1), delta)


def compute_f(inp, layer_1, layer_2):
    o_1 = inp
    o_2 = array_sigmoid(np.dot(o_1, layer_1))
    o_3 = array_sigmoid(np.dot(o_2, layer_2))
    return o_3


def grads(inp, layer_1, layer_2, out, k):
    o_1 = inp
    o_2 = array_sigmoid(np.dot(o_1, layer_1))
    o_3 = array_sigmoid(np.dot(o_2, layer_2))
    delta_3 = -o_3 * (1 - o_3) * (out - o_3)
    delta_2 = o_2 * (1 - o_2) * np.dot(layer_2, delta_3.swapaxes(0, 1)).swapaxes(0, 1)
    grad_1 = -k * np.dot(o_1.swapaxes(0, 1), delta_2)
    grad_2 = -k * np.dot(o_2.swapaxes(0, 1), delta_3)
    return grad_1, grad_2


def mse(inp, layer_1, layer_2, out):
    f_values = compute_f(inp, layer_1, layer_2)
    return (np.square(out - f_values)).sum() / inp.shape[0]


LAYER_1 = np.array([
    [0.2, 0.3, 0.4],
    [0.7, -0.2, -0.1],
])


LAYER_2 = np.array([
    [0.37431857, -0.4050446,  -0.20757726,  0.234686],
    [10, -0.4266498,   0.45890669, -0.0897836],
    [-0.14667352, -0.21691296, -0.02718759,  0.14773357]
])


def gen():
    inp = [[random.random() for _ in range(4)] for _ in range(10)]
    print(inp)
    print(compute_f(inp, LAYER_1, LAYER_2).tolist())


def main():
    layer_1 = LAYER_1
    layer_2 = np.random.rand(3, 4)
    mse_val = mse(inp, layer_1, layer_2, out)

    while mse_val > 10e-9:
        try:
            grad_1, grad_2 = grads(inp, layer_1, layer_2, out, 1)
            layer_1 += grad_1
            layer_2 += grad_2
            mse_val = mse(inp, layer_1, layer_2, out)
            print(mse_val)
        except KeyboardInterrupt:
            break
    print(layer_1)
    print(layer_2)
    print(compute_f(np.array([[0.5136712627291826, 0.8779685080171047]]), layer_1, layer_2))


if __name__ == '__main__':
    main()

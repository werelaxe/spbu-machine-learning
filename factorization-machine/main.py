from time import time, sleep
import random

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from one_hot_synthetics import get_dataset_with_encoder


FACTORS_COUNT = 3
random.seed(0)


def compute_f(vector_w, matrix_v, xs):
    s = np.zeros(xs.shape[0])
    for f in range(len(matrix_v[0])):
        x = xs.multiply(matrix_v[:, f])
        s = s + np.transpose(np.square(x.sum(axis=1)) - x.power(2).sum(axis=1))
    return vector_w[0] + xs.dot(vector_w[1:]) + s / 2


def mse(xs, ys, vector_w, matrix_v):
    f_xs = compute_f(vector_w, matrix_v, xs)
    return ((ys - f_xs) ** 2).sum() / len(xs)


def r2(xs, ys, vector_w, matrix_v):
    avg = np.average(ys)
    f_xs = compute_f(vector_w, matrix_v, xs)
    ss_res = ((ys - f_xs) ** 2).sum()
    ss_tot = ((ys - avg) ** 2).sum()
    return 1 - ss_res / ss_tot


def rmse(xs, ys, vector_w, matrix_v):
    return np.sqrt(mse(xs, ys, vector_w, matrix_v))


def grad_of_mse_ws(xs, ys, vector_w):
    value = xs.dot(vector_w[1:]) - ys + vector_w[0]
    pre_result = np.insert(csr_matrix(value).dot(xs).todense(), 0, value.sum())
    return 2 * pre_result / xs.shape[0]


def grad_of_mse_by_f_l(xs, ys, vector_w, matrix_v, f, l):
    v = matrix_v[:, f]
    vx = np.dot(xs, v)
    return 2 * ((compute_f(vector_w, matrix_v, xs) - ys) * (vx * xs[:, l] - v[l] * xs[:, l] ** 2)).sum() / xs.shape[0]


def grad_of_mse_v(xs, ys, vector_w, matrix_v):
    a = []
    for f in range(FACTORS_COUNT):
        b = []
        for l in range(xs.shape[1]):
            b.append(grad_of_mse_by_f_l(xs, ys, vector_w, matrix_v, f, l))
        a.append(b)
    return np.transpose(np.array(a))


def get_value():
    return random.random()


def main():
    xs, ys, encoder = get_dataset_with_encoder(sparse=True)
    n = xs.shape[0]
    m = xs.shape[1]
    vector_w = np.array([random.random() for _ in range(m + 1)])
    matrix_v = np.array([[1. for _ in range(FACTORS_COUNT)] for _ in range(m)])
    print(compute_f(vector_w, matrix_v, xs))
    print(grad_of_mse_ws(xs, ys, vector_w).A)
    # SIZE = 20
    # k = 0.01
    # start = time()
    #
    # rmse_value = 99999999999
    # while True:
    #     try:
    #         grad_of_v = grad_of_mse_v(xs, ys, vector_w, matrix_v)
    #         matrix_v -= grad_of_v * k
    #         grad_of_w = grad_of_mse_ws(xs, ys, vector_w)
    #         vector_w -= grad_of_w * k
    #         rmse_value = rmse(xs, ys, vector_w, matrix_v)
    #         print(rmse_value)
    #     except KeyboardInterrupt:
    #         break
    # print("rmse:", rmse(xs, ys, vector_w, matrix_v))


if __name__ == '__main__':
    main()

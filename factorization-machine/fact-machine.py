from time import time, sleep

import numpy as np
import random

FACTORS_COUNT = 3
random.seed(0)


def compute_f(vector_w, matrix_v, xs):
    s = np.zeros(xs.shape[0])
    for f in range(len(matrix_v[0])):
        x = xs * matrix_v[:, f]
        s += x.sum(axis=1) ** 2 - (x ** 2).sum(axis=1)
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
    xs = np.insert(xs, 0, np.array([1.] * xs.shape[0]), 1)
    value = np.dot(xs, vector_w) - ys
    return 2 * value.dot(xs) / xs.shape[0]


def do_step(xs, ys, vector_w, matrix_v, coef):
    pass


def grad_of_mse_by_f_l(xs, ys, vector_w, matrix_v, f, l):
    v = matrix_v[:, f]
    vx = np.dot(xs, v)
    # s_i = (compute_f(vector_w, matrix_v, xs) - ys)
    # (vx * xs[:, l] - v[l] * xs[:, l] ** 2)
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
    m = 10
    n = 200
    dataset = np.array([[get_value() for _ in range(m)] + [random.choice([1, 2, 3, 4, 5])] for _ in range(n)])
    for i in range(len(dataset)):
        dataset[i][-1] = dataset[i][0] + 2 * dataset[i][1] - dataset[i][2]
        # print(dataset[i])
    vector_w = np.array([0. for _ in range(m + 1)])
    matrix_v = np.array([[-10e-12 for _ in range(FACTORS_COUNT)] for _ in range(m)])
    SIZE = 20
    k = 0.01
    start = time()
    xs = dataset[:, :-1]
    ys = dataset[:, -1]

    rmse_value = 99999999999
    while True:
        try:
            np.random.shuffle(dataset)
            for i in range(len(dataset) // SIZE):
                actual_xs = xs[i * SIZE:i * SIZE + SIZE]
                actual_ys = ys[i * SIZE:i * SIZE + SIZE]

                grad_of_v = grad_of_mse_v(actual_xs, actual_ys, vector_w, matrix_v)
                matrix_v -= grad_of_v * k
                # grad_of_w = grad_of_mse_ws(actual_xs, actual_ys, vector_w)
                # vector_w -= grad_of_w * k
        except KeyboardInterrupt:
            break
        # k *= 0.99
        rmse_value = rmse(dataset[:, :-1], dataset[:, -1], vector_w, matrix_v)
        print(k, rmse_value)
    print(compute_f(vector_w, matrix_v, xs[0:1]))
    print(ys[0:1])
    print(matrix_v)
    print("rmse:", rmse(dataset[:, :-1], dataset[:, -1], vector_w, matrix_v))


if __name__ == '__main__':
    main()

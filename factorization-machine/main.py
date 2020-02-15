import random

import numpy as np
from sklearn.utils import shuffle

from dataset_reader import read_dataset, split_data_to_fold, FOLDS_COUNT

FACTORS_COUNT = 3


def compute_f(vector_w, w0, matrix_v, xs):
    vx = xs.dot(matrix_v)
    vx = (vx ** 2) - xs.power(2).dot(matrix_v ** 2)
    add_v = (np.sum(vx, axis=1) / 2).reshape(-1, 1)
    vx = xs.dot(vector_w)
    vx = w0 + vx
    return vx + add_v


def rmse(vector_w, w0, matrix_v, xs, ys):
    pre = compute_f(vector_w, w0, matrix_v, xs)
    mse = np.sum((ys - pre) ** 2) / pre.shape[0]
    return np.sqrt(mse)


def get_grads(xs, ys, vector_w, w0, matrix_v):
    value = compute_f(vector_w, w0, matrix_v, xs)
    delta = ys - value
    grad_w0 = 2 * np.sum(delta) / xs.shape[1]
    grad_w = 2 * (xs.T.dot(delta)) / xs.shape[1]
    grad_v = 2 * ((xs.T.dot(np.multiply(delta, xs.dot(matrix_v))))
                  - np.multiply(matrix_v, xs.T.power(2).dot(delta))) / xs.shape[1]
    return grad_w0, grad_w, grad_v


def do_step(batch_xs, batch_ys, vector_w, w0, matrix_v, learning_rate):
    grad_w0, grad_w, grad_v = get_grads(batch_xs, batch_ys, vector_w, w0, matrix_v)
    return w0 + grad_w0 * learning_rate, vector_w + grad_w * learning_rate, matrix_v + grad_v * learning_rate


def extract_batch(xs, ys, batch_index, batch_size):
    return xs.tocsr()[batch_index * batch_size: (batch_index + 1) * batch_size, :], \
           ys[batch_index * batch_size: (batch_index + 1) * batch_size, :]


def learn(xs, ys, k, learning_rate, iters_count, epoch_count, batch_size):
    vector_w = np.random.rand(xs.shape[1], 1)
    w0 = random.random()

    matrix_v = np.random.rand(xs.shape[1], k)

    rmse_val = 99999999999
    for iter_index in range(iters_count):
        xs, ys = shuffle(xs, ys)
        number_of_batches = xs.shape[0] // batch_size

        for _ in range(epoch_count):
            for batch_index in range(number_of_batches):
                batch_xs, batch_ys = extract_batch(xs, ys, batch_index, batch_size)
                w0, vector_w, matrix_v = do_step(batch_xs, batch_ys, vector_w, w0, matrix_v, learning_rate)
        rmse_val = rmse(vector_w, w0, matrix_v, xs, ys)
        print("{}: {}".format(iter_index, rmse_val))

    return vector_w, w0, matrix_v, rmse_val


def main():
    xs, ys, encoder = read_dataset()
    print("Start learning")

    rmses = []
    for fold_index in range(FOLDS_COUNT):
        train_xs, train_ys, test_xs, test_ys = split_data_to_fold(xs, ys, fold_index)

        vector_w, w0, matrix_v, rmse_val = learn(train_xs, train_ys, 3, 0.016, 100, 10, train_xs.shape[0])

        final_fold_rmse = rmse(vector_w, w0, matrix_v, test_xs, test_ys)
        print("final rmse:", final_fold_rmse)
        rmses.append(final_fold_rmse)
    print(rmses)
    print(np.average(rmses))


if __name__ == '__main__':
    main()

import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import OneHotEncoder

FOLDS_COUNT = 5


def read_dataset(n=None):
    xs = []
    ys = []
    i = 0
    with open("dataset.txt") as dataset_file:
        for line in dataset_file:
            if ':' in line:
                current_movie = line.split(":")[0]
            else:
                user_id, raw_mark, _ = line.split(",")
                xs.append((user_id, current_movie))
                ys.append(int(raw_mark))
            if n is not None:
                i += 1
                if i >= n:
                    break
    encoder = OneHotEncoder(categories='auto', sparse=True)
    encoder.fit(xs)
    return encoder.transform(xs), np.array(ys).reshape(-1, 1), encoder


def split_data_to_fold(xs, ys, fold_index):
    fold_size = xs.shape[0] // FOLDS_COUNT

    train_xs = vstack([xs[:fold_index * fold_size, :], xs[fold_size * (fold_index + 1):, :]])
    train_ys = np.vstack([ys[:fold_index * fold_size, :], ys[fold_size * (fold_index + 1):, :]])

    test_xs = xs[fold_index * fold_size:(fold_index + 1) * fold_size]
    test_ys = ys[fold_index * fold_size:(fold_index + 1) * fold_size]

    return train_xs, train_ys, test_xs, test_ys
    # return xs[:fold_size * 4], ys[:fold_size * 4], xs[fold_size * 4:], ys[fold_size * 4:]


def kek(xs, ys, fold_index):
    fold_size = xs.shape[0] // FOLDS_COUNT
    return xs[:, fold_size * 4], ys[:, fold_size * 4], xs[:, fold_size * 4:], ys[:, fold_size * 4:]

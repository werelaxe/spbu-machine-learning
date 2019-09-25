from sys import stderr

import numpy as np


DATASET_PATH = 'dataset.csv'


def normalize(xs):
    average = np.average(xs)
    std = np.std(xs)
    return np.array([(x - average) / std for x in xs])


def read_dataset():
    dataset = np.genfromtxt(DATASET_PATH, delimiter=',')
    features_for_removing = []
    for i in range(len(dataset[0]) - 1):
        if np.std(dataset[:, i]) == 0:
            features_for_removing.append(i)

    for feature_for_removing in features_for_removing[::-1]:
        dataset = np.delete(dataset, feature_for_removing, 1)
    print(f"These features have been removed: {features_for_removing}", file=stderr)
    for i in range(len(dataset[0]) - 1):
        dataset[:, i] = normalize(dataset[:, i])
    np.random.shuffle(dataset)
    return dataset


def split_to_folds(dataset, i, folds_count):
    fold_size = len(dataset) // folds_count
    return np.array(list(dataset[: i * fold_size]) + list(dataset[(i + 1) * fold_size:])),\
        dataset[i * fold_size:(i + 1) * fold_size]


def split_dataset(dataset):
    return dataset[:, :-1], dataset[:, -1]

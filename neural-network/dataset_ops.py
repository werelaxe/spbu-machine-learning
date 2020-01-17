import numpy as np


def normalize(x):
    max_value = np.max(x)
    min_value = np.min(x)
    denominator = max_value - min_value
    return (x - min_value) / denominator


def one_hot_encode(data):
    new_data = np.zeros((len(data), 10))
    for i in range(len(data)):
        new_data[i, int(data[i])] = 1
    return new_data


def read_dataset(filen_path, size=None):
    print(f"Start reading dataset '{filen_path}'")
    data_train = np.genfromtxt(filen_path, delimiter=",")
    train_input = data_train[:, 1:]
    train_output = data_train[:, 0]
    train_output = one_hot_encode(train_output)
    train_input = normalize(train_input)
    if size is None:
        size = train_input.shape[0]
    print(f"Dataset '{filen_path}' has been successfully read, size={size}")
    return train_input[:size], train_output[:size]


def transform_to_digit(x):
    return np.where(x >= 0.5, 1., 0.).argmax(axis=1)

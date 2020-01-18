import numpy as np


def normalize(x):
    return x / np.max(x)
    # max_value = np.max(x)
    # min_value = np.min(x)
    # denominator = max_value - min_value
    # return (x - min_value) / denominator


def one_hot_encode(data):
    new_data = np.zeros((len(data), 10))
    for i in range(len(data)):
        new_data[i, int(data[i])] = 1
    return new_data


def conv_element(data):
    np.random.seed(0)

    data = data.reshape((28, 28))
    b = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            b[i, j] = data[i * 2:i * 2 + 2, j * 2:j * 2 + 2].sum() / 4.
    return b.reshape((196,))


def conv(data):
    return np.array([conv_element(line) for line in data])


def read_dataset(filen_path, size):
    # print(f"Start reading dataset '{filen_path}'")
    # data_train = np.genfromtxt(filen_path, delimiter=",")
    data_train = np.load("dataset.npy")[:size]
    np.random.shuffle(data_train)

    train_input = data_train[:, 1:]
    train_output = data_train[:, 0]
    train_output = one_hot_encode(train_output)
    train_input = normalize(train_input)

    train_input = conv(train_input)

    print(f"Dataset '{filen_path}' has been successfully read, size={train_input.shape[0]}")
    return train_input, train_output


def transform_to_digit(x):
    return np.where(x >= 0.5, 1., 0.).argmax(axis=1)


def unison_shuffled_copies(a, b):
    return a, b
    # assert len(a) == len(b)
    # p = np.random.permutation(len(a))
    # return a[p], b[p]

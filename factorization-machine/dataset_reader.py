import numpy as np
from sklearn.preprocessing import OneHotEncoder


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
    encoder = OneHotEncoder(sparse=True)
    encoder.fit(xs)
    return encoder.transform(xs), np.array(ys).reshape(-1, 1), encoder

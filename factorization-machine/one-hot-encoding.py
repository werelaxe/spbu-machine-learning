import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def rnd(x):
    return random.randint(0, x)


random.seed(0)

OBJS_COUNT = 30
WS = [0.8444218515250481, 0.7579544029403025, 0.420571580830845, 0.25891675029296335, 0.5112747213686085, 0.4049341374504143, 0.7837985890347726, 0.30331272607892745, 0.4765969541523558, 0.5833820394550312, 0.9081128851953352, 0.5046868558173903, 0.28183784439970383, 0.7558042041572239, 0.6183689966753316, 0.25050634136244054, 0.9097462559682401]


def main():
    xs = [('user-10', 'item-1'), ('user-8', 'item-0'), ('user-1', 'item-1'), ('user-9', 'item-1'), ('user-4', 'item-1'), ('user-3', 'item-5'), ('user-5', 'item-4'), ('user-8', 'item-1'), ('user-0', 'item-2'), ('user-5', 'item-5'), ('user-0', 'item-4'), ('user-7', 'item-4'), ('user-1', 'item-4'), ('user-6', 'item-5'), ('user-10', 'item-5'), ('user-7', 'item-2'), ('user-8', 'item-3'), ('user-4', 'item-4'), ('user-6', 'item-3'), ('user-6', 'item-2'), ('user-1', 'item-5'), ('user-5', 'item-3'), ('user-4', 'item-0')]
    # print(xs)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(xs)
    xs = encoder.transform(xs)
    # ys = WS[0] + xs.dot(WS[1:])
    # clf = DecisionTreeRegressor()
    # clf.fit(xs, ys)
    # print(ys)
    # print(clf.predict(encoder.transform([['user-10', 'item-2']])))
    print(xs)


if __name__ == '__main__':
    main()

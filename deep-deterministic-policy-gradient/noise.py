import numpy as np


class GaussianNoise:
    def __init__(self, mean, std, size):
        self.mean = mean
        self.std = std
        self.size = size

    def get_noise(self):
        return np.random.normal(self.mean, self.std, self.size)

import numpy as np


class ScalarEncoder:
    def __init__(self, minval, maxval, n=100, w=21):
        self.minval = minval
        self.maxval = maxval
        self.n = n  # total number of bits
        self.w = w  # number of active bits
        self.resolution = (maxval - minval) / (n - w)

    def encode(self, value):
        value = min(max(value, self.minval), self.maxval)
        start = int((value - self.minval) / self.resolution)
        if start > self.n - self.w:
            start = self.n - self.w
        encoded = np.zeros(self.n, dtype=int)
        encoded[start:start + self.w] = 1
        return encoded

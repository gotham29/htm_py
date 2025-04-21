import numpy as np
from typing import List


class ScalarEncoder:
    def __init__(self, n: int, w: int, minval: float, maxval: float):
        assert n > w, "Total bits n must be greater than active bits w"
        self.n = n
        self.w = w
        self.minval = minval
        self.maxval = maxval
        self.resolution = (maxval - minval) / (n - w)

    def encode(self, value: float) -> np.ndarray:
        if value < self.minval:
            value = self.minval
        elif value > self.maxval:
            value = self.maxval

        offset = int((value - self.minval) / self.resolution)
        output = np.zeros(self.n, dtype=int)
        output[offset:offset + self.w] = 1
        return output

    def get_bucket_index(self, value: float) -> int:
        """Returns the index of the first 1 in the SDR."""
        if value < self.minval:
            value = self.minval
        elif value > self.maxval:
            value = self.maxval
        return int((value - self.minval) / self.resolution)

    def get_sdr_length(self) -> int:
        return self.n


# Example usage (to remove/comment in production):
if __name__ == "__main__":
    encoder = ScalarEncoder(n=100, w=21, minval=0, maxval=10)
    sdr = encoder.encode(5.3)
    print(sdr)
    print("Bucket index:", encoder.get_bucket_index(5.3))


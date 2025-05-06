import numpy as np


class RDSE:
    def __init__(self, min_val: float, max_val: float, n: int = 100, w: int = 21, resolution: float = None):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        if w >= n:
            raise ValueError("w must be smaller than n")

        self.min_val = min_val
        self.max_val = max_val
        self.n = n
        self.w = w
        self.range = max_val - min_val

        if resolution:
            self.resolution = resolution
            self.num_buckets = int(round(self.range / resolution))
        else:
            self.num_buckets = n - w + 1
            self.resolution = self.range / self.num_buckets

    def encode(self, value):
        output = np.zeros(self.n, dtype=np.int8)
        
        # Clip value to min/max
        clipped = max(min(value, self.max_val), self.min_val)

        # Calculate bucket index
        bucket_index = int((clipped - self.min_val) / self.resolution)
        
        # Ensure window fits
        if bucket_index > self.n - self.w:
            bucket_index = self.n - self.w

        start = bucket_index
        end = start + self.w
        output[start:end] = 1
        return output

    def getWidth(self):
        return self.n

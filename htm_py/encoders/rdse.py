import numpy as np

class RDSE:
    def __init__(self, min_val, max_val, n, w):
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
        self.w = w
        self.output_width = n  # Needed by MultiEncoder

        self.num_buckets = n
        self.resolution = (max_val - min_val) / (self.num_buckets - self.w)
        assert self.resolution > 0, "Resolution must be positive"

    def encode(self, value):
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value {value} outside range [{self.min_val}, {self.max_val}]")

        center_bucket = int((value - self.min_val) / self.resolution)
        center_bucket = min(center_bucket, self.num_buckets - 1)

        sdr = np.zeros(self.n, dtype=np.int64)
        half_width = self.w // 2
        for i in range(-half_width, half_width + 1):
            idx = (center_bucket + i) % self.n
            sdr[idx] = 1

        return sdr

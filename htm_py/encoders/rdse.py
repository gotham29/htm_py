import numpy as np

class RDSE:
    def __init__(self, min_val, max_val, n=None, w=21, resolution=None):
        self.min_val = min_val
        self.max_val = max_val
        self.w = w
        self.output_width = n if n is not None else 0  # Required by MultiEncoder

        if resolution is not None:
            self.resolution = resolution
            # Calculate number of buckets to satisfy the resolution and w constraint
            self.num_buckets = int((max_val - min_val) / resolution) + w
            self.n = self.num_buckets
        else:
            # Backward compatibility: use specified n directly
            self.n = n
            self.num_buckets = self.n
            self.resolution = (max_val - min_val) / (self.num_buckets - self.w)
            assert self.resolution > 0, "Resolution must be positive"

        self.output_width = self.n  # Ensure compatibility with MultiEncoder

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

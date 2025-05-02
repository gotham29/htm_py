import numpy as np
import logging

logger = logging.getLogger(__name__)

class RDSE:
    def __init__(self, resolution=None, n=2048, w=40, seed=42, minval=None, maxval=None):
        self.n = n
        self.w = int(w)  # Ensure integer on bits
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.bucket_map = {}

        if resolution is None:
            if minval is None or maxval is None:
                raise ValueError("If resolution is not provided, both minval and maxval must be.")
            # Heuristic: choose resolution so total buckets ≈ 1.5 * n/w
            approx_buckets = 1.5 * (n / self.w)
            self.resolution = (maxval - minval) / approx_buckets
            logger.debug(f"[RDSE] Auto-computed resolution: {self.resolution:.4f}")
        else:
            self.resolution = resolution

    def encode(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"RDSE.encode() expects a numeric value, got {type(value)}")

        bucket_idx = int(round(value / self.resolution))

        if bucket_idx not in self.bucket_map:
            bits = np.zeros(self.n, dtype=bool)
            active_indices = self.random.choice(self.n, self.w, replace=False)
            bits[active_indices] = True
            self.bucket_map[bucket_idx] = bits
            logger.debug(f"[RDSE] Created bucket {bucket_idx} for value={value}")

        return self.bucket_map[bucket_idx]

    def decode(self, sdr):
        raise NotImplementedError("RDSE decoding is not implemented.")

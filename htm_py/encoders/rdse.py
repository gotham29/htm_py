# import numpy as np

# class RDSE:
#     def __init__(self, size=563, resolution=0.88, min_val=0.0, max_val=100.0, seed=42):
#         self.resolution = resolution
#         self.seed = seed
#         self.min_val = min_val
#         self.max_val = max_val
#         self.totalBits = size
#         self.bucket_map = {}

#     @property
#     def size(self):
#         return self.totalBits

#     def encode(self, value):
#         if isinstance(value, (np.ndarray, list)):
#             value = np.array(value).flatten()[0]
#         value = float(value)

#         bucket = int(value / self.resolution)  # ← Numenta-correct (no min_val subtraction)

#         if bucket not in self.bucket_map:
#             np.random.seed(self.seed + bucket)
#             indices = np.random.choice(self.totalBits, 21, replace=False)
#             self.bucket_map[bucket] = indices

#         indices = self.bucket_map[bucket]
#         encoding = np.zeros(self.totalBits, dtype=np.int8)
#         encoding[indices] = 1

#         return np.ravel(encoding)

#     def _bucket_for_value(self, value):
#         return int((value - self.min_val) / self.resolution)


import numpy as np

class RDSE:
    def __init__(self, size, resolution=0.88, seed=42, min_val=0.0, max_val=100.0):
        self.resolution = resolution
        self.seed = seed
        self._size = size
        self.totalBits = size
        self.min_val = min_val
        self.max_val = max_val
        self.bucket_map = {}

    @property
    def size(self):
        return self.totalBits

    def _bucket_for_value(self, value):
        return int((value - self.min_val) / self.resolution)

    def encode(self, value):
        if isinstance(value, (np.ndarray, list)):
            value = np.array(value).flatten()[0]
        value = float(value)

        bucket = self._bucket_for_value(value)

        if bucket not in self.bucket_map:
            np.random.seed(self.seed + bucket)
            indices = np.random.choice(self.totalBits, 21, replace=False)
            self.bucket_map[bucket] = indices

        indices = self.bucket_map[bucket]
        encoding = np.zeros(self.totalBits, dtype=np.int8)
        encoding[indices] = 1

        return np.ravel(encoding)

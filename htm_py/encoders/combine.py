# htm_py/encoders/combine.py

import numpy as np

def combine_encodings(encodings):
    """
    Combine multiple 1D encoding vectors into a single SDR.
    Pads encodings to ensure proper concatenation and avoids dtype mismatch.

    Args:
        encodings (list of np.ndarray): List of 1D encoded arrays.

    Returns:
        np.ndarray: Combined 1D encoding (SDR).
    """
    arrays = []
    for i, enc in enumerate(encodings):
        if isinstance(enc, list):  # fix for DateEncoder output
            enc = np.array(enc, dtype=np.int8)

        if not isinstance(enc, np.ndarray) or enc.ndim != 1:
            raise ValueError(f"Encoding #{i + 1} is not a 1D numpy array")

        arrays.append(enc.astype(np.int8))

    return np.concatenate(arrays)

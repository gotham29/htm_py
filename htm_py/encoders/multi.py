import numpy as np

class MultiEncoder:
    def __init__(self, encoders):
        """
        encoders: dict of {feature_name: encoder_object}
        """
        self.encoders = encoders
        self.output_width = sum(enc.output_width for enc in encoders.values())

    def encode(self, input_dict):
        """
        input_dict: {feature_name: value, ...}
        Returns a flat SDR vector (numpy array) with all encoded bits.
        """
        assert isinstance(input_dict, dict), "Input must be a dict of feature: value"
        vectors = []

        for key, encoder in self.encoders.items():
            val = input_dict.get(key)
            assert val is not None, f"Missing input value for '{key}'"
            vec = encoder.encode(val)
            vectors.append(vec)

        return np.concatenate(vectors)

    def getEncodedFeatures(self):
        return list(self.encoders.keys())

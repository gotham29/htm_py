import numpy as np

class MultiEncoder:
    def __init__(self, encoders):
        """
        Args:
            encoders (dict): Mapping from feature name to encoder instances (e.g., RDSE, DateEncoder).
        """
        self.encoders = encoders
        self.output_width = sum(encoder.output_width for encoder in encoders.values())

    def encode(self, input_data):
        """
        Args:
            input_data (dict): Mapping from feature name to value.

        Returns:
            np.array: Concatenated SDR encoding across all features.
        """
        encoded_pieces = []
        for feature, encoder in self.encoders.items():
            value = input_data.get(feature)
            if value is None:
                raise ValueError(f"Missing input value for feature '{feature}'")
            encoded_pieces.append(encoder.encode(value))

        return np.concatenate(encoded_pieces)

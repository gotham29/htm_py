import numpy as np

class MultiEncoder:
    def __init__(self, field_encoders):
        """
        field_encoders: dict of {field_name: encoder_instance}
        """
        self.field_encoders = field_encoders
        self._field_slices = {}
        self._total_width = 0

        # Validate encoder widths and record slices
        start = 0
        for name, encoder in self.field_encoders.items():
            if hasattr(encoder, "getWidth"):
                width = encoder.getWidth()
            elif hasattr(encoder, "get_width"):
                width = encoder.get_width()
            else:
                raise ValueError(f"Encoder for '{name}' missing getWidth() method")

            self._field_slices[name] = slice(start, start + width)
            start += width
        self._total_width = start

    def encode(self, input_dict):
        result = np.zeros(self._total_width)

        for name, encoder in self.field_encoders.items():
            if name not in input_dict:
                raise KeyError(f"Missing input field: '{name}'")
            encoded = encoder.encode(input_dict[name])
            result[self._field_slices[name]] = encoded

        return result

    def getWidth(self):
        return self._total_width

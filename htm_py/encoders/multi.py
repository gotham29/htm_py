import numpy as np

class MultiEncoder:
    def __init__(self, field_encoders):
        """
        field_encoders: dict of {field_name: encoder instance}
        """
        self.field_encoders = field_encoders

        self._field_slices = {}
        self._total_width = 0
        for name, encoder in field_encoders.items():
            width = int(encoder.size)
            self._field_slices[name] = slice(self._total_width, self._total_width + width)
            self._total_width += width

    @property
    def size(self):
        return self._total_width

    def encode(self, values_dict):
        """
        values_dict: dict of {field_name: raw_value}
        """
        full_encoding = np.zeros(self._total_width, dtype=np.int8)
        for name, value in values_dict.items():
            encoder = self.field_encoders[name]
            encoding = encoder.encode(value)
            full_encoding[self._field_slices[name]] = encoding
        return full_encoding

import numpy as np

class MultiEncoder:
    def __init__(self, field_encoders):
        """
        field_encoders: dict of {field_name: encoder_instance}
        """
        self.field_encoders = field_encoders
        self._field_slices = {}
        self._total_width = 0

        start = 0
        for name, encoder in self.field_encoders.items():
            if encoder is None:
                raise ValueError(f"Encoder for '{name}' is None. Please instantiate all encoders before passing.")

            if hasattr(encoder, "getWidth"):
                width = encoder.getWidth()
            elif hasattr(encoder, "get_width"):
                width = encoder.get_width()
            else:
                raise ValueError(f"Encoder for '{name}' missing getWidth() method")

            self._field_slices[name] = slice(start, start + width)
            self._total_width += width
            start += width

    def getWidth(self):
        return self._total_width

    def encode(self, values: dict):
        """
        values: dict of {field_name: input_value}
        """
        encoded_fields = []
        for name, encoder in self.field_encoders.items():
            if name not in values:
                raise ValueError(f"Missing value for encoder field '{name}'")
            encoded = encoder.encode(values[name])
            encoded_fields.append(encoded)
        return np.concatenate(encoded_fields)

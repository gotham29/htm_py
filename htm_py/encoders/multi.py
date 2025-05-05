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
            width = encoder.get_width()
            self._field_slices[name] = slice(self._total_width, self._total_width + width)
            self._total_width += width
        self._total_width = int(self._total_width)

    def encode(self, values_dict):
        full_encoding = np.zeros(self._total_width, dtype=np.int8)
        for name in self.field_encoders:
            if name not in values_dict:
                raise KeyError(f"Missing value for field '{name}'")
            encoder = self.field_encoders[name]
            encoding = encoder.encode(values_dict[name])
            full_encoding[self._field_slices[name]] = encoding
        return full_encoding

    def add_encoder(self, name, encoder):
        """Dynamically add a new encoder and update field slices."""
        width = encoder.get_width() if hasattr(encoder, "get_width") else encoder.n
        self._field_slices[name] = slice(self._total_width, self._total_width + width)
        self._total_width += width
        self.field_encoders[name] = encoder

    def get_width(self):
        total = 0
        if self.timeOfDay:
            total += int(self.timeOfDay[0])
        if self.dayOfWeek:
            total += int(self.dayOfWeek[0])
        return total

    @property
    def size(self):
        return self._total_width
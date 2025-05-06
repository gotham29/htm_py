import numpy as np
import datetime
from math import pi, sin, cos


class DateEncoder:
    def __init__(self, timeOfDay=None, dayOfWeek=None, weekend=None):
        """
        Parameters:
        - timeOfDay: tuple (n, w) where n is total bits, w is active bits
        - dayOfWeek: tuple (n, w)
        - weekend: True, None, or tuple (n, w)
        """
        self.timeOfDay = timeOfDay
        self.dayOfWeek = dayOfWeek

        if weekend is True:
            self.weekend = (2, 1)
        elif isinstance(weekend, tuple) and len(weekend) == 2:
            self.weekend = weekend
        elif weekend is None:
            self.weekend = None
        else:
            raise ValueError("`weekend` must be a tuple like (n, w) or True or None")

    def getWidth(self):
        width = 0
        if self.timeOfDay:
            width += self.timeOfDay[0]
        if self.dayOfWeek:
            width += self.dayOfWeek[0]
        if self.weekend:
            width += self.weekend[0]
        return width

    def encode(self, dt: datetime.datetime):
        if not isinstance(dt, datetime.datetime):
            raise ValueError("DateEncoder expects a datetime.datetime input")

        encodings = []

        if self.timeOfDay:
            n, w = self.timeOfDay
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            max_seconds = 24 * 3600
            ratio = seconds / max_seconds
            radians = 2 * pi * ratio
            encoding = self._encode_sin_cos(n, w, radians)
            encodings.append(encoding)

        if self.dayOfWeek:
            n, w = self.dayOfWeek
            radians = 2 * pi * (dt.weekday() / 7.0)
            encoding = self._encode_sin_cos(n, w, radians)
            encodings.append(encoding)

        if self.weekend:
            n, w = self.weekend
            is_weekend = int(dt.weekday() >= 5)
            encoding = np.zeros(n)
            if is_weekend:
                start = (n - w) // 2
                encoding[start:start + w] = 1
            encodings.append(encoding)

        return np.concatenate(encodings) if encodings else np.array([])

    def _encode_sin_cos(self, n, w, radians):
        """Create SDR with a 1D Gaussian bump centered around sin/cos mapping of radians."""
        w = int(round(w))  # Ensure w is an int
        center = int((sin(radians) + 1) / 2 * (n - w))
        sdr = np.zeros(n)
        sdr[center:center + w] = 1
        return sdr


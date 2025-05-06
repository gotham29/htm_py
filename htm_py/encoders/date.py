import numpy as np
import datetime
import math

class DateEncoder:
    def __init__(self, timeOfDay=None, dayOfWeek=None, weekend=None):
        self.timeOfDay = timeOfDay
        self.dayOfWeek = dayOfWeek

        if weekend is True:
            self.weekend = (2, 1)  # Default (n, w)
        elif isinstance(weekend, tuple) and len(weekend) == 2:
            self.weekend = weekend
        elif weekend is None or weekend is False:
            self.weekend = None
        else:
            raise ValueError("`weekend` must be a tuple like (n, w) or True or None")

    def encode(self, dt: datetime.datetime):
        if not isinstance(dt, datetime.datetime):
            raise ValueError("Input must be a datetime.datetime object")

        encodings = []

        if self.timeOfDay:
            n, w = self.timeOfDay
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            total_seconds = 86400  # seconds in a day
            val = float(seconds) / total_seconds
            encodings.append(self._encode_scalar(val, n, w, periodic=True))

        if self.dayOfWeek:
            n, w = self.dayOfWeek
            val = float(dt.weekday()) / 7.0
            encodings.append(self._encode_scalar(val, n, w, periodic=True))

        if self.weekend:
            n, w = self.weekend
            is_weekend = int(dt.weekday() >= 5)
            enc = np.zeros(n)
            if is_weekend:
                center = n // 2
                half_w = w // 2
                enc[center - half_w: center + half_w + 1] = 1
            encodings.append(enc)

        if not encodings:
            return np.array([])

        return np.concatenate(encodings)

    def _encode_scalar(self, value, n, w, periodic=False):
        """Simple scalar encoder with optional periodic wrap-around."""
        encoding = np.zeros(n)
        if periodic:
            value = value % 1.0  # wrap to [0,1)
        center = int(round(value * n)) % n
        half_width = w // 2
        for i in range(-half_width, half_width + 1):
            idx = (center + i) % n if periodic else center + i
            if 0 <= idx < n:
                encoding[idx] = 1
        return encoding

    def getWidth(self):
        width = 0
        if self.timeOfDay:
            width += self.timeOfDay[0]
        if self.dayOfWeek:
            width += self.dayOfWeek[0]
        if self.weekend:
            width += self.weekend[0]
        return width

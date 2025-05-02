import numpy as np
import math
from datetime import datetime


# "c0_timeOfDay": {
#   "type": "DateEncoder",
#   "timeOfDay": [21, 9.49],
#   "fieldname": "c0",
#   "name": "c0"
# }


class DateEncoder:
    def __init__(
        self,
        season=31,
        dayOfWeek=7,
        weekend=2,
        timeOfDay=21,
        holidays=None,
        seed=42,
    ):
        self.seasonBits = season
        self.dayOfWeekBits = dayOfWeek
        self.weekendBits = weekend
        self.timeOfDayBits = timeOfDay
        self.holidays = holidays if holidays is not None else []
        self.random = np.random.RandomState(seed)

        # Initialize encodings
        self.totalBits = self.seasonBits + self.dayOfWeekBits + self.weekendBits + self.timeOfDayBits
        self.outputBits = np.zeros(self.totalBits, dtype=bool)

    def encode(self, dt):
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)

        output = np.zeros(self.totalBits, dtype=bool)
        i = 0

        # Encode season (month of year)
        monthOfYear = dt.month - 1  # 0–11
        self._encodeCategory(monthOfYear, 12, output, i, self.seasonBits)
        i += self.seasonBits

        # Encode day of week
        dayOfWeek = dt.weekday()  # 0–6
        self._encodeCategory(dayOfWeek, 7, output, i, self.dayOfWeekBits)
        i += self.dayOfWeekBits

        # Encode weekend
        weekend = 1 if dayOfWeek >= 5 else 0
        self._encodeCategory(weekend, 2, output, i, self.weekendBits)
        i += self.weekendBits

        # Encode time of day
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        self._encodeScalar(seconds, 24 * 3600, output, i, self.timeOfDayBits)

        return output

    def _encodeCategory(self, value, buckets, output, offset, nbits):
        bitsPerBucket = nbits // buckets
        start = offset + value * bitsPerBucket
        output[start : start + bitsPerBucket] = True

    def _encodeScalar(self, value, maxValue, output, offset, nbits):
        radians = (value / maxValue) * 2 * math.pi
        center = int((math.sin(radians) + 1) / 2 * (nbits - 1))
        width = max(1, nbits // 20)

        for j in range(center - width, center + width + 1):
            idx = offset + (j % nbits)
            output[idx] = True

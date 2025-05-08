import math
import numpy as np
from datetime import datetime


class DateEncoder:
    def __init__(self, timeOfDay=(21, 9.49), strptime_str="%Y-%m-%d %H:%M:%S"):
        """
        timeOfDay: (n, w) → number of bits, number of active bits in the encoding
        """
        self.timeOfDay = timeOfDay
        self.strptime_str = strptime_str
        self.output_width = timeOfDay[0]

    def encode(self, timestamp):
        n, w = self.timeOfDay
        w = int(round(w))
        half_w = w // 2

        # 🩹 Fix: auto-parse string timestamps
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, self.strptime_str)

        time_of_day = timestamp.hour + timestamp.minute / 60.0
        bucket = int(time_of_day / 24.0 * n)

        sdr = [0] * n
        for i in range(w):
            index = (bucket - half_w + i) % n
            sdr[index] = 1

        return np.array(sdr, dtype=np.int64)



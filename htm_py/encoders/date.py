# htm_py/encoders/date.py

import datetime
import math

class DateEncoder:
    def __init__(self, timeOfDay=(400, 21), weekend=None, dayOfWeek=None):  #weekend=(2, 1)
        self.timeOfDay = timeOfDay
        self.weekend = weekend
        self.dayOfWeek = dayOfWeek

    def encode(self, dt):
        if not isinstance(dt, datetime.datetime):
            raise ValueError("Input to DateEncoder.encode must be a datetime.datetime object")

        output = []

        if self.timeOfDay:
            n, w = self.timeOfDay
            total_minutes = dt.hour * 60 + dt.minute
            output.extend(self._cyclic_scalar_encode(total_minutes, 1440, n, w))

        if self.dayOfWeek:
            n, w = self.dayOfWeek
            day = dt.weekday()
            output.extend(self._cyclic_scalar_encode(day, 7, n, w))

        if self.weekend:
            is_weekend = int(dt.weekday() >= 5)  # 1 if Saturday/Sunday
            weekend_bits = [0] * self.weekend[0]
            if self.weekend[0] >= 2:
                weekend_bits[0] = 1 - is_weekend
                weekend_bits[1] = is_weekend
            output.extend(weekend_bits)

        return output

    def _cyclic_scalar_encode(self, value, period, n, w):
        encoding = [0] * n
        center = int((float(value % period) / period) * n)
        half_width = w // 2
        for i in range(-half_width, half_width + 1):
            idx = (center + i) % n
            encoding[idx] = 1
        return encoding

    def get_width(self):
        total = 0
        if self.timeOfDay:
            total += self.timeOfDay[0]
        # if self.dayOfWeek:
        #     total += self.dayOfWeek[0]
        # if self.weekend:
        #     total += self.weekend[0]
        return total

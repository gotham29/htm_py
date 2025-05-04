# # # import numpy as np
# # # import math
# # # from datetime import datetime


# # # # "c0_timeOfDay": {
# # # #   "type": "DateEncoder",
# # # #   "timeOfDay": [21, 9.49],
# # # #   "fieldname": "c0",
# # # #   "name": "c0"
# # # # }


# # # class DateEncoder:
# # #     def __init__(self, timeOfDay=None, dayOfWeek=None, weekend=None, season=None):
# # #         self.timeOfDay = timeOfDay  # (bits, radius)
# # #         self.dayOfWeek = dayOfWeek
# # #         self.weekend = weekend
# # #         self.season = season
# # #         # Numenta params: timeOfDay=(21, 9.49), weekend=1
# # #         self.timeOfDayBits = timeOfDay[0] if isinstance(timeOfDay, tuple) else timeOfDay
# # #         self.timeOfDayRadius = timeOfDay[1] if isinstance(timeOfDay, tuple) else 1.0
# # #         self.weekendBits = weekend
# # #         self.seasonBits = 0
# # #         self.dayOfWeekBits = 0
        
# # #         self.totalBits = self.timeOfDayBits + self.weekendBits + self.seasonBits + self.dayOfWeekBits

# # #     def encode(self, dt):
# # #         if isinstance(dt, str):
# # #             dt = datetime.fromisoformat(dt)

# # #         output = np.zeros(self.totalBits, dtype=bool)
# # #         i = 0

# # #         # Encode season (month of year)
# # #         monthOfYear = dt.month - 1  # 0–11
# # #         self._encodeCategory(monthOfYear, 12, output, i, self.seasonBits)
# # #         i += self.seasonBits

# # #         # Encode day of week
# # #         dayOfWeek = dt.weekday()  # 0–6
# # #         self._encodeCategory(dayOfWeek, 7, output, i, self.dayOfWeekBits)
# # #         i += self.dayOfWeekBits

# # #         # Encode weekend
# # #         weekend = 1 if dayOfWeek >= 5 else 0
# # #         self._encodeCategory(weekend, 2, output, i, self.weekendBits)
# # #         i += self.weekendBits

# # #         # Encode time of day
# # #         seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
# # #         self._encodeScalar(seconds, 24 * 3600, output, i, self.timeOfDayBits)

# # #         return output

# # #     def _encodeCategory(self, value, buckets, output, offset, nbits):
# # #         bitsPerBucket = nbits // buckets
# # #         start = offset + value * bitsPerBucket
# # #         output[start : start + bitsPerBucket] = True

# # #     def _encodeScalar(self, value, maxValue, output, offset, nbits):
# # #         radians = (value / maxValue) * 2 * math.pi
# # #         center = int((math.sin(radians) + 1) / 2 * (nbits - 1))
# # #         width = max(1, nbits // 20)

# # #         for j in range(center - width, center + width + 1):
# # #             idx = offset + (j % nbits)
# # #             output[idx] = True

# # #     @property
# # #     def size(self):
# # #         return self.totalBits

# # #     @property
# # #     def output_size(self):
# # #         return self.totalBits



# # import numpy as np
# # from datetime import datetime


# # class DateEncoder:
# #     def __init__(self, timeOfDay=(21, 9.49), weekend=1):
# #         self.timeOfDayBits, self.timeOfDayRadius = timeOfDay
# #         self.weekendBits = weekend
# #         self.totalBits = self.timeOfDayBits + self.weekendBits

# #     def encode(self, dt):
# #         bits = []

# #         if self.timeOfDayBits:
# #             timeOfDay = ((dt.hour * 60 + dt.minute) % (24 * 60)) / (24 * 60.0)
# #             bits.extend(self._encode_scalar(timeOfDay, self.timeOfDayBits))

# #         if self.weekendBits:
# #             weekend = 1.0 if dt.weekday() >= 5 else 0.0
# #             bits.extend(self._encode_scalar(weekend, self.weekendBits))

# #         return np.array(bits, dtype=np.int8)  # ← FIXED

# #     @property
# #     def output_size(self):
# #         return self.totalBits



# import numpy as np
# from datetime import datetime

# class DateEncoder:
#     def __init__(self, timeOfDay=None, dayOfWeek=None, season=None, weekend=None):
#         self.timeOfDayBits = timeOfDay[0] if timeOfDay else 0
#         self.timeOfDayRadius = timeOfDay[1] if timeOfDay else 0

#         self.dayOfWeekBits = dayOfWeek[0] if dayOfWeek else 0
#         self.dayOfWeekRadius = dayOfWeek[1] if dayOfWeek else 0

#         self.seasonBits = season[0] if season else 0
#         self.seasonRadius = season[1] if season else 0

#         self.weekendBits = weekend if weekend else 0

#         self.totalBits = self.timeOfDayBits + self.dayOfWeekBits + self.seasonBits + self.weekendBits

#     def _encode_scalar(self, value, bits, radius, periodic=True):
#         """Encodes a scalar value into a binary array using a circular (periodic) or linear encoding."""
#         if bits == 0:
#             return np.zeros(0, dtype=np.int8)

#         encoding = np.zeros(bits, dtype=np.int8)
#         if radius == 0:
#             return encoding  # can't encode if radius is 0

#         center = int((float(value) / radius) % bits if periodic else min(bits, float(value) / radius))
#         half_width = int(bits / 10) or 1

#         for i in range(-half_width, half_width + 1):
#             idx = (center + i) % bits
#             encoding[idx] = 1

#         return encoding

#     def encode(self, timestamp: datetime):
#         encodings = []

#         if self.timeOfDayBits:
#             seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
#             seconds_per_day = 86400
#             timeOfDay = seconds / seconds_per_day
#             enc = self._encode_scalar(timeOfDay, self.timeOfDayBits, self.timeOfDayRadius)
#             encodings.append(enc)

#         if self.dayOfWeekBits:
#             dayOfWeek = timestamp.weekday()
#             enc = self._encode_scalar(dayOfWeek, self.dayOfWeekBits, self.dayOfWeekRadius, periodic=True)
#             encodings.append(enc)

#         if self.seasonBits:
#             dayOfYear = timestamp.timetuple().tm_yday
#             enc = self._encode_scalar(dayOfYear, self.seasonBits, self.seasonRadius, periodic=True)
#             encodings.append(enc)

#         if self.weekendBits:
#             is_weekend = int(timestamp.weekday() >= 5)
#             encoding = np.zeros(self.weekendBits, dtype=np.int8)
#             if is_weekend:
#                 center = self.weekendBits // 2
#                 width = max(1, self.weekendBits // 10)
#                 for i in range(-width, width + 1):
#                     idx = (center + i) % self.weekendBits
#                     encoding[idx] = 1
#             encodings.append(encoding)

#         return np.concatenate(encodings) if encodings else np.zeros(0, dtype=np.int8)

#     @property
#     def size(self):
#         return self.timeOfDayBits[0] + (1 if self.weekendBits else 0)


import numpy as np
from datetime import datetime
import math

class DateEncoder:
    def __init__(self, timeOfDay=None, dayOfWeek=None, season=None, weekend=None):
        self.timeOfDayBits = timeOfDay[0] if timeOfDay else 0
        self.timeOfDayRadius = timeOfDay[1] if timeOfDay else 0

        self.dayOfWeekBits = dayOfWeek[0] if dayOfWeek else 0
        self.dayOfWeekRadius = dayOfWeek[1] if dayOfWeek else 0

        self.seasonBits = season[0] if season else 0
        self.seasonRadius = season[1] if season else 0

        self.weekendBits = weekend if weekend else 0

        self.totalBits = self.timeOfDayBits + self.dayOfWeekBits + self.seasonBits + self.weekendBits

    @property
    def size(self):
        return self.totalBits

    def encode(self, timestamp: datetime):
        sdr = []

        # Time of day encoding
        if self.timeOfDayBits:
            minutes = timestamp.hour * 60 + timestamp.minute
            day_minutes = 24 * 60
            radians = (2 * math.pi * minutes) / day_minutes
            phase = np.cos(radians) + np.sin(radians)
            center_bin = int((radians / (2 * math.pi)) * self.timeOfDayBits)
            vec = np.zeros(self.timeOfDayBits, dtype=np.int8)
            for i in range(self.timeOfDayBits):
                distance = min(abs(i - center_bin), self.timeOfDayBits - abs(i - center_bin))
                if distance <= self.timeOfDayRadius:
                    vec[i] = 1
            sdr.append(vec)

        # Weekend encoding
        if self.weekendBits:
            is_weekend = 1 if timestamp.weekday() >= 5 else 0
            vec = np.zeros(self.weekendBits, dtype=np.int8)
            if is_weekend:
                vec[0] = 1
            sdr.append(vec)

        return np.concatenate(sdr) if sdr else np.array([], dtype=np.int8)

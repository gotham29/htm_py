import unittest
import datetime
import numpy as np
from htm_py.encoders.date import DateEncoder

class TestDateEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = DateEncoder(
            timeOfDay=(1000, 21),  # large n for good resolution
            dayOfWeek=(21, 3),
            weekend=(2, 1)
        )

    def test_encode_datetime(self):
        dt = datetime.datetime(2024, 5, 5, 13, 30)  # Sunday
        encoding = self.encoder.encode(dt)
        self.assertIsInstance(encoding, np.ndarray)
        self.assertEqual(encoding.ndim, 1)
        self.assertEqual(len(encoding), self.encoder.getWidth())

    def test_encode_day_vs_night(self):
        day = datetime.datetime(2024, 5, 5, 13, 0)
        night = datetime.datetime(2024, 5, 5, 2, 0)
        e1 = self.encoder.encode(day)
        e2 = self.encoder.encode(night)
        self.assertTrue(np.any(e1 != e2))

    def test_encode_weekday_vs_weekend(self):
        weekday = datetime.datetime(2024, 5, 6)  # Monday
        weekend = datetime.datetime(2024, 5, 5)  # Sunday
        e1 = self.encoder.encode(weekday)
        e2 = self.encoder.encode(weekend)
        self.assertTrue(np.any(e1 != e2))

    def test_empty_encoder(self):
        no_enc = DateEncoder(timeOfDay=None, dayOfWeek=None, weekend=None)
        dt = datetime.datetime.now()
        encoding = no_enc.encode(dt)
        self.assertEqual(len(encoding), 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.encoder.encode("2024-05-05")  # not a datetime object

if __name__ == "__main__":
    unittest.main()

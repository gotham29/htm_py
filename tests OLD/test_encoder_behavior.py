
# import unittest
# import numpy as np
# from htm_py.encoders.rdse import RDSE
# from htm_py.encoders.date import DateEncoder
# from htm_py.encoders.multi import MultiEncoder
# from datetime import datetime

# class TestEncoderBehavior(unittest.TestCase):



import unittest
import numpy as np
from datetime import datetime
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder

class TestEncoderBehavior(unittest.TestCase):
    def test_rdse_output_shape(self):
        rdse = RDSE(resolution=0.5, size=2048)
        output = rdse.encode(5.0)
        self.assertEqual(len(output), rdse.size)
        self.assertTrue(np.sum(output) > 0)

    def test_rdse_multiple_values(self):
        rdse = RDSE(resolution=0.01, size=2048)
        a = rdse.encode(10.0)
        b = rdse.encode(10.1)
        self.assertTrue(np.any(a != b))  # Slight change → different encoding

    def test_date_encoder_output_shape(self):
        encoder = DateEncoder(timeOfDay=(21, 9.49), weekend=1)
        now = datetime(2025, 5, 2, 12, 0)
        output = encoder.encode(now)
        self.assertEqual(len(output), encoder.totalBits)
        self.assertTrue(np.sum(output) > 0)

    def test_rdse_similarity_for_close_values(self):
        rdse = RDSE(size=563, resolution=0.5, seed=42)
        v1 = rdse.encode(10.0)
        v2 = rdse.encode(10.4)
        v3 = rdse.encode(10.5)  # new bucket
        v4 = rdse.encode(11.0)  # definitely new bucket

        overlap_12 = np.sum(v1 & v2)
        overlap_13 = np.sum(v1 & v3)
        overlap_14 = np.sum(v1 & v4)

        print("Overlaps:", overlap_12, overlap_13, overlap_14)
        self.assertGreater(overlap_12, overlap_13)
        self.assertGreater(overlap_13, overlap_14)

    def test_date_encoder_timeofday_bits(self):
        enc = DateEncoder(timeOfDay=(21, 3), weekend=0)
        ts = datetime(2023, 5, 3, 12, 0)
        encoded = enc.encode(ts)
        self.assertEqual(len(encoded), 21)
        self.assertEqual(np.sum(encoded), 7)  # 2*radius+1 = 7

    def test_date_encoder_with_weekend(self):
        enc = DateEncoder(timeOfDay=(21, 3), weekend=1)
        ts = datetime(2023, 5, 6, 12, 0)  # Saturday
        encoded = enc.encode(ts)
        self.assertEqual(len(encoded), 22)
        self.assertEqual(encoded[-1], 1)

    def test_multi_encoder_combines_fields_correctly(self):
        rdse = RDSE(size=563, resolution=0.88, seed=42)
        date = DateEncoder(timeOfDay=(21, 3), weekend=1)
        enc = MultiEncoder({"rdse": rdse, "date": date})

        ts = datetime(2023, 5, 6, 12, 0)
        combined = enc.encode({"rdse": 13.5, "date": ts})
        self.assertEqual(len(combined), rdse.size + date.size)
        self.assertEqual(np.sum(combined), 21 + 7 + 1)  # RDSE 21 + date (7 time + 1 weekend)

    def test_rdse_output_size(self):
        rdse = RDSE(size=563, resolution=0.88, seed=42)
        v = rdse.encode(12.34)
        self.assertEqual(len(v), 563)
        self.assertEqual(np.sum(v), 21)

    def test_rdse_is_deterministic(self):
        rdse1 = RDSE(size=563, resolution=0.88, seed=123)
        rdse2 = RDSE(size=563, resolution=0.88, seed=123)
        v1 = rdse1.encode(25.0)
        v2 = rdse2.encode(25.0)
        np.testing.assert_array_equal(v1, v2)

    def test_rdse_bucket_alignment(self):
        rdse = RDSE(size=563, resolution=0.88, seed=99)
        v1 = rdse.encode(10.0)
        v2 = rdse.encode(10.0 + 0.01)
        np.testing.assert_array_equal(v1, v2)

    def test_rdse_bucket_shift(self):
        rdse = RDSE(size=563, resolution=0.88, seed=99)
        v1 = rdse.encode(10.0)
        v2 = rdse.encode(10.89)  # Slightly more than 1 resolution unit
        self.assertFalse(np.array_equal(v1, v2))

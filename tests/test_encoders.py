import unittest
import numpy as np
from datetime import datetime
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder


class TestEncoders(unittest.TestCase):
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

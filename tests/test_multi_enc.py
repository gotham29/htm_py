import unittest
import numpy as np
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder
import datetime


enc_params = {
    "rdse": {"n": 563, "w": 21, "min_val": 0, "max_val": 100},  #"resolution": 0.88
    "date": {"timeOfDay": (21, 9.49), "weekend": 1}
}

ENCODERS = {}
if "rdse" in enc_params:
    rdse_cfg = enc_params["rdse"]
    ENCODERS["rdse"] = RDSE(
        n=rdse_cfg.get("n", 100),
        w=rdse_cfg.get("w", 0.88),
        min_val=rdse_cfg.get("min_val", 0.0),
        max_val=rdse_cfg.get("max_val", 100.0),
    )
if "date" in enc_params:
    date_cfg = enc_params["date"]
    ENCODERS["date"] = DateEncoder(
        timeOfDay=date_cfg.get("encode_time_of_day", (21, 9.49)),
        weekend=date_cfg.get("weekend", 1)
    )

class TestMultiEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = MultiEncoder({
            "feature1": RDSE(min_val=0, max_val=100),
            "feature2": RDSE(min_val=0, max_val=1, n=21, w=3),
            "timestamp": DateEncoder(timeOfDay=(21, 3), dayOfWeek=(7, 2), weekend=(2, 1))
        })

    def test_encoding_output_length(self):
        sample = {
            "feature1": 25,
            "feature2": 0.5,
            "timestamp": datetime.datetime(2024, 5, 5, 12, 30)
        }
        output = self.encoder.encode(sample)
        expected_length = len(output)
        self.assertEqual(len(output), expected_length)

    def test_encoding_consistency(self):
        sample = {
            "feature1": 50,
            "feature2": 0.1,
            "timestamp": datetime.datetime(2024, 5, 6, 9, 0)
        }
        out1 = self.encoder.encode(sample)
        out2 = self.encoder.encode(sample)
        np.testing.assert_array_equal(out1, out2)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            self.encoder.encode({"feature1": 10})  # Missing feature2 and timestamp

if __name__ == "__main__":
    unittest.main()

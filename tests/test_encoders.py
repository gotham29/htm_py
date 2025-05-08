
import pytest
import numpy as np
import unittest
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder
from datetime import datetime

class TestEncoders(unittest.TestCase):

    def test_rdse_output_length_and_sparsity(self):
        rdse = RDSE(min_val=0, max_val=100, n=130, w=21)
        sdr = rdse.encode(42.0)
        assert len(sdr) == 130, "RDSE output should have length 130"
        assert sum(sdr) == 21, "RDSE output should have 21 active bits"

    def test_rdse_overlap_for_close_values(self):
        rdse = RDSE(min_val=0, max_val=100, n=130, w=21)
        sdr1 = rdse.encode(50.0)
        sdr2 = rdse.encode(51.0)
        overlap = sum(a & b for a, b in zip(sdr1, sdr2))
        assert overlap > 10, "RDSE encodings for close values should overlap significantly"

    def test_rdse_no_overlap_for_far_values(self):
        rdse = RDSE(min_val=0, max_val=100, n=130, w=21)
        sdr1 = rdse.encode(10.0)
        sdr2 = rdse.encode(90.0)
        overlap = sum(a & b for a, b in zip(sdr1, sdr2))
        assert overlap < 5, "RDSE encodings for distant values should have low overlap"

    def test_date_encoder_output_length_and_wraparound(self):
        enc = DateEncoder(timeOfDay=(21, 9))
        midnight = enc.encode(datetime(2024, 1, 1, 0, 0))
        just_before_midnight = enc.encode(datetime(2024, 1, 1, 23, 59))
        assert len(midnight) == 21, "DateEncoder output should match timeOfDay n"
        overlap = sum(a & b for a, b in zip(midnight, just_before_midnight))
        assert overlap > 5, "DateEncoder should produce overlapping SDRs near wraparound"

    def test_date_encoder_variability(self):
        enc = DateEncoder(timeOfDay=(21, 9))
        noon = enc.encode(datetime(2024, 1, 1, 12, 0))
        midnight = enc.encode(datetime(2024, 1, 1, 0, 0))
        overlap = sum(a & b for a, b in zip(noon, midnight))
        assert overlap < 10, "Dissimilar times should produce different encodings"

    def test_multiencoder_combines_components(self):
        rdse1 = RDSE(min_val=0, max_val=100, n=130, w=21)
        rdse2 = RDSE(min_val=0, max_val=50, n=50, w=5)
        date = DateEncoder(timeOfDay=(21, 9))  # Watch out: 9.49 is not valid!
        enc = MultiEncoder({"value": rdse1, "pressure": rdse2, "timestamp": date})
        dt = datetime(2024, 1, 1, 12, 0)
        result = enc.encode({"value": 42.0, "pressure": 10.0, "timestamp": dt})
        
        print("RDSE1 bits:", sum(rdse1.encode(42.0)))
        print("RDSE2 bits:", sum(rdse2.encode(10.0)))
        print("DateEncoder bits:", sum(date.encode(dt)))
        print("Combined bits:", sum(result))
        
        assert len(result) == 201
        assert sum(result) == 35

    def test_missing_input_raises_error(self):
        rdse1 = RDSE(min_val=0, max_val=100, n=130, w=21)
        rdse2 = RDSE(min_val=0, max_val=50, n=50, w=5)
        date = DateEncoder(timeOfDay=(21, 9))  # FIXED: use int for w
        enc = MultiEncoder({"value": rdse1, "pressure": rdse2, "timestamp": date})
        
        with pytest.raises(AssertionError):
            enc.encode({"value": 10.0, "pressure": 5.0})  # missing "timestamp"

    def test_dateencoder_fractional_w_rounds_down(self):
        enc = DateEncoder(timeOfDay=(21, 9.49))
        encoded = enc.encode(datetime.now())
        active_bits = sum(encoded)
        self.assertTrue(8 <= active_bits <= 9, f"Expected ~9 active bits, got {active_bits}")

import unittest
from htm_py.encoders.rdse import RDSE


class TestRDSE(unittest.TestCase):

    def setUp(self):
        self.encoder = RDSE(min_val=0.0, max_val=10.0)

    def test_encode_within_range(self):
        encoding = self.encoder.encode(5.0)
        self.assertEqual(len(encoding), 100)
        self.assertEqual(sum(encoding), 21)

    def test_encode_consistency(self):
        e1 = self.encoder.encode(3.7)
        e2 = self.encoder.encode(3.7)
        self.assertEqual(e1, e2)

    def test_encode_different_inputs(self):
        e1 = self.encoder.encode(1.0)
        e2 = self.encoder.encode(10.0)
        overlap = sum(a & b for a, b in zip(e1, e2))
        self.assertLess(overlap, 21)

    def test_input_out_of_range(self):
        with self.assertRaises(ValueError):
            self.encoder.encode(float("nan"))

    def test_clamping_behavior(self):
        e_low = self.encoder.encode(-100.0)
        e_high = self.encoder.encode(100.0)
        self.assertEqual(sum(e_low), 21)
        self.assertEqual(sum(e_high), 21)
        self.assertEqual(len(e_low), 100)
        self.assertEqual(len(e_high), 100)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from htm_py.spatial_pooler import SpatialPooler

class TestSPOverlap(unittest.TestCase):
    def setUp(self):
        self.sp = SpatialPooler(
            inputDimensions=(8,),
            columnDimensions=(4,),
            potentialPct=1.0,
            globalInhibition=False,
            stimulusThreshold=0,
            synPermConnected=0.5,
            synPermActiveInc=0.05,
            synPermInactiveDec=0.01,
            boostStrength=0.0,
            seed=42
        )
        self.sp._initialize()

    def test_overlap_score_basic(self):
        # All synapses connected, active input → full overlap
        self.sp.permanences[:] = 1.0
        input_vector = np.array([1] * 8, dtype=int)
        overlap = self.sp._compute_overlap(input_vector)
        self.assertTrue(np.all(overlap > 0))

    def test_overlap_score_thresholded(self):
        # Permanences below threshold → overlap = 0
        self.sp.permanences[:] = 0.1  # Below connected threshold
        input_vector = np.array([1] * 8, dtype=int)
        overlap = self.sp._compute_overlap(input_vector)
        self.assertTrue(np.all(overlap == 0))

    def test_overlap_score_with_boost(self):
        self.sp.permanences[:] = 1.0
        self.sp.boostFactors[:] = 2.0
        input_vector = np.array([1] * 8, dtype=int)
        overlap = self.sp._compute_overlap(input_vector)
        expected = np.sum(self.sp.connectedSynapsesMask(), axis=1) * 2.0
        np.testing.assert_array_equal(overlap, expected)

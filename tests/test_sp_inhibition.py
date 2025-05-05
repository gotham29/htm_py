import unittest
import numpy as np
from htm_py.spatial_pooler import SpatialPooler

class TestSPInhibition(unittest.TestCase):
    def setUp(self):
        self.sp = SpatialPooler(
            inputDimensions=(8,),
            columnDimensions=(4,),
            potentialPct=1.0,
            globalInhibition=False,
            localAreaDensity=0.5,
            stimulusThreshold=0,
            synPermConnected=0.5,
            synPermActiveInc=0.05,
            synPermInactiveDec=0.01,
            boostStrength=0.0,
            seed=42
        )
        self.sp._initialize()

    def test_local_inhibition(self):
        overlap = np.array([0.1, 0.8, 0.3, 0.9])
        active = self.sp._inhibit_columns(overlap)
        expected = np.array([False, True, False, True])  # Top 50% in local area
        np.testing.assert_array_equal(active, expected)

    def test_global_inhibition(self):
        self.sp.globalInhibition = True
        overlap = np.array([0.1, 0.8, 0.3, 0.9])
        active = self.sp._inhibit_columns(overlap)
        expected = np.array([False, True, False, True])  # Top 50% globally
        np.testing.assert_array_equal(active, expected)

    def test_stimulus_threshold(self):
        self.sp.stimulusThreshold = 0.5
        overlap = np.array([0.4, 0.8, 0.3, 0.9])
        active = self.sp._inhibit_columns(overlap)
        # Columns below threshold should not be active
        self.assertFalse(active[0])
        self.assertTrue(active[1])
        self.assertFalse(active[2])
        self.assertTrue(active[3])

import unittest
import numpy as np
from htm_py.spatial_pooler import SpatialPooler

class TestSPInhibition(unittest.TestCase):
    def setUp(self):
        self.sp = SpatialPooler(
            inputDimensions=(8,),
            columnDimensions=(4,),
            potentialPct=1.0,
            globalInhibition=True,
            numActiveColumnsPerInhArea=2,
            synPermConnected=0.5,
            synPermActiveInc=0.05,
            synPermInactiveDec=0.01,
            boostStrength=0.0,
            seed=42
        )

    def test_global_inhibition(self):
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        self.assertEqual(len(active_columns), self.sp.numActiveColumnsPerInhArea)

    def test_local_inhibition(self):
        self.sp.globalInhibition = False
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        # In this simple 1D setup, global/local inhibition may behave similarly
        self.assertTrue(0 < len(active_columns) <= self.sp.numActiveColumnsPerInhArea)

    def test_stimulus_threshold(self):
        self.sp.stimulusThreshold = 10  # Too high for 8 inputs
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        self.assertEqual(len(active_columns), 0)

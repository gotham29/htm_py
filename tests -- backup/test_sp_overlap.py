import unittest
import numpy as np
from htm_py.spatial_pooler import SpatialPooler


class TestSPOverlap(unittest.TestCase):
    def setUp(self):
        self.sp = SpatialPooler(
            inputDimensions=(8,),
            columnDimensions=(4,),
            potentialPct=1.0,
            globalInhibition=True,
            numActiveColumnsPerInhArea=2,
            stimulusThreshold=0,
            synPermConnected=0.19,  # Robust threshold
            boostStrength=0.0,
            seed=42
        )

    def test_overlap_score_basic(self):
        # ✅ Use dense input to ensure overlap
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        self.assertTrue(len(active_columns) > 0)

    def test_overlap_score_thresholded(self):
        self.sp._permanences[:] = 0.2  # Ensure all synapses are connected
        self.sp.connectedSynapses = self.sp._permanences >= self.sp.synPermConnected
        self.sp.stimulusThreshold = 1
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        self.assertTrue(len(active_columns) > 0)

    def test_overlap_score_with_boost(self):
        self.sp._permanences[:] = 0.2  # Ensure activity is possible
        self.sp.connectedSynapses = self.sp._permanences >= self.sp.synPermConnected
        self.sp.boostStrength = 1.0
        self.sp.stimulusThreshold = 0
        input_vector = np.array([1] * 8, dtype=np.int8)
        output = np.zeros(self.sp.numColumns, dtype=np.int8)

        for i in range(5):
            self.sp.compute(input_vector, learn=True, output=output, iteration=i)

        active_columns = np.nonzero(output)[0]
        self.assertTrue(len(active_columns) > 0)

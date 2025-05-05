import unittest
import numpy as np
from htm_py.spatial_pooler import SpatialPooler


class TestSPComputeBasic(unittest.TestCase):
    def setUp(self):
        self.sp = SpatialPooler(
            inputDimensions=(10,),
            columnDimensions=(5,),
            potentialPct=1.0,
            globalInhibition=True,
            numActiveColumnsPerInhArea=2,
            synPermConnected=0.2,
            stimulusThreshold=0,
            seed=42
        )
        self.sp.permanences = np.array([
            [0.1, 0.9, 0.9, 0.1, 0.0, 0.5, 0.0, 0.1, 0.1, 0.1],  # overlap = 2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0],  # overlap = 4
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # overlap = 0
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # overlap = 10
            [0.3, 0.2, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.3, 0.2],  # overlap = 6
        ])
        self.sp.connectedSynapses = self.sp.permanences >= self.sp.synPermConnected


    def test_selects_top_k_overlaps(self):
        input_vector = np.array([1] * 10, dtype=np.int8)  # All bits on
        active_columns = self.sp.compute(input_vector)
        expected = set([3, 4])
        
        self.assertEqual(set(active_columns), expected)

    def test_selects_fewer_if_not_enough(self):
        self.sp.numActiveColumnsPerInhArea = 4
        input_vector = np.array([0] * 10, dtype=np.int8)  # All off
        active_columns = self.sp.compute(input_vector)

        self.assertEqual(active_columns, [])  # No columns should be active


if __name__ == "__main__":
    unittest.main()

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

    def test_selects_top_k_overlaps(self):
        input_vector = np.array([1] * 10, dtype=np.int8)  # All bits on
        output = np.zeros(self.sp.numColumns, dtype=np.int8)

        # 👇 Ensure connected synapses by force-setting permanences
        self.sp._permanences[:] = 1.0

        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]

        # Assert number of active columns equals k
        self.assertEqual(len(active_columns), self.sp.numActiveColumnsPerInhArea)
        for col in active_columns:
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, self.sp.numColumns)

    def test_selects_fewer_if_not_enough(self):
        self.sp.numActiveColumnsPerInhArea = 4
        input_vector = np.array([0] * 10, dtype=np.int8)  # All bits off
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(input_vector, learn=False, output=output, iteration=0)
        active_columns = np.nonzero(output)[0]
        self.assertEqual(list(active_columns), [])  # No columns should be active

if __name__ == "__main__":
    unittest.main()

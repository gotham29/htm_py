import unittest
from htm_py.temporal_memory import TemporalMemory


import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTMIdempotenceAndReset(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),  # small domain
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=4,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
        )

    def test_compute_twice_no_learn_same_result(self):
        self.tm.compute([0], learn=False)
        pred1 = set(self.tm.predictiveCells)
        self.tm.compute([0], learn=False)
        pred2 = set(self.tm.predictiveCells)
        self.assertEqual(pred1, pred2)

    def test_reset_clears_predictive_cells(self):
        self.tm.compute([0], learn=True)  # Learn [0] → segment grown
        self.tm.compute([0], learn=True)  # Activate segment → prediction should appear
        self.assertTrue(len(self.tm.predictiveCells) > 0)
        self.tm.reset()
        self.assertEqual(len(self.tm.predictiveCells), 0)

    def test_predictive_cells_do_not_persist(self):
        self.tm.compute([0], learn=True)  # Learn
        self.tm.compute([0], learn=True)  # Form prediction
        pred1 = set(self.tm.predictiveCells)

        self.tm.compute([3], learn=True)  # Learn unrelated input
        self.tm.compute([3], learn=True)  # Form prediction
        pred2 = set(self.tm.predictiveCells)

        print(f"[DEBUG] Pred after column 0: {sorted(pred1)}")
        print(f"[DEBUG] Pred after column 3: {sorted(pred2)}")

        # Prediction sets should differ for unrelated inputs
        self.assertNotEqual(pred1, pred2, "Predictive cells persisted across unrelated inputs")


if __name__ == "__main__":
    unittest.main()

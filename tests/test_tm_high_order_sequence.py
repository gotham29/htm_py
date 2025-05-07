# tests/test_tm_high_order_sequence.py

import unittest
from htm_py.temporal_memory import TemporalMemory

import unittest
from htm_py.temporal_memory import TemporalMemory


class TestHighOrderSequence(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(5,),
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=5
        )

    def _run_sequence(self, col_sequence, learn=True):
        for col in col_sequence:
            self.tm.compute([col], learn=learn)

    def test_high_order_branching_sequence(self):
        # Repeat each sequence 3x to allow full learning
        for _ in range(3):
            self._run_sequence([0, 1, 2])  # A → B → C
            self.tm.reset()
        for _ in range(3):
            self._run_sequence([0, 3, 4])  # A → D → E
            self.tm.reset()

        # Test prediction after A → B
        self._run_sequence([0, 1], learn=False)
        pred_cells_after_B = self.tm.predictiveCells
        pred_cols_B = set(cell // self.tm.cellsPerColumn for cell in pred_cells_after_B)
        print(f"[TEST] Predicted columns after A→B: {sorted(pred_cols_B)}")

        self.tm.reset()

        # Test prediction after A → D
        self._run_sequence([0, 3], learn=False)
        pred_cells_after_D = self.tm.predictiveCells
        pred_cols_D = set(cell // self.tm.cellsPerColumn for cell in pred_cells_after_D)
        print(f"[TEST] Predicted columns after A→D: {sorted(pred_cols_D)}")

        # Assert that column 2 (C) is predicted after A→B
        self.assertIn(2, pred_cols_B, f"Expected C (col 2) to be predicted after A→B, got {pred_cols_B}")
        # Assert that column 4 (E) is predicted after A→D
        self.assertIn(4, pred_cols_D, f"Expected E (col 4) to be predicted after A→D, got {pred_cols_D}")


if __name__ == '__main__':
    unittest.main()

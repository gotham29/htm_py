import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTMBranchingBehavior(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(5,),  # A=0, B=1, C=2, X=3, Y=4
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

    def run_sequence(self, seq, learn=True):
        for col in seq:
            self.tm.compute([col], learn=learn)

    def test_segments_branch_on_different_contexts(self):
        """A→B→C and X→B→C should cause different segments to grow on C."""
        A, B, C, X = 0, 1, 2, 3
        for _ in range(3):
            self.run_sequence([A, B, C])
            self.tm.reset()
        for _ in range(3):
            self.run_sequence([X, B, C])
            self.tm.reset()

        c_cells = list(range(C * self.tm.cellsPerColumn, (C + 1) * self.tm.cellsPerColumn))
        segment_counts = [len(self.tm.connections.segmentsForCell(cell)) for cell in c_cells]
        total_segments = sum(segment_counts)

        self.assertGreaterEqual(total_segments, 2, "Expected multiple segments on C due to A→B→C vs X→B→C")

    def test_predictions_respect_branch_context(self):
        """After A→B→C and X→B→Y, A→B should predict C, not Y."""
        A, B, C, X, Y = 0, 1, 2, 3, 4
        for _ in range(3):
            self.run_sequence([A, B, C])
            self.tm.reset()
        for _ in range(3):
            self.run_sequence([X, B, Y])
            self.tm.reset()

        self.run_sequence([A, B], learn=False)
        predicted_cols = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}

        self.assertIn(C, predicted_cols, "Expected C to be predicted after A→B")
        self.assertNotIn(Y, predicted_cols, "Did not expect Y to be predicted after A→B")

    def test_no_cross_prediction_between_branches(self):
            # Train A→B→C and X→B→C
            for _ in range(3):
                self.run_sequence([0, 1, 2])
                self.tm.reset()
                self.run_sequence([4, 1, 2])
                self.tm.reset()

            # Test A→B→C should not predict Y (col 5)
            self.run_sequence([0, 1, 2], learn=False)
            pred_cols = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}
            self.assertNotIn(5, pred_cols, "Should not predict Y after A→B→C")

    def test_different_winner_cells_for_same_column_in_different_contexts(self):
        # Train A→B and X→B
        for _ in range(3):
            self.run_sequence([0, 1])  # A=0, B=1
            self.tm.reset()
            self.run_sequence([4, 1])  # X=4, B=1
            self.tm.reset()

        # Trigger A→B and capture winner cell for column 1
        self.run_sequence([0], learn=False)
        self.tm.compute([1], learn=False)  # Ensure column 1 is active
        winner_after_A = self.tm.winnerCellForColumn[1]

        # Trigger X→B and capture winner cell for column 1
        self.tm.reset()
        self.run_sequence([4], learn=False)
        self.tm.compute([1], learn=False)
        winner_after_X = self.tm.winnerCellForColumn[1]

        # Expect different winner cells due to context divergence
        self.assertNotEqual(winner_after_A, winner_after_X,
            "Expected different winner cells for column B based on A vs X context")

    def test_segment_reuse_across_identical_contexts(self):
        # Train A→B multiple times to establish strong context memory
        for _ in range(3):
            self.run_sequence([0, 1])  # A=0, B=1
            self.tm.reset()

        # Trigger A→B and get the segment used
        self.run_sequence([0], learn=False)
        self.tm.compute([1], learn=False)
        cell = self.tm.winnerCellForColumn[1]
        segment1 = self.tm.segmentActiveForCell[cell]

        self.tm.reset()

        # Repeat A→B again and confirm same segment is reused
        self.run_sequence([0], learn=False)
        self.tm.compute([1], learn=False)
        cell2 = self.tm.winnerCellForColumn[1]
        segment2 = self.tm.segmentActiveForCell[cell2]

        self.assertEqual(segment1, segment2, "Expected same segment to be reused for same A→B context")

    def test_segment_growth_when_context_changes_mid_sequence(self):
        # Train A→B→C and A→B→D
        for _ in range(3):
            self.run_sequence([0, 1, 2])  # A=0, B=1, C=2
            self.tm.reset()
            self.run_sequence([0, 1, 3])  # A=0, B=1, D=3
            self.tm.reset()

        # After A→B, column 2 (C) and 3 (D) should have distinct segments on different cells
        self.run_sequence([0, 1], learn=False)

        # Trigger prediction for column C
        self.tm.compute([2], learn=False)
        cell_C = self.tm.winnerCellForColumn[2]
        segment_C = self.tm.segmentActiveForCell.get(cell_C)

        self.tm.reset()
        self.run_sequence([0, 1], learn=False)

        # Trigger prediction for column D
        self.tm.compute([3], learn=False)
        cell_D = self.tm.winnerCellForColumn[3]
        segment_D = self.tm.segmentActiveForCell.get(cell_D)

        self.assertNotEqual(segment_C, segment_D,
            "Expected different segments for C and D due to branching after A→B")


if __name__ == "__main__":
    unittest.main()

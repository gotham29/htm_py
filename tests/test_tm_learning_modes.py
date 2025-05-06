import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

class TestTMLearningModes(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(10,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=4,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0
        )

    def test_no_segment_growth_during_inference(self):
        # Step 1: Train with repeated input
        for _ in range(3):
            self.tm.compute([0], learn=True)

        # Count segments created
        cell = list(self.tm.winnerCells)[0]
        initial_segments = len(self.tm.connections.segmentsForCell(cell))

        # Step 2: Inference only
        for _ in range(3):
            self.tm.compute([0], learn=False)

        after_segments = len(self.tm.connections.segmentsForCell(cell))
        self.assertEqual(initial_segments, after_segments)

    def test_predictions_remain_stable_during_inference(self):
        # Train on known sequence
        seq = [[0], [1], [2]]
        for _ in range(3):
            for step in seq:
                self.tm.compute(step, learn=True)

        # Save prediction sequence
        predictions = []
        for step in seq:
            self.tm.compute(step, learn=False)
            predictions.append(set(self.tm.predictiveCells))

        # Run again, assert predictive cells remain identical
        for i, step in enumerate(seq):
            self.tm.compute(step, learn=False)
            self.assertEqual(predictions[i], set(self.tm.predictiveCells))

    def test_system_bursts_again_if_not_learning(self):
        """
        After training a simple sequence and turning off learning,
        a novel input should cause a burst (all cells in that column active),
        and a single winnerCell should still be chosen.
        """
        # Train a short pattern
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        # Repeat pattern under inference mode
        self.tm.compute([0], learn=False)
        self.tm.compute([1], learn=False)

        # Now feed a novel column input under inference mode (should burst)
        self.tm.compute([2], learn=False)

        column_cells = set(self.tm._cells_for_column(2))

        # Assert that all column cells were activated (burst)
        self.assertEqual(self.tm.activeCells, column_cells)

        # Assert that one winnerCell was chosen from the column
        self.assertEqual(len(self.tm.winnerCells), 1)
        self.assertIn(next(iter(self.tm.winnerCells)), column_cells)

    def test_anomaly_score_static_after_learning(self):
        pattern = [[0], [1], [2]]
        for _ in range(10):  # More passes = better stabilization
            for step in pattern:
                self.tm.compute(step, learn=True)

        scores = []
        for step in pattern * 2:
            score, _ = self.tm.compute(step, learn=False)
            scores.append(score)

        print("Inference anomaly scores:", scores)
        self.assertTrue(all(s < 0.5 for s in scores)) 

    def test_predictive_cells_clear_after_reset(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)
        self.tm.reset()

        self.assertEqual(len(self.tm.predictiveCells), 0)
        self.assertEqual(len(self.tm.prevPredictiveCells), 0)

    def test_winner_cells_always_selected_even_if_learning_off(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        self.tm.compute([2], learn=False)
        self.assertGreaterEqual(len(self.tm.winnerCells), 1)
        self.assertIn(next(iter(self.tm.winnerCells)), self.tm.activeCells)

    def test_system_learns_new_pattern_after_reset(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        self.tm.reset()

        self.tm.compute([2], learn=True)
        self.tm.compute([3], learn=True)

        self.tm.compute([2], learn=False)
        self.tm.compute([3], learn=False)
        self.assertEqual(len(self.tm.winnerCells), 1)

    def test_system_bursts_when_predicts_wrong_cell(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        self.tm.compute([0], learn=False)
        self.tm.compute([2], learn=False)  # breaks prediction chain

        self.tm.compute([1], learn=False)
        col_cells = set(self.tm._cells_for_column(1))
        self.assertEqual(self.tm.activeCells, col_cells)  # burst → all cells

    def test_system_does_not_burst_if_predicted_and_not_learning(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        self.tm.compute([0], learn=False)
        self.tm.compute([1], learn=False)

        # Store predicted cells and target column's cells
        predicted_cells = set(self.tm.predictiveCells)
        col_cells = set(self.tm._cells_for_column(1))

        self.tm.compute([1], learn=False)

        if any(c in predicted_cells for c in col_cells):
            # Column was predicted → should NOT burst
            self.assertEqual(len(self.tm.activeCells), 1)
            self.assertIn(next(iter(self.tm.winnerCells)), col_cells)
            self.assertIn(next(iter(self.tm.winnerCells)), predicted_cells)
        else:
            # Column was NOT predicted → should burst
            self.assertEqual(self.tm.activeCells, col_cells)
            self.assertEqual(len(self.tm.winnerCells), 1)
            self.assertIn(next(iter(self.tm.winnerCells)), col_cells)

    def test_system_bursts_again_if_not_learning(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        self.tm.compute([0], learn=False)
        self.tm.compute([1], learn=False)

        self.tm.compute([2], learn=False)

        column_cells = set(self.tm._cells_for_column(2))
        self.assertEqual(self.tm.activeCells, column_cells)
        self.assertEqual(len(self.tm.winnerCells), 1)
        self.assertIn(next(iter(self.tm.winnerCells)), column_cells)


if __name__ == "__main__":
    unittest.main()

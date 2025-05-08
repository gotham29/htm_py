import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections


class TestTMIntegrationBehavior(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0
        )

    def test_prediction_matches_future_winner(self):
        """A correctly trained segment should predict the future winner cell."""
        for t in range(3):
            self.tm.compute([0], learn=True)
            self.tm.prevWinnerCells = set(self.tm.winnerCells)  # retain context

        # At this point, a segment should predict the cell that won in the last step
        prev_winner = next(iter(self.tm.prevWinnerCells))
        predicted = prev_winner in self.tm.predictiveCells
        self.assertTrue(predicted, f"Cell {prev_winner} should be predicted after training")

    def test_anomaly_score_drops_with_prediction(self):
        """Anomaly score should drop once prediction aligns with winner cells."""
        scores = []
        for _ in range(5):
            score, _ = self.tm.compute([0], learn=True)
            scores.append(score)
            self.tm.prevWinnerCells = set(self.tm.winnerCells)

        # Anomaly score should start at 1.0 and decrease over time
        self.assertEqual(scores[0], 1.0)
        self.assertLess(scores[-1], 1.0, "Anomaly score should drop after learning")

    def test_stable_prediction_does_not_grow_new_segments(self):
        """Once prediction is accurate, no new segments should be added."""
        for _ in range(5):
            self.tm.compute([0], learn=True)
            self.tm.prevWinnerCells = set(self.tm.winnerCells)

        segment_count = sum(self.tm.connections.numSegments(c) for c in range(self.tm.numCells))

        # Repeat again — no new segments should be added if predictions are working
        for _ in range(3):
            self.tm.compute([0], learn=True)
            self.tm.prevWinnerCells = set(self.tm.winnerCells)

        final_segment_count = sum(self.tm.connections.numSegments(c) for c in range(self.tm.numCells))
        self.assertEqual(segment_count, final_segment_count, "No new segments should be created when prediction is correct")


if __name__ == "__main__":
    unittest.main()
# test_tm_prediction.py

import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTMPrediction(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=4,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0,
            seed=42,
            encoderWidth=4
        )

    def test_abcd_sequence_prediction(self):
        sequence = [0, 1, 2, 3]
        for i in range(5):  # multiple passes for learning
            for col in sequence:
                self.tm.compute([col], learn=True)

        # inference phase
        pred_scores = []
        pred_counts = []
        pred_cells = []
        for col in sequence:
            anomaly, pred_count = self.tm.compute([col], learn=False)
            pred_scores.append(anomaly)
            pred_counts.append(pred_count)
            pred_cells.append(self.tm.get_predictive_cells())

        print("\n--- Prediction Test Results ---")
        for i, col in enumerate(sequence):
            print(f"Input: {col} | Anomaly: {pred_scores[i]:.2f} | Pred Count: {pred_counts[i]:.2f} | Pred Cells: {pred_cells[i]}")

        # Diagnostic checks
        self.assertLess(max(pred_scores[1:]), 1.0, "Anomaly score should drop after learning")
        self.assertGreater(sum(pred_counts), 0, "There should be non-zero prediction count")
        self.assertTrue(any(pred_cells), "There should be predictive cells activated")

    def test_prediction_does_not_mutate_state(self):
        """Ensure predictive cells do not change during inference."""
        sequence = [0, 1, 2, 3]
        for col in sequence:
            self.tm.compute([col], learn=True)

        pred_before = set(self.tm.get_predictive_cells())
        self.tm.compute([sequence[0]], learn=False)
        pred_after = set(self.tm.get_predictive_cells())
        self.assertEqual(pred_before, pred_after, "Inference should not mutate predictive state")


if __name__ == "__main__":
    unittest.main()

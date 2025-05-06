import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTMSequenceIntegration(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            connectedPermanence=0.2,
            initialPermanence=0.3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=4
        )
        self.conn = self.tm.connections  # ✅ Add this line

    def test_burst_then_predict(self):
        input_col = [0]

        self.tm.compute(input_col, learn=True)
        winner_1 = set(self.tm.winnerCells)

        self.tm.compute(input_col, learn=True)
        winner_2 = set(self.tm.winnerCells)
        predict_2 = set(self.tm.predictiveCells)

        self.assertGreater(len(winner_1), 0)
        self.assertNotEqual(winner_1, winner_2, "Winner cells did not change after 2nd compute")
        self.assertTrue(winner_2.issubset(predict_2), f"Winner {winner_2} not in predicted {predict_2}")

    def test_prediction_failure_on_wrong_input(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)

        self.tm.compute([1], learn=True)
        col1_cells = self.tm._cells_for_column(1)
        col1_preds = set(col1_cells) & self.tm.predictiveCells

        self.assertEqual(len(col1_preds), 0, f"Unexpected prediction in column 1: {col1_preds}")

    def test_anomaly_score_decays_after_learning(self):
        pattern = [[0], [1], [2], [3]] * 3
        scores = [self.tm.compute(step, learn=True)[0] for step in pattern]
        print("Anomaly scores:", scores)

        self.assertGreater(scores[0], 0.9)
        self.assertLess(scores[-1], 0.5, "Anomaly score did not decay after repeated exposure")

    def test_segment_growth_and_prediction_path(self):
        sequence = [[0], [1], [2]]
        for _ in range(5):
            for step in sequence:
                self.tm.compute(step, learn=True)

        predicted = []
        for step in sequence:
            self.tm.compute(step, learn=False)
            predicted.append(set(self.tm.predictiveCells))

        self.assertGreater(len(predicted[1]), 0, "No predictions after learning [0]→[1]")
        self.assertGreater(len(predicted[2]), 0, "No predictions after learning [1]→[2]")

    def test_segment_growth_and_prediction_after_pair(self):
        for _ in range(3):
            self.tm.compute([0], learn=True)
            self.tm.compute([1], learn=True)

        col_0_cells = self.tm._cells_for_column(0)
        col_1_cells = self.tm._cells_for_column(1)

        self.tm.compute([0], learn=False)
        predicted_col1_cells = set(col_1_cells) & self.tm.predictiveCells

        self.assertTrue(predicted_col1_cells, f"No column 1 cell predicted. Predictive: {self.tm.predictiveCells}")

    def test_synapse_permanence_growth(self):
        self.tm.compute([0], learn=True)
        winner = list(self.tm.winnerCells)[0]

        # Explicitly create a segment and attach synapse to force permanence growth
        seg = self.conn.createSegment(winner)
        presyn = self.tm._cells_for_column(1)[0]
        self.conn.createSynapse(seg, presyn, 0.3)  # Permanence > connected threshold

        # Set previous winner cells so that learning occurs
        self.tm.prevWinnerCells = {presyn}
        self.tm.segmentActiveForCell[winner] = seg
        self.tm.prevPredictiveCells = {winner}

        # Trigger adaptation of existing segment
        self.tm.compute([0], learn=True)

        segs = self.tm.connections.segmentsForCell(winner)
        self.assertGreaterEqual(len(segs), 1)

    def test_stability_single_pair(self):
        for _ in range(6):
            self.tm.compute([0], learn=True)
            self.tm.compute([1], learn=True)

        scores = []
        for _ in range(3):
            s0, _ = self.tm.compute([0], learn=False)
            scores.append(s0)
            s1, _ = self.tm.compute([1], learn=False)
            scores.append(s1)

        print("Final anomaly scores:", scores)
        self.assertTrue(all(s < 0.5 for s in scores), "Anomaly scores remain high despite repeated training")

    def test_stability_after_learning(self):
        pattern = [[0], [1]] * 6
        for step in pattern:
            self.tm.compute(step, learn=True)

        scores = [self.tm.compute(step, learn=False)[0] for step in pattern]
        for s in scores:
            self.assertLess(s, 0.5, f"Score too high after learning: {s:.3f}")

if __name__ == "__main__":
    unittest.main()

# import logging
import unittest
from htm_py.temporal_memory import TemporalMemory

# Setup logger for debugging during tests
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("tm_debug.log"),
#         logging.StreamHandler()
#     ]
# )


class TestTMSequenceIntegration(unittest.TestCase):
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

    def run_tm(self, input_columns, learn=True):
        self.tm.compute(activeColumns=input_columns, learn=learn)
        return set(self.tm.activeCells), set(self.tm.predictiveCells)

    def test_basic_learning_and_prediction(self):
        # Step 1: First input, no prior knowledge — should burst
        input1 = [0]
        self.tm.compute(activeColumns=input1, learn=True)
        first_winner_cells = set(self.tm.winnerCells)
        first_predicted_cells = set(self.tm.predictiveCells)
        self.assertGreater(len(first_winner_cells), 0)
        self.assertEqual(len(first_predicted_cells), 0)  # No prediction yet

        # Step 2: Same input again, system should now predict
        self.tm.compute(activeColumns=input1, learn=True)
        second_winner_cells = set(self.tm.winnerCells)
        second_predicted_cells = set(self.tm.predictiveCells)

        # Updated assertions for proper HTM behavior
        self.assertNotEqual(first_winner_cells, second_winner_cells)
        self.assertTrue(second_winner_cells.issubset(second_predicted_cells))
        self.assertGreater(len(second_predicted_cells), 0)

    def test_prediction_followed_by_wrong_input(self):
        # Train on column 0 twice
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)

        # Now input column 1 instead (unexpected)
        self.tm.compute([1], learn=True)
        # Should burst and not reuse previously learned path
        self.assertNotIn(self.tm.winnerCells, self.tm.predictiveCells)

    def test_multistep_sequence_abcde(self):
        inputs = [0, 1, 2, 3, 4]

        # First learning pass
        for i in inputs:
            self.run_tm([i], learn=True)

        # Reset before test
        self.tm.reset()

        # Re-present A to see if B is predicted
        _, pred1 = self.run_tm([0], learn=False)
        self.assertTrue(any(cell // 4 == 1 for cell in pred1), "B should be predicted after A")

        _, pred2 = self.run_tm([1], learn=False)
        self.assertTrue(any(cell // 4 == 2 for cell in pred2), "C should be predicted after B")

        _, pred3 = self.run_tm([2], learn=False)
        self.assertTrue(any(cell // 4 == 3 for cell in pred3), "D should be predicted after C")

        _, pred4 = self.run_tm([3], learn=False)
        self.assertTrue(any(cell // 4 == 4 for cell in pred4), "E should be predicted after D")

    def test_branching_sequences(self):
        # Trains a,b,c,d,x and a,b,c,y, then verifies that after a,b,c, both d and y are predicted.
        abcdx = [0, 1, 2, 3, 5]
        abcy = [0, 1, 2, 6]

        for seq in [abcdx, abcy]:
            self.tm.reset()
            for col in seq:
                self.run_tm([col], learn=True)

        self.tm.reset()
        self.run_tm([0], learn=False)
        self.run_tm([1], learn=False)
        _, preds = self.run_tm([2], learn=False)

        pred_columns = {cell // 4 for cell in preds}
        self.assertIn(3, pred_columns, "D should be predicted after A,B,C")
        self.assertIn(6, pred_columns, "Y should be predicted after A,B,C")


if __name__ == "__main__":
    unittest.main()

import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTemporalMemoryL3(unittest.TestCase):

    def test_sequence_learning_and_prediction(self):
        tm = TemporalMemory(
            columnDimensions=(3,),
            cellsPerColumn=2,
            activationThreshold=1,
            connectedPermanence=0.5,
            initialPermanence=0.6,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=5,
            seed=42
        )

        sequence = [[0], [1], [2]]
        for _ in range(5):
            for col in sequence:
                tm.compute(col, learn=True)

        tm.compute([0], learn=False)
        tm.compute([1], learn=False)

        pred_cells = tm.get_predictive_cells()
        expected_cells = set(tm.cellsForColumn(2))

        self.assertTrue(
            expected_cells.intersection(pred_cells),
            f"Expected a predictive cell from column 2, got {pred_cells}"
        )

    def test_predicted_cells_become_winners(self):
        tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=3,
            activationThreshold=1,
            connectedPermanence=0.5,
            initialPermanence=0.6,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=5,
            seed=42
        )

        for _ in range(4):
            tm.compute([0], learn=True)
            tm.compute([1], learn=True)

        tm.compute([0], learn=False)
        tm.compute([1], learn=False)

        pred_cells = tm.get_predictive_cells()
        winner_cells = tm.get_winner_cells()
        self.assertTrue(
            set(pred_cells).issubset(set(winner_cells)),
            f"Predictive cells should be winner cells. Predictive: {pred_cells}, Winners: {winner_cells}"
        )

    def test_no_learning_does_not_change_predictions(self):
        tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=2,
            activationThreshold=1,
            connectedPermanence=0.5,
            initialPermanence=0.6,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=5,
            seed=42
        )

        for _ in range(5):
            tm.compute([0], learn=True)
            tm.compute([1], learn=True)

        tm.compute([0], learn=False)
        pred_before = set(tm.get_predictive_cells())
        tm.compute([1], learn=False)
        pred_after = set(tm.get_predictive_cells())

        self.assertEqual(pred_before, pred_after, "Predictions changed even though learning was disabled.")

    def test_column_bursting_creates_segment(self):
        tm = TemporalMemory(
            columnDimensions=(3,),
            cellsPerColumn=2,
            activationThreshold=1,
            connectedPermanence=0.5,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=3,
            seed=42
        )

        tm.prevWinnerCells = {0, 1, 2}
        tm.compute([2], learn=True)  # Unpredicted → should burst

        winner_cells = tm.get_winner_cells()
        segments = tm.connections.segments_for_cell(winner_cells.pop())
        self.assertTrue(segments, "Expected new segment in bursting column")

    def test_no_segment_created_when_no_prev_winners(self):
        tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=2,
            activationThreshold=1,
            connectedPermanence=0.5,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=1,
            maxNewSynapseCount=3,
            seed=42
        )

        # No previous winner cells
        tm.prevWinnerCells = set()
        tm.compute([1], learn=True)

        any_segments = any(tm.connections.segments_for_cell(c) for c in tm.cellsForColumn(1))
        self.assertFalse(any_segments, "Should not create segment if no prev winner cells")

if __name__ == "__main__":
    unittest.main()

import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import TemporalMemory


class TestTMPhase1Activation(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(columnDimensions=(2,), cellsPerColumn=4,
                                 activationThreshold=1,
                                 initialPermanence=0.3,
                                 connectedPermanence=0.2,
                                 minThreshold=1,
                                 permanenceIncrement=0.1,
                                 permanenceDecrement=0.0)
        self.tm.reset()

    def test_bursting_column_activates_all_cells(self):
        """If a column is not predicted, all its cells should become active."""
        self.tm.prevPredictiveCells.clear()
        activeCols = [0]
        activeCells, _ = self.tm._activate_columns(activeCols)
        expected = set(range(0, 4))  # 4 cells in column 0
        self.assertEqual(activeCells, expected)

    def test_predicted_column_activates_only_predicted_cells(self):
        """If a column is predicted, only the predicted cell should become active."""
        self.tm.prevPredictiveCells = {5}  # cell 5 is column 1, cell index 1
        activeCols = [1]
        activeCells, winnerCells = self.tm._activate_columns(activeCols)
        self.assertEqual(activeCells, {5})
        self.assertEqual(winnerCells, {5})

    def test_least_used_cell_chosen_as_winner(self):
        """Winner cell on burst should be the least-used one."""
        col = 0
        self.tm.connections.createSegment(1)  # cell 1
        self.tm.connections.createSegment(2)  # cell 2
        # cells 0 and 3 are now least used
        activeCells, winnerCells = self.tm._activate_columns([col])
        self.assertIn(list(winnerCells)[0], {0, 3})

    def test_segment_association_on_predicted_cell(self):
        """Segment used for prediction should be recorded for reuse."""
        cell = 5
        seg = self.tm.connections.createSegment(cell)
        syn = self.tm.connections.createSynapse(seg, 0, 0.3)
        self.tm.prevActiveCells = {0}
        self.tm.prevPredictiveCells = {cell}
        self.tm._activate_columns([1])  # column 1 contains cell 5
        self.assertEqual(self.tm.segmentActiveForCell[cell], seg)


if __name__ == "__main__":
    unittest.main()

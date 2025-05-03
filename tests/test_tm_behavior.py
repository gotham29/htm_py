import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTemporalMemoryBehavior(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            seed=42,
        )

    def test_predicted_cells_activate_correctly(self):
        prev_active_cells = [0, 1, 2, 3]
        self.tm.prevActiveCells = set(prev_active_cells)  # Simulate previous active cells

        col1_cells = self.tm.cells_for_column(1)
        target_cell = col1_cells[0]
        seg = self.tm.connections.create_segment(target_cell)
        for c in prev_active_cells:
            self.tm.connections.create_synapse(seg, c, 0.5)

        self.tm.compute([1], learn=False)
        self.assertIn(target_cell, self.tm.prevActiveCells)

    def test_column_burst_activates_all_cells(self):
        col = 2
        expected = self.tm.cells_for_column(col)
        self.tm.compute([col], learn=False)
        for cell in expected:
            self.assertIn(cell, self.tm.prevActiveCells)

    def test_one_winner_per_burst_column(self):
        col = 3
        self.tm.compute([col], learn=False)  # Priming step to set prevWinnerCells
        self.tm.compute([col], learn=True)   # Real test step
        # self.tm.compute([col], learn=True)
        burst_cells = self.tm.cells_for_column(col)
        winners = [c for c in self.tm.prevWinnerCells if c in burst_cells]
        self.assertEqual(len(winners), 1)

    def test_predicted_cells_are_winners(self):
        self.tm.connected_permanence = 0.5
        self.tm.min_threshold = 2

        prev_active_cells = [0, 1, 2]
        self.tm.prevActiveCells = set(prev_active_cells)

        col = 1
        self.tm.compute([col], learn=False)  # Priming step: sets up winner cell
        winner_cell = list(self.tm.prevWinnerCells)[0]  # Get the true winner selected by HTM

        # Create a segment on the actual HTM-selected winner cell
        seg = self.tm.connections.create_segment(winner_cell)
        for c in prev_active_cells:
            self.tm.connections.create_synapse(seg, c, 0.5)

        # Now run learning step
        self.tm.compute([col], learn=True)

        # Verify that this same cell remains a winner
        self.assertIn(winner_cell, self.tm.prevWinnerCells)

    def test_no_segment_without_winner_cells(self):
        col = 1
        self.tm.compute([], learn=True)
        winner_cells = self.tm.get_winner_cells()
        self.assertEqual(winner_cells, [])


if __name__ == "__main__":
    unittest.main()

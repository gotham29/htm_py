import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTemporalMemoryBehavior(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=2,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            minThreshold=2,
            maxNewSynapseCount=3,
            predictedSegmentDecrement=0.0,
            seed=42
        )

    def test_predicted_cells_activate_correctly(self):
        prev_cells = [0, 1, 2, 3]
        col1_cells = self.tm.cellsForColumn(1)
        cell4 = col1_cells[0]

        seg = self.tm.connections.create_segment(cell4)
        for c in prev_cells:
            self.tm.connections.create_synapse(seg, c, 0.5)

        self.tm.activeCells = prev_cells
        self.tm.compute([1], learn=False)

        self.assertIn(cell4, self.tm.get_active_cells())

    def test_column_burst_activates_all_cells(self):
        col = 2
        expected_cells = self.tm.cellsForColumn(col)
        self.tm.compute([col], learn=False)

        for c in expected_cells:
            self.assertIn(c, self.tm.get_active_cells())

    def test_predicted_cells_are_winners(self):
        prev_cells = [0, 1, 2]
        cell = self.tm.cellsForColumn(1)[0]
        seg = self.tm.connections.create_segment(cell)
        for c in prev_cells:
            self.tm.connections.create_synapse(seg, c, 0.5)

        self.tm.activeCells = prev_cells
        self.tm.compute([1], learn=False)

        self.assertIn(cell, self.tm.get_winner_cells())

    def test_one_winner_per_burst_column(self):
        col = 3
        self.tm.compute([col], learn=False)
        winners = self.tm.get_winner_cells()

        self.assertEqual(len(winners), 1)
        self.assertIn(winners[0], self.tm.cellsForColumn(col))

    def test_no_segment_without_winner_cells(self):
        col = 1
        self.tm.compute([], learn=True)  # Step 1: no input
        self.tm.compute([col], learn=True)  # Step 2: input column with no winners

        for c in self.tm.cellsForColumn(col):
            segs = self.tm.connections.segments_for_cell(c)
            self.assertEqual(len(segs), 0, "No segment should be created if no previous winners")


if __name__ == "__main__":
    unittest.main()

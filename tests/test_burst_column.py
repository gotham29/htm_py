import unittest
from htm_py.temporal_memory import TemporalMemory

class TestBurstColumn(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(columnDimensions=(4,), cellsPerColumn=4, activationThreshold=1,
                                 connectedPermanence=0.2, initialPermanence=0.3,
                                 permanenceIncrement=0.1, permanenceDecrement=0.05,
                                 minThreshold=1, maxNewSynapseCount=4)

    # In test_burst_column.py
    def test_burst_column_creates_segments(self):
        self.tm.prevWinnerCells = set([0])  # fake prior activity
        self.tm.compute([1], learn=True)
        cell_range = self.tm._cells_for_column(1)
        num_segments = sum(len(self.tm.connections.segmentsForCell(c)) for c in cell_range)
        self.assertGreater(num_segments, 0)

    def test_burst_column_marks_winner_cells(self):
        self.tm.activeColumns = {0}
        self.tm.compute(list(self.tm.activeColumns), learn=True)
        self.assertTrue(len(self.tm.winnerCells) > 0)

if __name__ == "__main__":
    unittest.main()

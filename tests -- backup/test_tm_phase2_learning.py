import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections


class TestTMPhase2Learning(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(columnDimensions=(1,), cellsPerColumn=4,
                                 activationThreshold=1,
                                 initialPermanence=0.3,
                                 connectedPermanence=0.2,
                                 minThreshold=1,
                                 permanenceIncrement=0.1,
                                 permanenceDecrement=0.0)
        self.tm.reset()

    def test_adapts_existing_segment_when_context_matches(self):
        cell = 0
        seg = self.tm.connections.createSegment(cell)
        self.tm.connections.createSynapse(seg, 1, 0.3)  # use valid presynaptic cell
        self.tm.prevWinnerCells = {1}
        self.tm.winnerCellForColumn[0] = cell

        self.tm._learn_segments([0], self.tm.prevWinnerCells)

        perma = self.tm.connections.dataForSynapse(next(iter(self.tm.connections.synapsesForSegment(seg)))).permanence
        self.assertGreater(perma, 0.3)

    def test_does_not_adapt_if_context_does_not_match(self):
        cell = 0
        seg = self.tm.connections.createSegment(cell)
        self.tm.connections.createSynapse(seg, 1, 0.3)
        self.tm.prevWinnerCells = {2}  # different presynaptic
        self.tm.winnerCellForColumn[0] = cell

        self.tm._learn_segments([0], self.tm.prevWinnerCells)
        self.assertEqual(self.tm.connections.numSegments(cell), 2)

    def test_creates_new_segment_if_none_exists(self):
        self.tm.winnerCellForColumn[0] = 2
        self.tm.prevWinnerCells = {0, 1}
        self.tm._learn_segments([0], self.tm.prevWinnerCells)
        segs = self.tm.connections.segmentsForCell(2)
        self.assertEqual(len(segs), 1)

    def test_does_not_create_multiple_segments_for_same_context(self):
        self.tm.prevWinnerCells = {3}
        self.tm.winnerCellForColumn[0] = 1
        for _ in range(3):
            self.tm._learn_segments([0], self.tm.prevWinnerCells)
        segs = self.tm.connections.segmentsForCell(1)
        self.assertEqual(len(segs), 1)


if __name__ == "__main__":
    unittest.main()

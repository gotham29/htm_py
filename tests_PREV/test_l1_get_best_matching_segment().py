import unittest
from htm_py.connections import Connections, Segment, Synapse
from htm_py.temporal_memory import TemporalMemory

class TestBestMatchingSegment(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            # connections = Connections(2),
            columnDimensions=(1,), cellsPerColumn=2,
            activationThreshold=1, minThreshold=2,
            maxNewSynapseCount=2, initialPermanence=0.5,
            connectedPermanence=0.2, permanenceIncrement=0.1,
            permanenceDecrement=0.05, seed=42
        )
        self.con = self.tm.connections

        self.cellA = 0  # column 0, cell 0
        self.cellB = 1  # column 0, cell 1
        self.col_cells = [self.cellA, self.cellB]

        # Create 2 segments with different overlaps
        self.segment1 = self.con.create_segment(self.cellA, [10, 20], 0.5)
        self.segment2 = self.con.create_segment(self.cellB, [20, 30, 40], 0.5)  # higher overlap

        # Adjust permanence to ensure synapses are connected
        for syn in self.segment1.synapses:
            syn.permanence = 0.3
        for syn in self.segment2.synapses:
            syn.permanence = 0.3

    def test_best_matching_segment(self):
        # Only 2 presyn cells active; segment2 should win if minThreshold=2
        prevActiveCells = {20, 30}

        seg, overlap = self.tm._get_best_matching_segment(self.col_cells, prevActiveCells)

        self.assertEqual(seg, self.segment2)
        self.assertEqual(overlap, 2)

    def test_below_threshold(self):
        # Only 1 active presyn cell; below minThreshold=2
        prevActiveCells = {10}

        seg, overlap = self.tm._get_best_matching_segment(self.col_cells, prevActiveCells)

        self.assertIsNone(seg)
        self.assertEqual(overlap, 0)

if __name__ == '__main__':
    unittest.main()

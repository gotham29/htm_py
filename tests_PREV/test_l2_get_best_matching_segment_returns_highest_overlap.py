import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTemporalMemoryL2(unittest.TestCase):
    def test_get_best_matching_segment_returns_highest_overlap(self):
        tm = TemporalMemory(columnDimensions=(4,), 
                            cellsPerColumn=3,
                            activationThreshold=1,
                            initialPermanence=0.21,
                            connectedPermanence=0.2,
                            permanenceIncrement=0.1,
                            permanenceDecrement=0.1,
                            minThreshold=1,
                            maxNewSynapseCount=1,
                            seed=1)
        tm.prevActiveCells = {99, 98, 97}

        cell = 0
        seg1 = tm.connections.create_segment(cell, presynapticCells=[99, 98], initialPermanence=0.3)
        seg2 = tm.connections.create_segment(cell, presynapticCells=[99], initialPermanence=0.3)
        seg3 = tm.connections.create_segment(cell, presynapticCells=[88], initialPermanence=0.3)

        col_cells = [0,1,2]
        best, overlap = tm._get_best_matching_segment(col_cells, tm.prevActiveCells)

        assert best is not None
        assert best == seg1

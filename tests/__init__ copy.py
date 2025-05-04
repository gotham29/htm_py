import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTMSegments(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=2,  # deliberately >1
            maxNewSynapseCount=3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )

    def test_get_best_matching_segment_filters_by_min_threshold(self):
        col = 0
        cell = self.tm._cells_for_column(col)[0]
        seg = self.tm.connections.create_segment(cell)

        # Below threshold: 1 active connected synapse
        self.tm.connections.create_synapses(seg, [0], 0.6)
        self.tm.prevActiveCells = {0}

        # minThreshold is 2, so this segment should NOT be returned
        c, s = self.tm._get_best_matching_segment(col)
        self.assertIsNone(c)
        self.assertIsNone(s)

        # Add another synapse to meet the minThreshold
        self.tm.connections.create_synapses(seg, [1], 0.6)
        self.tm.prevActiveCells = {0, 1}

        c, s = self.tm._get_best_matching_segment(col)
        self.assertEqual(c, cell)
        self.assertEqual(s, seg)

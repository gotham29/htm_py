import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections


class TestPhase1(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.6,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )

    def test_cells_for_column(self):
        self.assertEqual(self.tm._cells_for_column(1), [2, 3])

    def test_get_least_used_cell(self):
        cell = self.tm._get_least_used_cell(1)
        self.assertIn(cell, [2, 3])

    def test_num_active_connected_synapses(self):
        cell = self.tm._get_least_used_cell(0)
        segment = self.tm.connections.create_segment(cell)
        
        # Create one connected and one disconnected synapse
        self.tm.connections.create_synapses(segment, [0], permanence=0.6)  # connected
        self.tm.connections.create_synapses(segment, [1], permanence=0.3)  # not connected
        
        # Only cell 0 should be counted as active connected synapse
        self.assertEqual(self.tm._num_active_connected_synapses(segment, {0, 1}), 1)

    def test_get_best_matching_segment(self):
        cell, seg = self.tm._get_best_matching_segment(0)
        self.assertIn(cell, [0, 1])
        self.assertTrue(seg is None or isinstance(seg, int))
# File: tests/test_connections_synapses.py
import unittest
from htm_py.connections import Connections

class TestConnectionsSynapseActivity(unittest.TestCase):
    def setUp(self):
        self.conn = Connections(4)
        self.seg = self.conn.create_segment(0)
        self.syn1 = self.conn.create_synapse(self.seg, 1, 0.6)  # connected
        self.syn2 = self.conn.create_synapse(self.seg, 2, 0.2)  # not connected

    def test_connected_only_true(self):
        active = self.conn.active_synapses(self.seg, active_cells={1, 2}, connected_only=True)  #prev_active
        self.assertIn(self.syn1, active)
        self.assertNotIn(self.syn2, active)

    def test_connected_only_false(self):
        active = self.conn.active_synapses(self.seg, active_cells={1, 2}, connected_only=False)  #prev_active
        self.assertIn(self.syn1, active)
        self.assertIn(self.syn2, active)

    def test_no_active_sources(self):
        active = self.conn.active_synapses(self.seg, active_cells=set(), connected_only=True)  #prev_active
        self.assertEqual(len(active), 0)


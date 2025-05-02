import unittest
from htm_py.connections import Connections

class TestConnections(unittest.TestCase):

    def setUp(self):
        self.connections = Connections()

    def test_create_segment_adds_synapses(self):
        cell = 0
        presynaptic_cells = {100, 101, 102}
        initial_perm = 0.25

        segment = self.connections.create_segment(cell, presynapticCells=presynaptic_cells, initialPermanence=initial_perm)

        self.assertIsNotNone(segment)
        self.assertEqual(segment.cell, cell)
        self.assertEqual(len(segment.synapses), len(presynaptic_cells))

        connected = [syn for syn in segment.synapses if syn.presynapticCell in presynaptic_cells]
        self.assertEqual(len(connected), len(presynaptic_cells))
        for syn in connected:
            self.assertAlmostEqual(syn.permanence, initial_perm, places=5)
        # Clear internal state just to be sure
        self.connections.clear()

    def test_create_segment_grows_synapses_to_prevwinners(self):
        # Setup
        cell = 42
        prev_winner_cells = {1, 2, 3}
        initial_perm = 0.21

        # Act
        segment = self.connections.create_segment(cell, presynapticCells=prev_winner_cells, initialPermanence=initial_perm)

        # Assert
        self.assertEqual(segment.cell, cell)
        self.assertEqual(len(segment.synapses), len(prev_winner_cells))
        for syn in segment.synapses:
            self.assertIn(syn.presynapticCell, prev_winner_cells)
            self.assertAlmostEqual(syn.permanence, initial_perm, places=5)
        # Clear internal state just to be sure
        self.connections.clear()
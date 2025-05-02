import unittest
from htm_py.connections import Connections, Synapse, Segment

class TestAdaptSegment(unittest.TestCase):
    def setUp(self):
        self.connections = Connections()
        self.cell = 0
        self.segment = self.connections.create_segment(self.cell)

        # Create synapses from presynaptic cells [10, 20, 30]
        self.presynaptic_cells = {10, 20, 30}
        for presyn in self.presynaptic_cells:
            self.connections.create_synapse(self.segment, presyn, 0.21)

        self.prev_winner_cells = {10, 40}  # 10 should be reinforced, 20/30 punished
        self.perm_inc = 0.1
        self.perm_dec = 0.05

        # Clear internal state just to be sure
        self.connections.clear()

    def test_positive_reinforcement(self):
        self.connections.adapt_segment(
            self.segment,
            prevWinnerCells=self.prev_winner_cells,
            positive_reinforcement=True,
            perm_inc=self.perm_inc,
            perm_dec=self.perm_dec,
        )

        # Check that synapse from cell 10 increased
        syn10 = next(s for s in self.segment.synapses if s.presynapticCell == 10)
        self.assertAlmostEqual(syn10.permanence, 0.31)

        # Check that synapses from cells 20 and 30 decreased
        for cell in [20, 30]:
            syn = next(s for s in self.segment.synapses if s.presynapticCell == cell)
            self.assertAlmostEqual(syn.permanence, 0.16)

    def test_negative_reinforcement(self):
        self.connections.adapt_segment(
            self.segment,
            prevWinnerCells=self.prev_winner_cells,
            positive_reinforcement=False,
            perm_inc=self.perm_inc,
            perm_dec=self.perm_dec,
        )

        # All synapses should be punished if they were in prevWinnerCells
        for syn in self.segment.synapses:
            if syn.presynapticCell in self.prev_winner_cells:
                expected = max(0.0, 0.21 - self.perm_dec)
                self.assertAlmostEqual(syn.permanence, expected)

if __name__ == "__main__":
    unittest.main()

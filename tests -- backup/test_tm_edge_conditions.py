import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTMEdgeConditions(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=2,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            minThreshold=2,
            maxNewSynapseCount=3
        )
        self.conn = self.tm.connections

    def test_segment_creation_on_empty_cell_set(self):
        self.tm.compute([], learn=True)
        for cell in range(self.tm.numCells):
            self.assertEqual(len(self.conn.segmentsForCell(cell)), 0)

    def test_no_synapses_above_threshold(self):
        cell = self.tm._cells_for_column(0)[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, 1, 0.19)  # below threshold
        self.conn.createSynapse(seg, 2, 0.15)
        self.tm.activeCells = {1, 2}
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_tm_resets_all_state(self):
        self.tm.activeColumns = [1]
        self.tm.compute([1], learn=True)
        self.assertGreater(len(self.tm.activeCells), 0)
        self.tm.reset()
        self.assertEqual(len(self.tm.activeCells), 0)
        self.assertEqual(len(self.tm.predictiveCells), 0)
        self.assertEqual(len(self.tm.winnerCells), 0)

    def test_multiple_segments_same_cell_priority(self):
        self.tm.activationThreshold = 1  # ✅ ensure threshold allows prediction from single matching synapse

        cell = self.tm._cells_for_column(0)[0]
        seg1 = self.conn.createSegment(cell)
        seg2 = self.conn.createSegment(cell)
        self.conn.createSynapse(seg1, 3, 0.3)
        self.conn.createSynapse(seg2, 4, 0.3)

        self.tm.activeCells = {3, 4}
        self.tm.prevActiveCells = self.tm.activeCells.copy()
        self.tm._predict_cells()

        print("[TEST] Predictive cells:", sorted(self.tm.predictiveCells))
        self.assertIn(cell, self.tm.predictiveCells)

    def test_no_prediction_if_segment_below_min_threshold(self):
        cell = self.tm._cells_for_column(2)[1]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, 5, 0.3)
        # Only one active cell, threshold is 2
        self.tm.activeCells = {5}
        self.tm.prevActiveCells = self.tm.activeCells.copy()
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)


if __name__ == "__main__":
    unittest.main()

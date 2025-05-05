import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import TemporalMemory


class TestPredictCells(unittest.TestCase):
    def setUp(self):
        self.columns = 4
        self.cells_per_column = 4
        self.num_cells = self.columns * self.cells_per_column
        self.tm = TemporalMemory(
            columnDimensions=(self.columns,),
            cellsPerColumn=self.cells_per_column,
            activationThreshold=2,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=4,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0
        )
        self.conn = self.tm.connections
        self.presynaptic = list(range(8))  # Define 8 cells to use for presynaptic synapses

    def test_predictive_cell_triggered(self):
        cell = self.presynaptic[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.21)
        self.conn.createSynapse(seg, self.presynaptic[2], 0.22)

        self.tm.activeCells = {self.presynaptic[1], self.presynaptic[2]}
        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells)

    def test_predictive_cell_not_triggered_below_threshold(self):
        cell = self.tm._cells_for_column(0)[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.21)  # only one active synapse

        self.tm.activeCells = {self.presynaptic[1]}
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_predictive_cell_not_triggered_if_synapse_inactive(self):
        cell = self.tm._cells_for_column(0)[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.15)  # permanence < connected threshold

        self.tm.activeCells = {self.presynaptic[1]}
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_no_predictive_cells_when_none_active(self):
        cell = self.tm._cells_for_column(0)[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.25)
        self.conn.createSynapse(seg, self.presynaptic[2], 0.25)

        self.tm.activeCells = set()  # no active cells
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_multiple_predictive_cells(self):
        cells = self.tm._cells_for_column(0)
        seg1 = self.conn.createSegment(cells[0])
        seg2 = self.conn.createSegment(cells[1])

        self.conn.createSynapse(seg1, cells[2], 0.21)
        self.conn.createSynapse(seg1, cells[3], 0.21)
        self.conn.createSynapse(seg2, cells[2], 0.22)
        self.conn.createSynapse(seg2, cells[3], 0.23)

        self.tm.activeCells = {cells[2], cells[3]}
        self.tm._predict_cells()

        self.assertIn(cells[0], self.tm.predictiveCells)
        self.assertIn(cells[1], self.tm.predictiveCells)

    def test_no_predictive_cells_if_all_synapses_below_permanence(self):
        cell = self.tm._cells_for_column(1)[0]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.1)
        self.conn.createSynapse(seg, self.presynaptic[2], 0.19)

        self.tm.activeCells = {self.presynaptic[1], self.presynaptic[2]}
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_predictive_cell_requires_exact_threshold_match(self):
        cell = self.tm._cells_for_column(1)[1]
        seg = self.conn.createSegment(cell)

        for i in range(self.tm.activationThreshold):
            self.conn.createSynapse(seg, self.presynaptic[i], 0.3)

        self.tm.activeCells = set(self.presynaptic[:self.tm.activationThreshold])
        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells)

    def test_prediction_with_mixed_active_and_inactive_synapses(self):
        cell = self.tm._cells_for_column(2)[0]
        seg = self.conn.createSegment(cell)

        # 2 active synapses (permanence > threshold), 1 below threshold
        self.conn.createSynapse(seg, self.presynaptic[0], 0.1)  # inactive
        self.conn.createSynapse(seg, self.presynaptic[1], 0.25)
        self.conn.createSynapse(seg, self.presynaptic[2], 0.25)

        self.tm.activeCells = {self.presynaptic[0], self.presynaptic[1], self.presynaptic[2]}
        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells)

    def test_predictive_cells_clear_on_reset(self):
        cell = self.tm._cells_for_column(3)[1]
        seg = self.conn.createSegment(cell)
        self.conn.createSynapse(seg, self.presynaptic[1], 0.3)
        self.conn.createSynapse(seg, self.presynaptic[2], 0.3)

        self.tm.activeCells = {self.presynaptic[1], self.presynaptic[2]}
        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells)

        # Clear activity and rerun
        self.tm.activeCells = set()
        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_predictive_cell_with_multiple_segments(self):
        cell = self.tm._cells_for_column(3)[2]
        seg1 = self.conn.createSegment(cell)
        seg2 = self.conn.createSegment(cell)

        self.conn.createSynapse(seg1, self.presynaptic[1], 0.25)
        self.conn.createSynapse(seg1, self.presynaptic[2], 0.25)
        self.conn.createSynapse(seg2, self.presynaptic[3], 0.25)
        self.conn.createSynapse(seg2, self.presynaptic[4], 0.25)

        self.tm.activeCells = {self.presynaptic[1], self.presynaptic[2], self.presynaptic[3]}
        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells)


if __name__ == "__main__":
    unittest.main()

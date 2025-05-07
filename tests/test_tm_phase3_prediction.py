import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections


class TestTMPhase3Prediction(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(columnDimensions=(2,), cellsPerColumn=16,
                                 activationThreshold=1, initialPermanence=0.3,
                                 connectedPermanence=0.2, minThreshold=1,
                                 permanenceIncrement=0.1, permanenceDecrement=0.0)
        self.tm.reset()

    def test_predicts_cells_with_connected_synapses_from_prev_active(self):
        cell = 0
        seg = self.tm.connections.createSegment(cell)
        for src in [4, 5]:  # Assume these were active last time
            self.tm.connections.createSynapse(seg, src, 0.3)

        self.tm.prevActiveCells = {4, 5}
        self.tm._predict_cells()

        self.assertIn(cell, self.tm.predictiveCells)
        self.assertEqual(self.tm.predictedSegmentForCell[cell], seg)

    def test_does_not_predict_if_overlap_below_threshold(self):
        cell = 0
        seg = self.tm.connections.createSegment(cell)
        self.tm.connections.createSynapse(seg, 7, 0.3)
        self.tm.connections.createSynapse(seg, 8, 0.1)  # below connectedPermanence

        self.tm.prevActiveCells = {7, 8}
        self.tm.activationThreshold = 2  # require 2 connected
        self.tm._predict_cells()

        self.assertNotIn(cell, self.tm.predictiveCells)

    def test_multiple_cells_get_predicted(self):
        seg1 = self.tm.connections.createSegment(0)
        seg2 = self.tm.connections.createSegment(1)

        self.tm.connections.createSynapse(seg1, 3, 0.3)
        self.tm.connections.createSynapse(seg2, 3, 0.3)

        self.tm.prevActiveCells = {3}
        self.tm._predict_cells()

        self.assertIn(0, self.tm.predictiveCells)
        self.assertIn(1, self.tm.predictiveCells)

    def test_prediction_fails_if_no_segments_exist(self):
        self.tm.prevActiveCells = {2, 3}
        self.tm._predict_cells()

        self.assertEqual(len(self.tm.predictiveCells), 0)

    def test_predictions_use_connected_synapses_only(self):
        seg = self.tm.connections.createSegment(0)
        self.tm.connections.createSynapse(seg, 2, 0.3)
        self.tm.connections.createSynapse(seg, 3, 0.1)  # below connected

        self.tm.prevActiveCells = {2, 3}
        self.tm.activationThreshold = 2  # need both connected
        self.tm._predict_cells()

        self.assertNotIn(0, self.tm.predictiveCells)


if __name__ == "__main__":
    unittest.main()

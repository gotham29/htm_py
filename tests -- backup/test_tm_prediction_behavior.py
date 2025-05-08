import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTMPredictionBehavior(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=2,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

    def test_predictive_cell_activated_when_threshold_met(self):
        # Setup: create segment with 2 synapses from prevActiveCells
        self.tm.prevActiveCells = {1, 2}
        cell = 0
        seg = self.tm.connections.createSegment(cell, sequence=True)
        self.tm.connections.createSynapse(seg, 1, 0.25)
        self.tm.connections.createSynapse(seg, 2, 0.25)

        self.tm._predict_cells()
        self.assertIn(cell, self.tm.predictiveCells, "Cell should be predictive when threshold is met.")

    def test_predictive_cell_not_activated_when_threshold_not_met(self):
        self.tm.prevActiveCells = {1}
        cell = 0
        seg = self.tm.connections.createSegment(cell, sequence=True)
        self.tm.connections.createSynapse(seg, 1, 0.25)

        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells, "Cell should not be predictive when threshold not met.")

    def test_non_sequence_segment_does_not_predict(self):
        self.tm.prevActiveCells = {1, 2}
        cell = 0
        seg = self.tm.connections.createSegment(cell, sequence=False)
        self.tm.connections.createSynapse(seg, 1, 0.25)
        self.tm.connections.createSynapse(seg, 2, 0.25)

        self.tm._predict_cells()
        self.assertNotIn(cell, self.tm.predictiveCells, "Non-sequence segments should not drive predictions.")


if __name__ == "__main__":
    unittest.main()
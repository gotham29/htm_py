import unittest
from htm_py.temporal_memory import TemporalMemory


class TestPhase3(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )

    def test_prediction_updates(self):
        self.tm.prevPredictiveCells = {0}
        self.tm._predict_cells()
        self.assertIsInstance(self.tm.predictiveCells, set)
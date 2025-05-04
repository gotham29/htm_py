import unittest
from htm_py.temporal_memory import TemporalMemory


class TestPhase2(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=3,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )

    def test_learn_segments_creates_segments(self):
        activeCols = [0]
        self.tm.prevActiveCells = {0, 1}
        self.tm.compute(activeCols, learn=True)
        self.assertGreater(len(self.tm.connections.data), 0)
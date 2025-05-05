import unittest
from htm_py.temporal_memory import TemporalMemory


class TestSimpleSequence(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.6,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=4,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )

    def test_simple_repeating_sequence(self):
        sequence = [0, 1, 2, 3, 1, 2]
        scores = []
        for _ in range(5):
            for col in sequence:
                score, pred = self.tm.compute([col], learn=True)
                scores.append(score)
        self.assertLess(scores[-1], 0.1)
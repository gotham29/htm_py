import unittest
from htm_py.temporal_memory import TemporalMemory


def setup_tm():
    return TemporalMemory(
        columnDimensions=(4,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.41,
        connectedPermanence=0.5,
        minThreshold=1,
        maxNewSynapseCount=2,
        maxSynapsesPerSegment=32,
        maxSegmentsPerCell=5,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0
    )


class TestTMSequence(unittest.TestCase):
    def test_simple_sequence_learning(self):
        tm = setup_tm()
        for val in [0, 1, 2, 3]:
            tm.compute([val], learn=True)
        tm.compute([0], learn=False)
        self.assertIn(1, tm.predictiveCells)

    def test_multiple_transitions(self):
        tm = setup_tm()
        for val in [0, 1, 2, 3, 0, 1, 2]:
            tm.compute([val], learn=True)
        self.assertGreaterEqual(len(tm.connections.segments), 1)

    def test_repeated_inputs_no_extra_segments(self):
        tm = setup_tm()
        for _ in range(10):
            tm.compute([0], learn=True)
        initial_segments = len(tm.connections.segments)
        for _ in range(5):
            tm.compute([0], learn=True)
        self.assertEqual(initial_segments, len(tm.connections.segments))

    def test_prediction_count_reflects_expected_number(self):
        tm = setup_tm()
        for val in [0, 1, 2]:
            tm.compute([val], learn=True)
        tm.compute([0], learn=False)
        self.assertGreater(tm.predictionCount, 0)

    def test_learning_and_inference_consistency(self):
        tm = setup_tm()
        sequence = [0, 1, 2, 3]
        for val in sequence:
            tm.compute([val], learn=True)
        tm.compute([0], learn=False)
        self.assertIn(1, tm.predictiveCells)

    def test_anomaly_score_increases_on_unexpected_input(self):
        tm = setup_tm()
        for val in [0, 1, 2, 3]:
            tm.compute([val], learn=True)
        tm.compute([0], learn=False)
        expected_anomaly = tm.anomalyScore
        tm.compute([99], learn=False)
        self.assertGreater(tm.anomalyScore, expected_anomaly)

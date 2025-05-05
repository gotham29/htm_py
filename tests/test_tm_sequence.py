import unittest
from htm_py.temporal_memory import TemporalMemory

def setup_simple_tm():
    return TemporalMemory(
        columnDimensions=(10,),       # only 10 columns
        cellsPerColumn=4,
        activationThreshold=1,        # lowered due to sparse input
        initialPermanence=0.41,
        connectedPermanence=0.5,
        minThreshold=1,               # match a single active synapse
        maxNewSynapseCount=4,
        maxSynapsesPerSegment=32,
        maxSegmentsPerCell=5,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        seed=42
    )

class TestTMSequenceLearning(unittest.TestCase):

    def test_sequence_prediction(self):
        tm = setup_simple_tm()
        sequence = [1, 2, 3, 4, 5]

        # Train with repeated sequence
        for _ in range(5):
            for val in sequence:
                tm.compute([val], learn=True)

        # Run again to see if it's learned
        anomaly_scores = []
        for val in sequence:
            score, _ = tm.compute([val], learn=False)
            anomaly_scores.append(score)

        print("[TEST] Anomaly Scores:", anomaly_scores)

        # Assert that anomaly drops (i.e., the system has learned the pattern)
        self.assertLess(anomaly_scores[-1], 1.0, "Expected lower anomaly on known sequence")


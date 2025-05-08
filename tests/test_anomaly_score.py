import unittest
from htm_py.temporal_memory import TemporalMemory


def setup_tm_for_scoring():
    tm = TemporalMemory(
        columnDimensions=(2,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.3,
        connectedPermanence=0.2,
        minThreshold=1,
        maxNewSynapseCount=4,
        permanenceIncrement=0.05,
        permanenceDecrement=0.0,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=5,
        seed=42,
    )
    return tm


class TestAnomalyScoring(unittest.TestCase):

    def test_anomaly_score_is_1_when_no_predictions_match(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1}
        tm.predictiveCells = {2, 3}

        score = tm._calculate_anomaly_score()
        self.assertEqual(score, 1.0)

    def test_anomaly_score_is_0_when_all_predictions_match(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1}
        tm.predictiveCells = {0, 1, 2}

        score = tm._calculate_anomaly_score()
        self.assertEqual(score, 0.0)

    def test_prediction_count_matches_overlap(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1, 2}
        tm.predictiveCells = {1, 2, 3}

        count = tm._calculate_prediction_count()
        self.assertEqual(count, 2)

    def test_prediction_count_zero_when_no_predictions(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1}
        tm.predictiveCells = set()

        count = tm._calculate_prediction_count()
        self.assertEqual(count, 0)

    def test_normalized_prediction_count_matches_column_ratio(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1, 2, 3}  # one column
        tm.predictiveCells = {0, 1}    # 2 predicted

        pred_count = tm._calculate_prediction_count()
        normalized = pred_count / (len(tm.activeCells) / tm.cellsPerColumn)
        self.assertAlmostEqual(normalized, 2.0)

    def test_get_normalized_prediction_count_matches_internal_calc(self):
        tm = setup_tm_for_scoring()
        tm.activeCells = {0, 1, 2, 3}  # full column
        tm.predictiveCells = {0, 1}

        expected = tm._calculate_prediction_count() / (len(tm.activeCells) / tm.cellsPerColumn)
        self.assertAlmostEqual(tm.getNormalizedPredictionCount(), expected)


if __name__ == "__main__":
    unittest.main()

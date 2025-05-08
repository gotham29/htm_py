import unittest
from htm_py.temporal_memory import TemporalMemory


def setup_tm_prediction():
    return TemporalMemory(
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


class TestPhase3Prediction(unittest.TestCase):

    def test_predictive_cell_fires_after_learning_sequence(self):
        tm = setup_tm_prediction()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevActiveCells = {0}
        tm.compute([0], learn=False)

        preds = tm.getPredictiveCells()
        self.assertIn(0, preds, "Expected cell 0 to be predictive after learning")

    def test_predictive_cell_does_not_fire_if_context_missing(self):
        tm = setup_tm_prediction()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevActiveCells = {1}  # wrong context
        tm.compute([0], learn=False)

        preds = tm.getPredictiveCells()
        self.assertNotIn(0, preds, "Prediction should not happen with wrong context")

    def test_predictive_cells_cleared_each_timestep(self):
        tm = setup_tm_prediction()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevActiveCells = {0}
        tm.compute([0], learn=False)
        preds_1 = set(tm.getPredictiveCells())

        tm.prevActiveCells = set()
        tm.compute([1], learn=False)
        preds_2 = set(tm.getPredictiveCells())

        self.assertTrue(preds_1)
        self.assertFalse(preds_2, "Predictive cells should be cleared each timestep")

    def test_prediction_threshold_enforced(self):
        tm = setup_tm_prediction()
        tm.activationThreshold = 2  # set high threshold

        tm.prevWinnerCells = {0}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)  # only 1 synapse

        tm.prevActiveCells = {0}
        tm._predict_next()
        self.assertNotIn(0, tm.getPredictiveCells(), "Threshold not met — no prediction")

    def test_predictive_state_is_column_specific(self):
        tm = setup_tm_prediction()

        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevActiveCells = {0}
        tm.compute([0], learn=False)
        preds = tm.getPredictiveCells()

        for cell in preds:
            col = cell // tm.cellsPerColumn
            self.assertEqual(col, 0, "Predictions should come from correct column")

    def test_predictive_state_matches_active_segment(self):
        tm = setup_tm_prediction()
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(3)
        tm.connections.createSynapse(seg, 0, 0.3)
        tm.connections.createSynapse(seg, 1, 0.3)

        tm.prevActiveCells = {0, 1}
        tm._predict_next()

        self.assertIn(3, tm.getPredictiveCells())

    def test_learning_and_prediction_do_not_mix_in_inference(self):
        tm = setup_tm_prediction()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        # Inference step
        tm.prevActiveCells = {0}
        tm.compute([0], learn=False)

        # Ensure nothing learned — synapse count should not grow
        preds = tm.getPredictiveCells()
        segs = tm.connections.segmentsForCell(0)
        syns = tm.connections.synapsesForSegment(segs[0])
        self.assertIn(0, preds)
        self.assertEqual(len(syns), 1, "Inference step should not grow synapses")

    def test_predictive_state_matches_active_segment(self):
        tm = setup_tm_prediction()
        tm.iteration = 0
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(3)
        tm.connections.createSynapse(seg, 0, 0.3)
        tm.connections.createSynapse(seg, 1, 0.3)

        tm.prevActiveCells = {0, 1}
        tm._predict_next()

        self.assertIn(3, tm.getPredictiveCells())


if __name__ == "__main__":
    unittest.main()

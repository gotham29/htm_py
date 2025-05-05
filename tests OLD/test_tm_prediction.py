import unittest
from htm_py.temporal_memory import TemporalMemory


def setup_tm(activation_threshold=1, connected_perm=0.2):
    return TemporalMemory(
        columnDimensions=(4,),
        cellsPerColumn=2,
        activationThreshold=activation_threshold,
        initialPermanence=0.21,
        connectedPermanence=connected_perm,
        minThreshold=1,
        maxNewSynapseCount=4,
        maxSynapsesPerSegment=32,
        maxSegmentsPerCell=5,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.01,
        seed=42
    )


class TestTMPrediction(unittest.TestCase):

    def test_predictive_cell_activation(self):
        tm = setup_tm()  # Setup with low thresholds
        for _ in range(5):
            tm.compute([0], learn=True)  # Trains a sequence with column 0

        tm.compute([0], learn=False)  # No learning, just prediction
        pred = tm.predictiveCells
        print("Predictive cells:", pred)
        self.assertGreater(len(pred), 0, "At least one cell should become predictive")

    def test_tm_predicts_when_active_synapse_matches(self):
        tm = setup_tm()
        cell = 0
        source_cell = 1
        tm.prevActiveCells = {source_cell}
        segment = tm.connections.createSegment(cell, iteration_num=0)
        tm.connections.create_synapses(
            segment=segment,
            presynaptic_cells={source_cell},
            initial_permanence=0.25,
            max_new_synapses=1
        )
        tm.activeCells = {source_cell}
        tm._predictCells()
        assert cell in tm.predictiveCells, f"Expected cell {cell} to be predictive"

    def test_tm_does_not_predict_if_synapse_below_threshold(self):
        tm = setup_tm(connected_perm=0.3)
        cell = 0
        source_cell = 1
        tm.prevActiveCells = {source_cell}
        segment = tm.connections.createSegment(cell, iteration_num=0)
        tm.connections.create_synapses(
            segment=segment,
            presynaptic_cells={source_cell},
            initial_permanence=0.29,
            max_new_synapses=1
        )
        tm.activeCells = {source_cell}
        tm._predictCells()
        assert cell not in tm.predictiveCells, "Prediction should not happen below threshold"

    def test_tm_predicts_only_above_activation_threshold(self):
        tm = setup_tm(activation_threshold=2)
        cell = 0
        sources = {1, 2}
        tm.prevActiveCells = sources
        segment = tm.connections.createSegment(cell, iteration_num=0)
        tm.connections.create_synapses(
            segment=segment,
            presynaptic_cells=sources,
            initial_permanence=0.25,
            max_new_synapses=2
        )
        tm.activeCells = {1}
        tm._predictCells()
        assert cell not in tm.predictiveCells, "Prediction requires >= activationThreshold active sources"
        tm.activeCells = sources
        tm._predictCells()
        assert cell in tm.predictiveCells, "Cell should now be predictive with enough active sources"

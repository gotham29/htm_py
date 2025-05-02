import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTemporalMemoryL2(unittest.TestCase):

    def test_predicted_column_activation(self):
        tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.5,
            connectedPermanence=0.5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            minThreshold=1,
            maxNewSynapseCount=0,
            seed=42
        )

        tm.prevPredictiveCells = {3}
        activeColumns = [1]  # Column 1 includes cells 2 and 3
        tm.compute(activeColumns, learn=False)

        assert 3 in tm.activeCells, "Predicted cell should be active"
        assert 3 in tm.winnerCells, "Predicted cell should be winner"

    def test_column_bursting_behavior(self):
        tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.5,
            connectedPermanence=0.5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            minThreshold=1,
            maxNewSynapseCount=0,
            seed=42
        )

        tm.prevPredictiveCells = set()  # No prediction
        activeColumns = [2]  # Cells 4 and 5

        tm.compute(activeColumns, learn=False)

        assert 4 in tm.activeCells and 5 in tm.activeCells, "All cells should burst"
        assert len(tm.winnerCells) == 1, "Exactly one winner cell should be chosen"

    def test_learn_creates_segment(self):
        tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            minThreshold=1,
            maxNewSynapseCount=5,
            seed=42
        )

        tm.prevWinnerCells = {0, 1}
        activeColumns = [2]

        tm.compute(activeColumns, learn=True)

        segments = tm.connections.segments_for_cell(next(iter(tm.winnerCells)))
        assert len(segments) > 0, "New segment should be created"
        synapses = segments[0].synapses
        assert len(synapses) > 0, "Synapses should be added to new segment"

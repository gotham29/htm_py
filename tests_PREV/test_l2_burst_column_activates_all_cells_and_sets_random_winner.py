import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

class TestTemporalMemoryL2(unittest.TestCase):
    def test_burst_column_activates_all_cells_and_sets_random_winner(self):
        tm = TemporalMemory(columnDimensions=(4,), 
                            cellsPerColumn=3,
                            activationThreshold=1,
                            initialPermanence=0.21,
                            connectedPermanence=0.2,
                            permanenceIncrement=0.1,
                            permanenceDecrement=0.1,
                            minThreshold=1,
                            maxNewSynapseCount=1,
                            seed=1)
        col = 1
        all_cells = tm.cellsForColumn(col)

        # Reset TM state
        tm.unpredictedColumns = set()
        tm.prevPredictiveCells = set()
        tm.prevActiveCells = set(range(4 * 3))  # All possible cells

        # Burst column and check results
        connections = Connections()
        segments_per_cell = {
            cell: connections.segments_for_cell(cell)
            for cell in connections.allCells()
        }
        winner = tm.burst_column(col, all_cells, segments_per_cell, learn=True)

        # All cells in the column should be active
        for c in all_cells:
            assert c in tm.activeCells

        # One and only one winner cell should be added
        assert winner in tm.winnerCells
        assert len([c for c in tm.winnerCells if c in all_cells]) == 1

    def test_create_segment_on_burst(self):
        from htm_py.temporal_memory import TemporalMemory
        from htm_py.connections import Connections

        # Setup TM with simplified params
        tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=1,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            seed=42
        )

        # Force known previous state
        tm.prevActiveCells = set([0])  # Simulated prior active state
        tm.prevWinnerCells = set([0])
        tm.prevPredictiveCells = set()

        # Active input: column 2 (no prediction, so it should burst)
        tm.compute([2], learn=True)

        # Check if a segment was created on a cell in column 2
        col_cells = tm.cellsForColumn(2)
        segs_created = [
            seg for cell in col_cells
            for seg in tm.connections.segments_for_cell(cell)
        ]

        assert len(segs_created) == 1, "Expected one segment to be created on burst"
        created_segment = segs_created[0]
        assert len(created_segment.synapses) == 1, "Expected 1 synapse to be formed"
        assert list(created_segment.synapses)[0].presynapticCell == 0, "Expected synapse to connect to previous active cell"

        print("✅ test_create_segment_on_burst passed.")

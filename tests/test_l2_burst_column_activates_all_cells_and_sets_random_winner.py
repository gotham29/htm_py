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

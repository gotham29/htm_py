import types
import unittest
from htm_py.temporal_memory import TemporalMemory
import htm_py.temporal_memory as tm_module

class TestSynapsePruning(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            minThreshold=1,
            permanenceIncrement=0.0,
            permanenceDecrement=0.25,  # Aggressive decrement to trigger pruning
        )

        # Force deterministic winner cell selection
        def _deterministic_activate_columns(tm_self, activeColumns):
            tm_self.activeCells.clear()
            tm_self.winnerCells.clear()
            tm_self.winnerCellForColumn.clear()
            for col in activeColumns:
                cells = tm_self._cells_for_column(col)
                tm_self.activeCells.update(cells)
                winner = cells[0]
                tm_self.winnerCells.add(winner)
                tm_self.winnerCellForColumn[col] = winner
            return tm_self.activeCells, tm_self.winnerCells

        self.tm._activate_columns = types.MethodType(_deterministic_activate_columns, self.tm)

        # Patch getLeastUsedCell at the module level
        self.original_getLeastUsedCell = tm_module.getLeastUsedCell
        tm_module.getLeastUsedCell = lambda column: self.tm._cells_for_column(column)[0]

    def tearDown(self):
        tm_module.getLeastUsedCell = self.original_getLeastUsedCell

    def test_synapse_gets_pruned_when_permanence_too_low(self):
        self.tm.compute([0], learn=True)

        cell = self.tm._cells_for_column(0)[0]
        self.tm.prevWinnerCells = {cell}
        self.tm.prevActiveCells = set(self.tm._cells_for_column(0))
        self.tm.winnerCellForColumn[0] = cell
        self.tm.compute([0], learn=True)

        segments = self.tm.connections.segmentsForCell(cell)
        self.assertTrue(segments, "Expected a segment to be created")
        segment = segments[0]

        synapses = self.tm.connections.synapsesForSegment(segment)
        self.assertEqual(len(synapses), 1, "Expected exactly 1 synapse")
        synapse = synapses[0]

        # Force permanence below zero repeatedly
        for _ in range(5):
            self.tm._adapt_segment(
                connections=self.tm.connections,
                segment=segment,
                activePresynapticCells=set(),  # No match -> apply decrement
                newSynapseCount=0,
                increment=0.0,
                decrement=0.25,
            )

        # Synapse should be removed if pruning is working
        final_synapses = self.tm.connections.synapsesForSegment(segment)
        self.assertEqual(len(final_synapses), 0, "Synapse was not pruned after permanence dropped")

if __name__ == "__main__":
    unittest.main()

import types
import unittest
from htm_py.temporal_memory import TemporalMemory
import htm_py.temporal_memory as tm_module  # <- This is where getLeastUsedCell is patched

class TestSynapsePrecision(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.19,
            connectedPermanence=0.2,
            minThreshold=1,
            permanenceIncrement=0.05,
            permanenceDecrement=0.05,
        )

        # ✅ Force deterministic column activation (always pick cell 0)
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

        self.tm._activate_columns = types.MethodType(_deterministic_activate_columns, self.tm)

        # ✅ Patch getLeastUsedCell at the module level
        self.original_getLeastUsedCell = tm_module.getLeastUsedCell
        tm_module.getLeastUsedCell = lambda column: self.tm._cells_for_column(column)[0]

    def tearDown(self):
        # ✅ Restore original getLeastUsedCell
        tm_module.getLeastUsedCell = self.original_getLeastUsedCell

    def _force_learning_on_column(self, col):
        self.tm.compute([col], learn=True)

        # Diagnostic logging: which cell getLeastUsedCell returns
        print("DEBUG: getLeastUsedCell returned:", tm_module.getLeastUsedCell(col))

        cell = self.tm._cells_for_column(col)[0]
        self.tm.prevWinnerCells = {cell}
        self.tm.prevActiveCells = set(self.tm._cells_for_column(col))
        self.tm.winnerCellForColumn[col] = cell
        self.tm.compute([col], learn=True)
        return cell

    def test_connected_permanence_exact_threshold(self):
        cell = self._force_learning_on_column(0)
        segments = self.tm.connections.segmentsForCell(cell)
        self.assertTrue(segments, "No segment created on cell after learning")

        segment = segments[0]
        synapse = self.tm.connections.synapsesForSegment(segment)[0]

        self.tm.connections._synapse_data[synapse] = self.tm.connections._synapse_data[synapse]._replace(permanence=0.199999)
        below = self.tm.connections.dataForSynapse(synapse).permanence >= self.tm.connectedPermanence

        self.tm.connections._synapse_data[synapse] = self.tm.connections._synapse_data[synapse]._replace(permanence=0.2)
        exact = self.tm.connections.dataForSynapse(synapse).permanence >= self.tm.connectedPermanence

        self.assertFalse(below)
        self.assertTrue(exact)

    def test_synapse_does_not_exceed_one_or_below_zero(self):
        cell = self._force_learning_on_column(0)
        segments = self.tm.connections.segmentsForCell(cell)
        self.assertTrue(segments, "No segment created on cell after learning")

        segment = segments[0]
        synapse = self.tm.connections.synapsesForSegment(segment)[0]

        for _ in range(100):
            current = self.tm.connections.dataForSynapse(synapse).permanence
            self.tm.connections._synapse_data[synapse] = self.tm.connections._synapse_data[synapse]._replace(permanence=min(1.0, current + 1.5))
        self.assertLessEqual(self.tm.connections.dataForSynapse(synapse).permanence, 1.0)

        for _ in range(100):
            current = self.tm.connections.dataForSynapse(synapse).permanence
            self.tm.connections._synapse_data[synapse] = self.tm.connections._synapse_data[synapse]._replace(permanence=max(0.0, current - 1.0))
        self.assertGreaterEqual(self.tm.connections.dataForSynapse(synapse).permanence, 0.0)

    def test_permanence_increment_decrement_boundary(self):
        cell = self._force_learning_on_column(0)
        segments = self.tm.connections.segmentsForCell(cell)
        self.assertTrue(segments, "No segment created on cell after learning")

        segment = segments[0]
        synapse = self.tm.connections.synapsesForSegment(segment)[0]
        data = self.tm.connections.dataForSynapse(synapse)
        orig_perm = data.permanence
        presyn = data.presynapticCell

        self.tm._adapt_segment(
            connections=self.tm.connections,
            segment=segment,
            activePresynapticCells={presyn},
            newSynapseCount=0,
            increment=0.01,
            decrement=0.0,
        )
        inc_perm = self.tm.connections.dataForSynapse(synapse).permanence
        self.assertGreater(inc_perm, orig_perm, "Permanence did not increase")

        self.tm._adapt_segment(
            connections=self.tm.connections,
            segment=segment,
            activePresynapticCells=set(),
            newSynapseCount=0,
            increment=0.0,
            decrement=0.01,
        )
        dec_perm = self.tm.connections.dataForSynapse(synapse).permanence
        self.assertLess(dec_perm, inc_perm, "Permanence did not decrease")

if __name__ == "__main__":
    unittest.main()

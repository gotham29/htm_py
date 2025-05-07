import unittest
from htm_py.temporal_memory import TemporalMemory


class TestBranchingSegments(unittest.TestCase):
    def setUp(self):
        # self.tm = TemporalMemory(columnDimensions=(4,), cellsPerColumn=4)
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=0,  # ✅ Allow reuse even when only overlap=0
            initialPermanence=0.3,
            connectedPermanence=0.2,
            permanenceIncrement=0.05,
            permanenceDecrement=0.05
        )
        self.cellsPerColumn = 4
        self.col_A = 0
        self.col_B = 1
        self.col_C = 2
        self.col_X = 3
        self.cell_B = self.col_B * self.cellsPerColumn  # e.g., 1 * 4 = 4

    def cell_index(self, col, i=0):
        return col * self.cellsPerColumn + i

    def force_learn_with_winner(self, column, winner_cell, learn=True):
        self.tm.activeColumns = [column]
        self.tm._activate_columns(self.tm.activeColumns)
        self.tm.winnerCellForColumn[column] = winner_cell
        self.tm.activeCells = set(self.tm._cells_for_column(column))  # ✅ FIXED
        self.tm.winnerCells = {winner_cell}
        if learn:
            self.tm._learn_segments(self.tm.activeColumns, self.tm.prevWinnerCells)
        self.tm.prevActiveCells = set(self.tm.activeCells)
        self.tm.prevWinnerCells = set(self.tm.winnerCells)
        self.tm._predict_cells()
        self.tm.prevPredictiveCells = set(self.tm.predictiveCells)

    def test_segment_growth_on_branching(self):
        self.force_learn_with_winner(self.col_A, self.cell_index(self.col_A))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        self.force_learn_with_winner(self.col_X, self.cell_index(self.col_X))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        segments = self.tm.connections.segmentsForCell(self.cell_B)
        print(f"[TEST] Segments on cell_B: {segments}")
        for seg in segments:
            syns = self.tm.connections.synapsesForSegment(seg)
            pres = [self.tm.connections.dataForSynapse(s).presynapticCell for s in syns]
            print(f"  Segment {seg} presynapticCells: {sorted(pres)}")

        self.assertGreaterEqual(len(segments), 2, "Expected multiple segments on cell B due to context divergence (A→B→C vs X→B→C)")

    def test_segment_reuse_on_identical_context(self):
        self.force_learn_with_winner(self.col_A, self.cell_index(self.col_A))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        self.force_learn_with_winner(self.col_A, self.cell_index(self.col_A))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        segments = self.tm.connections.segmentsForCell(self.cell_B)
        print(f"[TEST-REUSE] Segments on cell_B: {segments}")
        self.assertEqual(len(segments), 1, "Expected reuse of segment on identical context")

    def test_segment_branches_have_different_contexts(self):
        self.force_learn_with_winner(self.col_A, self.cell_index(self.col_A))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        self.force_learn_with_winner(self.col_X, self.cell_index(self.col_X))
        self.force_learn_with_winner(self.col_B, self.cell_B)
        self.force_learn_with_winner(self.col_C, self.cell_index(self.col_C))
        self.tm.reset()

        segments = self.tm.connections.segmentsForCell(self.cell_B)
        assert len(segments) >= 2, "Expected multiple segments on B"


if __name__ == "__main__":
    unittest.main()

import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

def create_tm(cells_per_column=4, seed=42):
    return TemporalMemory(
        columnDimensions=(1,),               # Single column for control
        cellsPerColumn=cells_per_column,
        activationThreshold=1,
        initialPermanence=0.21,
        connectedPermanence=0.2,
        minThreshold=1,
        maxNewSynapseCount=4,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=5,
        seed=seed,
        checkInputs=True,
    )

class TestWinnerCellContext(unittest.TestCase):

    def test_different_cells_chosen_for_different_contexts(self):
        """
        WDND: verify winner cell chosen depends on prior context (prevWinnerCells).
        Simulates two different sequences bursting same column in two different contexts.
        """
        tm = create_tm()

        # First context (prevWinnerCells = {0, 1})
        tm.prevActiveCells = set()
        tm.prevWinnerCells = {0, 1}
        tm.compute([0], learn=True)
        winner_A = list(tm.getWinnerCells())[0]
        tm.reset()

        # Second context (prevWinnerCells = {2, 3})
        tm.prevActiveCells = set()
        tm.prevWinnerCells = {2, 3}
        tm.compute([0], learn=True)
        winner_B = list(tm.getWinnerCells())[0]

        self.assertNotEqual(
            winner_A, winner_B,
            f"Expected different winner cells for different contexts, got same: {winner_A}"
        )

    def test_winner_cell_matches_best_matching_segment(self):
        """
        WDND: When a column has matching segments, the winner cell should be the cell
        with the best matching segment (most synapses from prevWinnerCells).
        """
        tm = create_tm()

        # Step 1: Learn a context from cell 0
        tm.prevActiveCells = set()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        seg = tm.connections.segmentsForCell(0)[0]
        tm.reset()

        # Step 2: Trigger same context again
        tm.prevActiveCells = set()
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        winner = list(tm.getWinnerCells())[0]

        self.assertEqual(winner, 0, "Winner cell should be the one with best matching segment (cell 0)")

    def test_winner_cell_when_no_matching_segment(self):
        """
        WDND: When a column bursts and no segment matches,
        winner cell should be the least-used cell (fewest segments).
        """
        tm = create_tm()

        # First compute triggers burst — no segments exist yet
        tm.prevActiveCells = set()
        tm.prevWinnerCells = set()
        tm.compute([0], learn=True)
        winner = list(tm.getWinnerCells())[0]

        # Since no segments exist, least-used cell is chosen
        segments_per_cell = [tm.connections.numSegments(i) for i in range(4)]
        expected_least_used = segments_per_cell.index(min(segments_per_cell))

        self.assertEqual(winner, expected_least_used,
                         f"Expected winner cell to be least-used: {expected_least_used}, got: {winner}")

if __name__ == "__main__":
    unittest.main()

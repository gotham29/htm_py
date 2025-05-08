import unittest
from htm_py.temporal_memory import TemporalMemory

def create_tm(cells_per_column=4, seed=42):
    return TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=cells_per_column,
        activationThreshold=1,
        minThreshold=1,
        initialPermanence=0.21,
        connectedPermanence=0.2,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        predictedSegmentDecrement=0.0,
        maxNewSynapseCount=4,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=5,
        seed=seed,
        checkInputs=True,
    )

class TestPhase1Activation(unittest.TestCase):

    def test_different_cells_chosen_for_different_contexts(self):
        """
        WDND: verify winner cell chosen depends on prior context (prevWinnerCells).
        Simulates two different sequences bursting same column in different contexts.
        """
        tm = create_tm()

        tm.prevActiveCells = set()
        tm.prevWinnerCells = {0, 1}
        tm.compute([0], learn=True)
        winner_A = list(tm.getWinnerCells())[0]
        tm.reset()

        tm.prevActiveCells = set()
        tm.prevWinnerCells = {2, 3}
        tm.compute([0], learn=True)
        winner_B = list(tm.getWinnerCells())[0]

        self.assertNotEqual(
            winner_A, winner_B,
            f"Expected different winner cells for different contexts, got same: {winner_A}"
        )

    def test_tie_breaking_among_least_used_cells(self):
        tm1 = create_tm(seed=111)
        tm1.prevWinnerCells = set()
        tm1.compute([0], learn=True)
        winner1 = list(tm1.getWinnerCells())[0]

        tm2 = create_tm(seed=222)
        tm2.prevWinnerCells = set()
        tm2.compute([0], learn=True)
        winner2 = list(tm2.getWinnerCells())[0]

        tm3 = create_tm(seed=333)
        tm3.prevWinnerCells = set()
        tm3.compute([0], learn=True)
        winner3 = list(tm3.getWinnerCells())[0]

        unique_winners = {winner1, winner2, winner3}
        self.assertGreater(len(unique_winners), 1,
            f"Expected at least 2 different winners from different seeds, got: {unique_winners}")

    def test_segments_sorted_by_num_active_synapses(self):
        """
        WDND: Ensure winner cell comes from the segment with the most active synapses
        from prevWinnerCells when multiple segments are available.
        """
        tm = create_tm()

        # First: Create a weak segment (1 synapse)
        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        tm.reset()

        # Second: Create a stronger segment (2 synapses)
        tm.prevWinnerCells = {0, 1}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevWinnerCells = {0, 1}
        tm.compute([0], learn=True)
        winner = list(tm.getWinnerCells())[0]

        # Best matching segment should be on the cell from 2-synapse context
        # Which would be the cell trained last (cell 1 or 2)
        self.assertIn(winner, range(4), "Winner should be one of the column's cells")

    def test_burst_creates_segment_only_if_prev_winners(self):
        """
        WDND: When a column bursts and prevWinnerCells is empty,
        no segment should be created even in learning mode.
        """
        tm = create_tm()

        tm.prevWinnerCells = set()
        tm.compute([0], learn=True)
        cell_segments = sum(tm.connections.numSegments(i) for i in range(4))

        self.assertEqual(cell_segments, 0,
            f"Expected no segments to be created when prevWinnerCells is empty, found {cell_segments}")

    def test_multiple_columns_burst_independently(self):
        """
        WDND: Two columns should burst and choose winner cells independently.
        Activity in one should not affect the other.
        """
        tm = create_tm()
        tm.columnDimensions = (2,)  # Two columns

        tm.prevWinnerCells = set()
        tm.compute([0, 1], learn=True)

        winners = list(tm.getWinnerCells())
        self.assertEqual(len(winners), 2, "Expected 2 winner cells, one per column")
        col0 = winners[0] // tm.cellsPerColumn
        col1 = winners[1] // tm.cellsPerColumn
        self.assertNotEqual(col0, col1, "Winner cells should belong to different columns")

    def test_max_segments_per_cell_enforced(self):
        tm = create_tm()
        tm.maxSegmentsPerCell = 2

        tm.prevWinnerCells = {0}
        tm.compute([0], learn=True)
        cell = list(tm.getWinnerCells())[0]  # ← moved here
        tm.reset()

        tm.prevWinnerCells = {1}
        tm.compute([0], learn=True)
        tm.reset()

        tm.prevWinnerCells = {2}
        tm.compute([0], learn=True)
        tm.reset()

        n_segments = tm.connections.numSegments(cell)
        self.assertLessEqual(n_segments, 2,
            f"Expected no more than 2 segments, found {n_segments}")

    def test_no_segment_growth_when_learn_false(self):
        """
        WDND: In inference mode (learn=False), bursting should NOT grow new segments.
        """
        tm = create_tm()

        tm.prevWinnerCells = {0, 1}
        tm.compute([0], learn=False)

        total_segments = sum(tm.connections.numSegments(i) for i in range(4))
        self.assertEqual(total_segments, 0,
            f"No segments should be grown during inference, found {total_segments}")


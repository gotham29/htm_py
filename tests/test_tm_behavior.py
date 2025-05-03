# import unittest
# from htm_py.temporal_memory import TemporalMemory


# class TestTemporalMemoryBehavior(unittest.TestCase):
#     def setUp(self):
#         self.tm = TemporalMemory(
#             columnDimensions=(4,),
#             cellsPerColumn=4,
#             activationThreshold=1,
#             initialPermanence=0.21,
#             connectedPermanence=0.5,
#             minThreshold=1,
#             maxNewSynapseCount=5,
#             permanenceIncrement=0.1,
#             permanenceDecrement=0.1,
#             predictedSegmentDecrement=0.0,
#             seed=42,
#         )

#     def setUp(self):
#         self.tm = TemporalMemory(
#             columnDimensions=(4,),
#             cellsPerColumn=2,
#             activationThreshold=1,
#             initialPermanence=0.3,
#             connectedPermanence=0.5,
#             minThreshold=1,
#             permanenceIncrement=0.1,
#             permanenceDecrement=0.1,
#             predictedSegmentDecrement=0.0,
#             maxNewSynapseCount=3
#         )

#     def test_predicted_cells_activate_correctly(self):
#         prev_active_cells = [0, 1, 2, 3]
#         self.tm.prevActiveCells = set(prev_active_cells)  # Simulate previous active cells

#         col1_cells = self.tm.cells_for_column(1)
#         target_cell = col1_cells[0]
#         seg = self.tm.connections.create_segment(target_cell)
#         for c in prev_active_cells:
#             self.tm.connections.create_synapse(seg, c, 0.5)

#         self.tm.compute([1], learn=False)
#         self.assertIn(target_cell, self.tm.prevActiveCells)

#     def test_column_burst_activates_all_cells(self):
#         col = 2
#         expected = self.tm.cells_for_column(col)
#         self.tm.compute([col], learn=False)
#         for cell in expected:
#             self.assertIn(cell, self.tm.prevActiveCells)

#     def test_one_winner_per_burst_column(self):
#         col = 3
#         self.tm.compute([col], learn=False)  # Priming step to set prevWinnerCells
#         self.tm.compute([col], learn=True)   # Real test step
#         # self.tm.compute([col], learn=True)
#         burst_cells = self.tm.cells_for_column(col)
#         winners = [c for c in self.tm.prevWinnerCells if c in burst_cells]
#         self.assertEqual(len(winners), 1)

#     def test_predicted_cells_are_winners(self):
#         self.tm.connectedPermanence = 0.5
#         self.tm.minThreshold = 2

#         prev_active_cells = [0, 1, 2]
#         self.tm.prevActiveCells = set(prev_active_cells)

#         col = 1
#         self.tm.compute([col], learn=False)  # Priming step: sets up winner cell
#         winner_cell = list(self.tm.prevWinnerCells)[0]  # Get the true winner selected by HTM

#         # Create a segment on the actual HTM-selected winner cell
#         seg = self.tm.connections.create_segment(winner_cell)
#         for c in prev_active_cells:
#             self.tm.connections.create_synapse(seg, c, 0.5)

#         # Now run learning step
#         self.tm.compute([col], learn=True)

#         # Verify that this same cell remains a winner
#         self.assertIn(winner_cell, self.tm.prevWinnerCells)

#     def test_no_segment_without_winner_cells(self):
#         """
#         If a column bursts but there are no matching segments and no
#         previous winner cells, it should not create any synapses.
#         """
#         col = 1
#         self.tm.prevWinnerCells = set()  # explicitly empty
#         self.tm.prevActiveCells = set()  # ensures no predictions possible

#         self.tm.compute([col], learn=True)

#         segments = self.tm.connections.segments_for_cell(self.tm.cells_for_column(col)[0])
#         self.assertEqual(len(segments), 1)
#         self.assertEqual(len(segments[0].synapses), 0)

#     def test_no_segment_without_winner_cells(self):
#         self.tm.prevWinnerCells = set()
#         self.tm.prevActiveCells = set()
#         self.tm.compute([1], learn=True)
#         segments = self.tm.connections.segments_for_cell(self.tm.cells_for_column(1)[0])
#         self.assertEqual(segments, [])

#     def test_predicted_cells_are_winners(self):
#         self.tm.connectedPermanence = 0.5
#         self.tm.minThreshold = 2
#         prev_active_cells = [0, 1, 2]
#         self.tm.prevActiveCells = set(prev_active_cells)
#         col = 1
#         self.tm.compute([col], learn=False)
#         winner_cell = list(self.tm.prevWinnerCells)[0]
#         seg = self.tm.connections.create_segment(winner_cell)
#         for c in prev_active_cells:
#             self.tm.connections.create_synapse(seg, c, 0.5)
#         self.tm.compute([col], learn=True)

#     def test_winner_cells_can_include_multiple(self):
#         self.tm.connectedPermanence = 0.5
#         self.tm.minThreshold = 1
#         self.tm.prevActiveCells = set(range(6))
#         col = 1
#         self.tm.compute([col], learn=False)
#         winner_cell = list(self.tm.prevWinnerCells)[0]
#         seg1 = self.tm.connections.create_segment(winner_cell)
#         for c in range(3):
#             self.tm.connections.create_synapse(seg1, c, 0.5)
#         seg2 = self.tm.connections.create_segment(winner_cell)
#         for c in range(3, 6):
#             self.tm.connections.create_synapse(seg2, c, 0.5)
#         self.tm.compute([col], learn=True)
#         self.assertIn(winner_cell, self.tm.winnerCells)

#     def test_reset_clears_state(self):
#         self.tm.compute([0], learn=True)
#         self.tm.reset()
#         self.assertEqual(self.tm.activeCells, set())
#         self.assertEqual(self.tm.winnerCells, set())
#         self.assertEqual(self.tm.predictiveCells, set())

#     def test_column_dimensions_produce_correct_cell_count(self):
#         tm = TemporalMemory(columnDimensions=(3, 2), cellsPerColumn=4)
#         self.assertEqual(tm.number_of_columns(), 6)
#         self.assertEqual(tm.number_of_cells(), 24)


#     # --- Inserted: Missing Behavioral Tests ---

#     def test_activation_threshold_is_respected(self):
#         self.tm.activationThreshold = 3
#         prev_active = [0, 1]
#         self.tm.prevActiveCells = set(prev_active)
#         col = 2
#         winner_cell = self.tm.cells_for_column(col)[0]
#         seg = self.tm.connections.create_segment(winner_cell)
#         for c in prev_active:
#             self.tm.connections.create_synapse(seg, c, 0.5)
#         self.tm.compute([col], learn=False)
#         self.assertNotIn(winner_cell, self.tm.predictiveCells)

#     def test_bursting_column_creates_activity_in_all_cells(self):
#         self.tm.prevActiveCells = set()
#         self.tm.compute([2], learn=False)
#         expected = set(self.tm.cells_for_column(2))
#         self.assertTrue(expected.issubset(self.tm.activeCells))

#     def test_segments_are_not_adapted_without_learning(self):
#         self.tm.compute([0], learn=True)
#         winner_cell = list(self.tm.winnerCells)[0]
#         seg = self.tm.connections.segments_for_cell(winner_cell)[0]
#         for c in [0, 1]:
#             self.tm.connections.create_synapse(seg, c, 0.3)
#         before = [s.permanence for s in seg.synapses]
#         self.tm.compute([0], learn=False)
#         after = [s.permanence for s in seg.synapses]
#         self.assertEqual(before, after)


# if __name__ == "__main__":
#     unittest.main()


import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTemporalMemoryBehavior(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=2,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.5,
            minThreshold=1,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            maxNewSynapseCount=3
        )

    def test_predicted_cells_are_winners(self):
        self.tm.connectedPermanence = 0.5
        self.tm.minThreshold = 2
        prev_active_cells = [0, 1, 2]
        self.tm.prevActiveCells = set(prev_active_cells)
        col = 1
        self.tm.compute([col], learn=False)
        winner_cell = list(self.tm.prevWinnerCells)[0]
        seg = self.tm.connections.create_segment(winner_cell)
        for c in prev_active_cells:
            self.tm.connections.create_synapse(seg, c, 0.5)
        self.tm.compute([col], learn=True)

    def test_winner_cells_can_include_multiple(self):
        self.tm.connectedPermanence = 0.5
        self.tm.minThreshold = 1
        self.tm.prevActiveCells = set(range(6))
        col = 1
        self.tm.compute([col], learn=False)
        winner_cell = list(self.tm.prevWinnerCells)[0]
        seg1 = self.tm.connections.create_segment(winner_cell)
        for c in range(3):
            self.tm.connections.create_synapse(seg1, c, 0.5)
        seg2 = self.tm.connections.create_segment(winner_cell)
        for c in range(3, 6):
            self.tm.connections.create_synapse(seg2, c, 0.5)
        self.tm.compute([col], learn=True)
        self.assertIn(winner_cell, self.tm.winnerCells)

    def test_reset_clears_state(self):
        self.tm.compute([0], learn=True)
        self.tm.reset()
        self.assertEqual(self.tm.activeCells, set())
        self.assertEqual(self.tm.winnerCells, set())
        self.assertEqual(self.tm.predictiveCells, set())

    def test_column_dimensions_produce_correct_cell_count(self):
        tm = TemporalMemory(
            columnDimensions=(3, 2),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )
        self.assertEqual(tm.number_of_columns(), 6)
        self.assertEqual(tm.number_of_cells(), 24)

    def test_activation_threshold_is_respected(self):
        self.tm.activationThreshold = 3
        prev_active = [0, 1]
        self.tm.prevActiveCells = set(prev_active)
        col = 2
        winner_cell = self.tm.cells_for_column(col)[0]
        seg = self.tm.connections.create_segment(winner_cell)
        for c in prev_active:
            self.tm.connections.create_synapse(seg, c, 0.5)
        self.tm.compute([col], learn=False)
        self.assertNotIn(winner_cell, self.tm.predictiveCells)

    def test_bursting_column_creates_activity_in_all_cells(self):
        self.tm.prevActiveCells = set()
        self.tm.compute([2], learn=False)
        expected = set(self.tm.cells_for_column(2))
        self.assertTrue(expected.issubset(self.tm.activeCells))

    def test_segments_are_not_adapted_without_learning(self):
        self.tm.compute([0], learn=True)
        winner_cell = list(self.tm.winnerCells)[0]
        seg = self.tm.connections.segments_for_cell(winner_cell)[0]
        for c in [0, 1]:
            self.tm.connections.create_synapse(seg, c, 0.3)
        before = [s.permanence for s in seg.synapses]
        self.tm.compute([0], learn=False)
        after = [s.permanence for s in seg.synapses]
        self.assertEqual(before, after)

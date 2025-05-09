# Placeholder for htm_py/__init__.py
import os
import unittest
import pandas as pd
from htm_py.temporal_memory import TemporalMemory

class TestPhase1Activation(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            column_dimensions=[100],  # Match the test columns
            cells_per_column=4,       # Ensure this is 4!
            activation_threshold=2,
            initial_permanence=0.21,
            connected_permanence=0.5,
            min_threshold=1,
            max_new_synapse_count=5,
            permanence_increment=0.1,
            permanence_decrement=0.1,
            predicted_segment_decrement=0.01
        )

        # Initialize debug log ONCE here
        os.makedirs("results", exist_ok=True)
        debug_log_path = "results/segment_adapt_debug.csv"
        with open(debug_log_path, "w") as f:
            f.write("timestep,segment_id,synapse_id,prev_perm,new_perm,event\n")

    def test_activate_dendrites_sets_active_segments(self):
        # Simulate some active cells to drive activation
        self.tm.active_cells = {0, 1}
        # Manually create a segment with enough active connected synapses
        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0, 1}, 0.6, 2)  # Permanence above connected threshold

        self.tm.activate_dendrites(learn=False)

        self.assertIn(segment, self.tm.active_segments, "Segment should be active after dendrite activation.")

    def test_matching_segments_set_correctly(self):
        self.tm.active_cells = {0}
        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0}, 0.4, 1)  # Below connected, but above min_threshold for matching

        self.tm.activate_dendrites(learn=False)

        self.assertIn(segment, self.tm.matching_segments, "Segment should be matching based on potential synapses.")

    def test_predicted_segment_decrement_only_applied_to_failed_segments(self):
        # Create two segments, one will become active, the other not
        self.tm.active_cells = {2}  # Only presynaptic cell 2 is active

        segment_active = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment_active, {0, 1}, 0.6, 2)

        segment_inactive = self.tm.connections.create_segment(1)
        self.tm.connections.grow_synapses(segment_inactive, {2, 3}, 0.4, 2)  # Synapses to cells 2 and 3

        # Lower connected_permanence to ensure 0.4 is "potential"
        self.tm.connected_permanence = 0.3

        # Phase 1 Activation with learning
        self.tm.activate_dendrites(learn=True)

        # Confirm adaptation occurred via debug log
        debug_df = pd.read_csv("results/segment_adapt_debug.csv")
        adapted_segments = debug_df["segment_id"].unique()
        self.assertIn(segment_inactive, adapted_segments, 
            "Segment adaptation was not called for inactive segment as expected.")

        # Verify permanence updates: 
        # - Synapse to active cell 2 should remain at 0.4 (no decrement).
        # - Synapse to inactive cell 3 should have decremented to 0.39.
        for synapse in self.tm.connections.synapses_for_segment(segment_inactive):
            presynaptic_cell, perm = self.tm.connections.synapse_data_for(synapse)
            if presynaptic_cell == 2:
                self.assertAlmostEqual(perm, 0.4, places=5, 
                    msg="Synapse to active cell 2 should not have decremented.")
            elif presynaptic_cell == 3:
                self.assertAlmostEqual(perm, 0.39, places=5, 
                    msg="Synapse to inactive cell 3 should have decremented.")
            else:
                self.fail(f"Unexpected presynaptic cell {presynaptic_cell} found.")

        # Verify active segment permanence remains unchanged
        for synapse in self.tm.connections.synapses_for_segment(segment_active):
            _, perm = self.tm.connections.synapse_data_for(synapse)
            self.assertAlmostEqual(perm, 0.6, places=5, 
                msg="Active segment's synapses should remain at 0.6 permanence.")

    def test_no_matching_segments_when_no_active_cells(self):
        self.tm.active_cells = set()  # Clear active cells

        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0}, 0.4, 1)

        self.tm.activate_dendrites(learn=False)

        self.assertNotIn(segment, self.tm.matching_segments, 
            "No segments should be matching when there are no active cells.")

    def test_synapse_pruning_on_permanence_below_zero(self):
        # Create a segment and add synapses
        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0, 1}, 0.05, 2)  # Very low permanence

        # Artificially decrement permanence to below zero via multiple adaptations
        for _ in range(10):  # Enough iterations to ensure permanence drops to zero
            self.tm.connections.adapt_segment(
                segment, prev_active_cells=set(),  # No active cells → permanence will decrement
                permanence_increment=0.0,
                permanence_decrement=0.01,
                iteration=self.tm.iteration
            )

        # Now explicitly prune synapses with permanence <= 0
        for synapse in list(self.tm.connections.synapses_for_segment(segment)):
            _, perm = self.tm.connections.synapse_data_for(synapse)
            if perm <= 0.0:
                self.tm.connections.destroy_synapse(synapse)

        # Validate all synapses with 0 permanence were pruned
        remaining_synapses = self.tm.connections.synapses_for_segment(segment)
        for synapse in remaining_synapses:
            _, perm = self.tm.connections.synapse_data_for(synapse)
            self.assertGreater(perm, 0.0, "All zero permanence synapses should have been pruned.")

        # Verify that some synapses were actually pruned
        self.assertLess(len(remaining_synapses), 2, "Some synapses should have been pruned.")

    def test_segment_destruction_removes_all_synapses(self):
        # Create a segment and grow synapses
        segment = self.tm.connections.create_segment(0)
        synapses = [
            self.tm.connections.create_synapse(segment, cell, 0.5) for cell in [0, 1, 2]
        ]

        # Ensure synapses were created
        self.assertEqual(len(self.tm.connections.synapses_for_segment(segment)), 3, 
            "Should have created 3 synapses before destruction.")

        # Destroy the segment
        self.tm.connections.destroy_segment(segment)

        # Verify segment is completely removed
        self.assertNotIn(segment, self.tm.connections.segment_to_synapses, 
            "Segment should have been fully removed from segment_to_synapses.")

        # Verify all associated synapses are removed
        for synapse in synapses:
            self.assertNotIn(synapse, self.tm.connections.synapse_data, 
                f"Synapse {synapse} should have been removed when segment was destroyed.")

        # Verify the segment is no longer associated with the cell
        self.assertNotIn(segment, self.tm.connections.cell_to_segments.get(0, []), 
            "Segment should have been removed from the owning cell.")

    def test_winner_cell_selection_with_equal_segments(self):
        # Setup: Two cells in the same column with zero segments (tie condition)
        column = 0
        cells = self.tm.cells_for_column(column)  # e.g., [0, 1]

        # Confirm no segments exist yet
        for cell in cells:
            self.assertEqual(self.tm.connections.num_segments(cell), 0)

        # Call select_winner_cell multiple times to ensure both cells get selected over time
        selection_counts = {cell: 0 for cell in cells}
        for _ in range(1000):  # Large number of trials to test fairness
            winner = self.tm.select_winner_cell(column)
            self.assertIn(winner, cells, "Winner cell must belong to the correct column.")
            selection_counts[winner] += 1

        # Verify both cells were selected at least once (fairness check)
        for cell in cells:
            self.assertGreater(selection_counts[cell], 0, 
                f"Cell {cell} should have been selected at least once.")

        # Optional: Verify selection distribution is approximately uniform
        diff = abs(selection_counts[cells[0]] - selection_counts[cells[1]])
        self.assertLess(diff, 200, "Winner cell selection should be reasonably balanced.")

    def test_synapse_growth_respects_max_new_synapses(self):
        # Create a segment and provide more candidate presynaptic cells than allowed by max_new_synapses
        segment = self.tm.connections.create_segment(0)
        candidate_presynaptic_cells = set(range(10))  # 10 possible presynaptic cells

        max_allowed_synapses = 5  # Explicit limit for growth
        self.tm.connections.grow_synapses(
            segment,
            prev_winner_cells=candidate_presynaptic_cells,
            initial_permanence=0.5,
            max_new_synapses=max_allowed_synapses
        )

        # Verify no more than max_new_synapses were created
        synapses = self.tm.connections.synapses_for_segment(segment)
        self.assertLessEqual(len(synapses), max_allowed_synapses, 
            "Number of new synapses should not exceed max_new_synapses limit.")

        # Verify that all presynaptic cells in the new synapses are from the candidate set
        for synapse in synapses:
            presynaptic_cell, _ = self.tm.connections.synapse_data_for(synapse)
            self.assertIn(presynaptic_cell, candidate_presynaptic_cells, 
                "Synapse connected to an invalid presynaptic cell.")

    def test_no_matching_segments_with_high_min_threshold(self):
        # Create a segment with 5 active potential synapses
        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0, 1, 2, 3, 4}, 0.5, 5)

        # Activate all 5 presynaptic cells
        self.tm.active_cells = {0, 1, 2, 3, 4}

        # Set min_threshold higher than possible active potential synapses
        self.tm.min_threshold = 10  # Impossible to satisfy

        # Run Phase 1 Activation
        self.tm.activate_dendrites(learn=False)

        # Ensure no segments are marked as matching
        self.assertNotIn(segment, self.tm.matching_segments, 
            "Segment should not be matching when min_threshold is impossibly high.")

    def test_large_scale_segment_and_synapse_stability(self):
        num_columns = 100
        cells_per_column = 4
        num_segments_per_cell = 10
        num_synapses_per_segment = 20

        # Create a large number of segments and synapses
        for col in range(num_columns):
            assert len(self.tm.cells_for_column(col)) == 4, "Each column should have exactly 4 cells."

            for cell in self.tm.cells_for_column(col):
                assert len(self.tm.cells_for_column(col)) == 4, "Each column should have exactly 4 cells."
                for _ in range(num_segments_per_cell):
                    segment = self.tm.connections.create_segment(cell)
                    presynaptic_cells = set(range(1000))  # Large candidate pool
                    self.tm.connections.grow_synapses(
                        segment, presynaptic_cells, 0.5, num_synapses_per_segment
                    )

        total_segments = sum(len(self.tm.connections.segments_for_cell(cell)) 
                            for col in range(num_columns) 
                            for cell in self.tm.cells_for_column(col))
        
        # Verify segment and synapse counts match expected scale
        self.assertEqual(total_segments, num_columns * cells_per_column * num_segments_per_cell, 
            "Unexpected total segment count under stress test.")

        total_synapses = len(self.tm.connections.synapse_data)
        expected_synapses = total_segments * num_synapses_per_segment
        self.assertEqual(total_synapses, expected_synapses, 
            "Unexpected total synapse count under stress test.")


if __name__ == '__main__':
    unittest.main()

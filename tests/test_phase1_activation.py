# Placeholder for htm_py/__init__.py
import unittest
import pandas as pd
from htm_py.temporal_memory import TemporalMemory

class TestPhase1Activation(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            column_dimensions=[4],
            cells_per_column=2,
            activation_threshold=2,
            initial_permanence=0.21,
            connected_permanence=0.5,
            min_threshold=1,
            max_new_synapse_count=5,
            permanence_increment=0.1,
            permanence_decrement=0.1,
            predicted_segment_decrement=0.01
        )

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
        self.tm.active_cells = {0, 1, 2, 3}

        segment_active = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment_active, {0, 1}, 0.6, 2)  # Above connected permanence

        segment_inactive = self.tm.connections.create_segment(1)
        self.tm.connections.grow_synapses(segment_inactive, {2, 3}, 0.4, 2)  # Exactly at 0.4 permanence

        # Lower connected_permanence to ensure 0.4 is "potential"
        self.tm.connected_permanence = 0.3

        # Phase 1 Activation with learning
        self.tm.activate_dendrites(learn=True)

        # Check debug log to confirm adaptation occurred
        debug_df = pd.read_csv("results/segment_adapt_debug.csv")
        adapted_segments = debug_df["segment_id"].unique()
        self.assertIn(segment_inactive, adapted_segments, 
            "Segment adaptation was not called for inactive segment as expected.")

        # Verify the inactive segment received correct decrement to 0.39
        for synapse in self.tm.connections.synapses_for_segment(segment_inactive):
            _, perm = self.tm.connections.synapse_data_for(synapse)
            self.assertAlmostEqual(perm, 0.39, places=5, 
                msg="Permanence should have decreased to 0.39 for inactive predictive segment.")

        # Verify the active segment permanence remains at 0.6
        for synapse in self.tm.connections.synapses_for_segment(segment_active):
            _, perm = self.tm.connections.synapse_data_for(synapse)
            self.assertAlmostEqual(perm, 0.6, places=5, 
                msg="Permanence should remain at 0.6 for active segment.")

    def test_no_matching_segments_when_no_active_cells(self):
        self.tm.active_cells = set()  # Clear active cells

        segment = self.tm.connections.create_segment(0)
        self.tm.connections.grow_synapses(segment, {0}, 0.4, 1)

        self.tm.activate_dendrites(learn=False)

        self.assertNotIn(segment, self.tm.matching_segments, 
            "No segments should be matching when there are no active cells.")


if __name__ == '__main__':
    unittest.main()

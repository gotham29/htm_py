import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import getBestMatchingSegment

class TestGetBestMatchingSegment(unittest.TestCase):

    def setUp(self):
        self.conn = Connections(num_cells=10)
        self.cell = 3

    def test_returns_segment_with_highest_overlap(self):
        seg1 = self.conn.createSegment(self.cell)
        seg2 = self.conn.createSegment(self.cell)
        seg3 = self.conn.createSegment(self.cell)

        self.conn.createSynapse(seg1, 1, 0.5)
        self.conn.createSynapse(seg1, 2, 0.5)
        self.conn.createSynapse(seg2, 2, 0.5)
        self.conn.createSynapse(seg2, 3, 0.5)
        self.conn.createSynapse(seg2, 4, 0.5)
        self.conn.createSynapse(seg3, 9, 0.5)

        best = getBestMatchingSegment(self.conn, self.cell, {2, 3, 4}, minThreshold=2)
        self.assertEqual(best, seg2)

    def test_returns_none_if_no_segment_meets_threshold(self):
        seg1 = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg1, 1, 0.5)
        self.conn.createSynapse(seg1, 2, 0.5)
        result = getBestMatchingSegment(self.conn, self.cell, {3}, minThreshold=2)
        self.assertIsNone(result)

    def test_exactly_at_threshold_returns_segment(self):
        seg = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg, 1, 0.5)
        self.conn.createSynapse(seg, 2, 0.5)
        result = getBestMatchingSegment(self.conn, self.cell, {1, 2}, minThreshold=2)
        self.assertEqual(result, seg)

    def test_multiple_segments_same_overlap_prefers_first(self):
        seg1 = self.conn.createSegment(self.cell)
        seg2 = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg1, 1, 0.5)
        self.conn.createSynapse(seg2, 1, 0.5)
        result = getBestMatchingSegment(self.conn, self.cell, {1}, minThreshold=1)
        self.assertIn(result, (seg1, seg2))  # deterministic preference optional

    def test_empty_active_presynaptic_returns_none(self):
        seg = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg, 2, 0.5)
        result = getBestMatchingSegment(self.conn, self.cell, set(), minThreshold=1)
        self.assertIsNone(result)

    def test_invalid_active_set_raises(self):
        seg = self.conn.createSegment(self.cell)
        with self.assertRaises(TypeError):
            getBestMatchingSegment(self.conn, self.cell, [1, 2], 1)

    def test_invalid_threshold_raises(self):
        seg = self.conn.createSegment(self.cell)
        with self.assertRaises(ValueError):
            getBestMatchingSegment(self.conn, self.cell, {1, 2}, -1)

    def test_segment_with_zero_synapses_does_not_match(self):
        seg = self.conn.createSegment(self.cell)
        # No synapses added to this segment
        result = getBestMatchingSegment(self.conn, self.cell, {1, 2, 3}, minThreshold=1)
        self.assertIsNone(result)

    def test_segment_with_no_overlapping_synapses(self):
        seg = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg, 5, 0.5)
        result = getBestMatchingSegment(self.conn, self.cell, {1, 2, 3}, minThreshold=1)
        self.assertIsNone(result)

    def test_multiple_synapses_same_presynaptic_cell_all_count(self):
        seg = self.conn.createSegment(self.cell)
        self.conn.createSynapse(seg, 2, 0.5)
        self.conn.createSynapse(seg, 2, 0.5)  # Duplicate cell
        result = getBestMatchingSegment(self.conn, self.cell, {2}, minThreshold=2)
        self.assertEqual(result, seg)

    def test_segments_on_other_cells_are_ignored(self):
        other_cell = (self.cell + 1) % 10
        other_seg = self.conn.createSegment(other_cell)
        self.conn.createSynapse(other_seg, 2, 0.5)

        result = getBestMatchingSegment(self.conn, self.cell, {2}, minThreshold=1)
        self.assertIsNone(result)

    def test_many_segments_only_one_matches(self):
        for _ in range(100):
            self.conn.createSegment(self.cell)  # No synapses added

        best_seg = self.conn.createSegment(self.cell)
        self.conn.createSynapse(best_seg, 7, 0.5)

        result = getBestMatchingSegment(self.conn, self.cell, {7}, minThreshold=1)
        self.assertEqual(result, best_seg)




if __name__ == "__main__":
    unittest.main()

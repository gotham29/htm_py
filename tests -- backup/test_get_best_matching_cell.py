import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import getBestMatchingCell


class TestGetBestMatchingCell(unittest.TestCase):

    def setUp(self):
        self.conn = Connections(num_cells=20)
        self.col = [0, 1, 2, 3]

    def test_returns_cell_with_best_overlap(self):
        seg0 = self.conn.createSegment(0)
        seg1 = self.conn.createSegment(1)
        self.conn.createSynapse(seg0, 5, 0.5)
        self.conn.createSynapse(seg1, 5, 0.5)
        self.conn.createSynapse(seg1, 6, 0.5)

        best = getBestMatchingCell(self.conn, self.col, {5, 6}, minThreshold=2)
        self.assertEqual(best, 1)

    def test_returns_least_used_if_no_overlap(self):
        # All segments miss active set
        for c in self.col:
            self.conn.createSegment(c)
        self.conn.createSynapse(self.conn.createSegment(2), 9, 0.5)

        result = getBestMatchingCell(self.conn, self.col, {15}, minThreshold=2)
        # Should return a cell with 1 segment (cell 0, 1, or 3)
        self.assertIn(result, [0, 1, 3])

    def test_empty_column_raises(self):
        with self.assertRaises(ValueError):
            getBestMatchingCell(self.conn, [], {1}, minThreshold=1)

    def test_ties_choose_first_best_overlap(self):
        seg0 = self.conn.createSegment(0)
        seg1 = self.conn.createSegment(1)
        self.conn.createSynapse(seg0, 7, 0.5)
        self.conn.createSynapse(seg1, 7, 0.5)
        result = getBestMatchingCell(self.conn, [0, 1], {7}, minThreshold=1)
        self.assertIn(result, [0, 1])

    def test_all_cells_have_no_segments(self):
        result = getBestMatchingCell(self.conn, self.col, {1, 2, 3}, minThreshold=1)
        self.assertIn(result, self.col)

    def test_cells_with_varying_segment_counts(self):
        self.conn.createSegment(0)
        self.conn.createSegment(0)
        self.conn.createSegment(1)
        result = getBestMatchingCell(self.conn, self.col, {99}, minThreshold=1)
        self.assertIn(result, [2, 3])  # Least used cells

    def test_presynaptic_set_with_invalid_cells_does_not_crash(self):
        seg = self.conn.createSegment(0)
        self.conn.createSynapse(seg, 2, 0.5)
        try:
            getBestMatchingCell(self.conn, self.col, {999, 1000}, minThreshold=1)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

    def test_segment_with_zero_synapses_does_not_match(self):
        for c in self.col:
            seg = self.conn.createSegment(c)
            # no synapses created
        result = getBestMatchingCell(self.conn, self.col, {1, 2}, minThreshold=1)
        self.assertIn(result, self.col)  # fallback to least used

    def test_synapses_with_zero_permanence_count_as_overlap(self):
        seg = self.conn.createSegment(0)
        self.conn.createSynapse(seg, 1, 0.0)  # still counts if presynaptic match
        result = getBestMatchingCell(self.conn, self.col, {1}, minThreshold=1)
        self.assertEqual(result, 0)

    def test_tie_breaker_prefers_better_match_over_least_used(self):
        self.conn.createSegment(0)
        seg = self.conn.createSegment(1)
        self.conn.createSynapse(seg, 7, 0.5)
        result = getBestMatchingCell(self.conn, [0, 1], {7}, minThreshold=1)
        self.assertEqual(result, 1)  # segment with overlap wins

    def test_non_integer_cell_id_raises(self):
        with self.assertRaises(TypeError):
            getBestMatchingCell(self.conn, ["a", 2], {1}, minThreshold=1)


if __name__ == "__main__":
    unittest.main()

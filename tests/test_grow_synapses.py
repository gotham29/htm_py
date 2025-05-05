import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import growSynapsesToSegment

class TestGrowSynapses(unittest.TestCase):

    def setUp(self):
        self.conn = Connections(num_cells=10)
        self.seg = self.conn.createSegment(0)

    def test_create_synapses_basic(self):
        new = growSynapsesToSegment(self.conn, self.seg, [1, 2, 3], 0.4)
        self.assertEqual(len(new), 3)

        for syn in new:
            data = self.conn.dataForSynapse(syn)
            self.assertIn(data.presynapticCell, [1, 2, 3])
            self.assertAlmostEqual(data.permanence, 0.4)

    def test_does_not_duplicate_existing_synapses(self):
        growSynapsesToSegment(self.conn, self.seg, [2, 3], 0.5)
        new = growSynapsesToSegment(self.conn, self.seg, [3, 4], 0.5)

        # Should only create one new synapse (to 4)
        self.assertEqual(len(new), 1)
        data = self.conn.dataForSynapse(new[0])
        self.assertEqual(data.presynapticCell, 4)

    def test_empty_input_returns_empty_list(self):
        new = growSynapsesToSegment(self.conn, self.seg, [], 0.3)
        self.assertEqual(new, [])

    def test_invalid_initial_permanence_raises(self):
        with self.assertRaises(ValueError):
            growSynapsesToSegment(self.conn, self.seg, [1], -0.1)
        with self.assertRaises(ValueError):
            growSynapsesToSegment(self.conn, self.seg, [1], 1.5)
        with self.assertRaises(ValueError):
            growSynapsesToSegment(self.conn, self.seg, [1], "not-a-number")

    def test_segment_integrity_after_growth(self):
        before = set(self.conn.synapsesForSegment(self.seg))
        growSynapsesToSegment(self.conn, self.seg, [5], 0.25)
        after = set(self.conn.synapsesForSegment(self.seg))
        self.assertTrue(after.issuperset(before))

    def test_idempotency_with_duplicate_requests(self):
        growSynapsesToSegment(self.conn, self.seg, [1, 2], 0.45)
        growSynapsesToSegment(self.conn, self.seg, [2, 3], 0.45)
        all_synapses = self.conn.synapsesForSegment(self.seg)

        presynaptic = {self.conn.dataForSynapse(s).presynapticCell for s in all_synapses}
        self.assertEqual(presynaptic, {1, 2, 3})

    def test_presynaptic_cell_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            growSynapsesToSegment(self.conn, self.seg, [-1], 0.3)

        with self.assertRaises(IndexError):
            growSynapsesToSegment(self.conn, self.seg, [10], 0.3)  # num_cells = 10 → valid range = [0–9]

    def test_extreme_valid_permanence_values(self):
        # Lower bound
        new = growSynapsesToSegment(self.conn, self.seg, [1], 0.0)
        self.assertEqual(self.conn.dataForSynapse(new[0]).permanence, 0.0)

        # Upper bound
        new = growSynapsesToSegment(self.conn, self.seg, [2], 1.0)
        self.assertEqual(self.conn.dataForSynapse(new[0]).permanence, 1.0)

    def test_massive_number_of_presynaptic_cells(self):
        conn = Connections(num_cells=1000)
        seg = conn.createSegment(0)  # Fix: use same instance!
        presynaptic = list(range(500))  # 500 valid unique cells
        new = growSynapsesToSegment(conn, seg, presynaptic, 0.25)
        self.assertEqual(len(new), 500)

    def test_no_op_when_all_presynaptic_already_connected(self):
        growSynapsesToSegment(self.conn, self.seg, [1, 2, 3], 0.3)
        before = set(self.conn.synapsesForSegment(self.seg))
        new = growSynapsesToSegment(self.conn, self.seg, [1, 2, 3], 0.9)  # no-op
        after = set(self.conn.synapsesForSegment(self.seg))

        self.assertEqual(new, [])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
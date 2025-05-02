import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTemporalMemoryLearning(unittest.TestCase):

    def test_reinforce_active_segments(self):
        tm = TemporalMemory((1,), 4, 3, 0.2, 0.5, 2, 4, 0.1, 0.08, 0.02, 42)
        prev = [0, 1, 2, 3]
        cell = 5
        seg = tm.connections.create_segment(cell)
        s_active = [tm.connections.create_synapse(seg, i, 0.5) for i in prev[:3]]
        s_inactive = tm.connections.create_synapse(seg, 81, 0.5)
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)
        for s in s_active:
            p = tm.connections.data_for_synapse(s).permanence
            self.assertAlmostEqual(p, 0.6, delta=1e-6)
        p = tm.connections.data_for_synapse(s_inactive).permanence
        self.assertAlmostEqual(p, 0.42, delta=1e-6)

    def test_burst_selects_best_matching_segment(self):
        tm = TemporalMemory((1,), 4, 3, 0.21, 0.5, 2, 3, 0.1, 0.08, 0.0, 42)
        prev = [0, 1, 2, 3]
        seg_best = tm.connections.create_segment(4)
        [tm.connections.create_synapse(seg_best, c, 0.3) for c in prev[:3]]
        tm.connections.create_synapse(seg_best, 81, 0.3)
        seg_comp = tm.connections.create_segment(5)
        [tm.connections.create_synapse(seg_comp, c, 0.3) for c in prev[:2]]
        tm.connections.create_synapse(seg_comp, 81, 0.3)
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)
        perms = [s.permanence for s in tm.connections.synapses_for_segment(seg_best)]
        self.assertTrue(all(0.39 < p < 0.41 for p in perms[:3]))

    def test_nonselected_segments_unchanged(self):
        tm = TemporalMemory((1,), 4, 3, 0.21, 0.5, 2, 3, 0.1, 0.08, 0.0, 42)
        prev = [0, 1, 2, 3]
        seg_best = tm.connections.create_segment(4)
        [tm.connections.create_synapse(seg_best, c, 0.3) for c in prev[:3]]
        seg_other = tm.connections.create_segment(5)
        s1 = tm.connections.create_synapse(seg_other, prev[0], 0.3)
        s2 = tm.connections.create_synapse(seg_other, prev[1], 0.3)
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)
        for s in [s1, s2]:
            self.assertAlmostEqual(s.permanence, 0.3, delta=1e-6)

    def test_predicted_column_matching_segments_unchanged(self):
        tm = TemporalMemory((1,), 4, 3, 0.21, 0.5, 2, 3, 0.1, 0.1, 0.0, 42)
        prev = [0, 1, 2, 3]
        target = 4
        seg_active = tm.connections.create_segment(target)
        [tm.connections.create_synapse(seg_active, c, 0.5) for c in prev]
        seg_match = tm.connections.create_segment(target)
        s1 = tm.connections.create_synapse(seg_match, prev[0], 0.3)
        s2 = tm.connections.create_synapse(seg_match, prev[1], 0.3)
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)
        for s in [s1, s2]:
            self.assertAlmostEqual(s.permanence, 0.3, delta=1e-6)

    def test_destroy_weak_synapse_on_reinforcement(self):
        tm = TemporalMemory((1,), 4, 3, 0.2, 0.5, 2, 4, 0.1, 0.1, 0.02, 42)
        prev = [0, 1, 2, 3]
        seg = tm.connections.create_segment(5)
        [tm.connections.create_synapse(seg, c, 0.5) for c in prev[:3]]
        tm.connections.create_synapse(seg, 81, 0.09)
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)
        self.assertEqual(tm.connections.num_synapses(seg), 3)

    def test_new_segment_add_synapses_to_subset_of_winner_cells(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 2, 0.1, 0.1, 0.0, 42)
        tm.compute([0], learn=True)
        tm.compute([0], learn=True)  # learn again to form a new segment
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        self.assertLessEqual(tm.connections.num_synapses(seg), 2)

    def test_new_segment_add_synapses_to_all_winner_cells(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 10, 0.1, 0.1, 0.0, 42)
        tm.compute([0], learn=True)
        tm.compute([0], learn=True)
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        self.assertGreaterEqual(tm.connections.num_synapses(seg), 1)

    def test_matching_segment_adds_subset_synapses(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 2, 0.1, 0.1, 0.0, 42)
        tm.compute([0], learn=True)
        tm.compute([0], learn=True)
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        self.assertLessEqual(tm.connections.num_synapses(seg), 2)

    def test_matching_segment_adds_all_winner_synapses(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 10, 0.1, 0.1, 0.0, 42)
        tm.compute([0], learn=True)
        tm.compute([0], learn=True)
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        self.assertGreaterEqual(tm.connections.num_synapses(seg), 1)

    def test_active_segment_grow_synapses_according_to_overlap(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 5, 0.1, 0.1, 0.0, 42)
        tm.compute([0], learn=True)
        tm.compute([0], learn=True)
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        self.assertLessEqual(tm.connections.num_synapses(seg), 5)


if __name__ == "__main__":
    unittest.main()

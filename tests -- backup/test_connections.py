# test_connections.py

import unittest
from htm_py.connections import Connections, SynapseData

class TestConnections(unittest.TestCase):

    def setUp(self):
        self.num_cells = 10
        self.conn = Connections(num_cells=self.num_cells)

    def test_create_segment_basic(self):
        segment_ids = []
        for cell in range(self.num_cells):
            seg = self.conn.createSegment(cell)
            segment_ids.append(seg)
            self.assertEqual(seg, segment_ids[-1])
            self.assertEqual(self.conn._cell_for_segment[seg], cell)
            self.assertIn(seg, self.conn._segments_for_cell[cell])
            self.assertEqual(self.conn._synapses_for_segment[seg], [])
        self.assertEqual(len(set(segment_ids)), self.num_cells)
        self.assertEqual(self.conn._next_segment_id, self.num_cells)

    def test_create_synapse_basic(self):
        segment = self.conn.createSegment(0)
        synapse_ids = []
        for i in range(5):
            presyn_cell = i
            perm = 0.21 + i * 0.01
            syn = self.conn.createSynapse(segment, presyn_cell, perm)
            synapse_ids.append(syn)
            self.assertIn(syn, self.conn._synapses_for_segment[segment])
            self.assertIn(syn, self.conn._synapse_data)
            data = self.conn._synapse_data[syn]
            self.assertEqual(data.presynapticCell, presyn_cell)
            self.assertAlmostEqual(data.permanence, perm)
        self.assertEqual(len(set(synapse_ids)), 5)
        self.assertEqual(self.conn._next_synapse_id, 5)

    def test_create_segment_edge_case_last_cell(self):
        last_cell = self.num_cells - 1
        seg = self.conn.createSegment(last_cell)
        self.assertIn(seg, self.conn._segments_for_cell[last_cell])
        self.assertEqual(self.conn._cell_for_segment[seg], last_cell)

    def test_multiple_segments_same_cell(self):
        cell = 2
        seg1 = self.conn.createSegment(cell)
        seg2 = self.conn.createSegment(cell)
        seg3 = self.conn.createSegment(cell)
        segments = self.conn._segments_for_cell[cell]
        self.assertEqual(len(segments), 3)
        self.assertEqual(set(segments), {seg1, seg2, seg3})
        self.assertTrue(all(self.conn._cell_for_segment[s] == cell for s in segments))

    def test_segment_and_synapse_id_monotonicity(self):
        segs = [self.conn.createSegment(0) for _ in range(3)]
        self.assertEqual(segs, list(range(3)))
        syns = [self.conn.createSynapse(segs[0], i, 0.3) for i in range(3)]
        self.assertEqual(syns, list(range(3)))

    def test_create_synapse_on_multiple_segments(self):
        seg1 = self.conn.createSegment(0)
        seg2 = self.conn.createSegment(1)
        syn1 = self.conn.createSynapse(seg1, 3, 0.5)
        syn2 = self.conn.createSynapse(seg2, 4, 0.6)
        self.assertIn(syn1, self.conn._synapses_for_segment[seg1])
        self.assertIn(syn2, self.conn._synapses_for_segment[seg2])
        self.assertNotEqual(syn1, syn2)
        self.assertEqual(self.conn._synapse_data[syn1].presynapticCell, 3)
        self.assertEqual(self.conn._synapse_data[syn2].presynapticCell, 4)

    def test_create_segment_invalid_cell(self):
        with self.assertRaises(IndexError):
            self.conn.createSegment(-1)
        with self.assertRaises(IndexError):
            self.conn.createSegment(self.num_cells)

    def test_create_synapse_invalid_presynaptic_cell(self):
        segment = self.conn.createSegment(0)
        with self.assertRaises(IndexError):
            self.conn.createSynapse(segment, -1, 0.2)
        with self.assertRaises(IndexError):
            self.conn.createSynapse(segment, self.num_cells, 0.2)

    def test_create_synapse_negative_permanence(self):
        segment = self.conn.createSegment(0)
        syn = self.conn.createSynapse(segment, 1, -0.1)
        data = self.conn._synapse_data[syn]
        self.assertLess(data.permanence, 0.0)

    def test_internal_segment_mapping_consistency(self):
        for cell, segments in self.conn._segments_for_cell.items():
            for seg in segments:
                self.assertEqual(self.conn._cell_for_segment[seg], cell)
        for seg, synapses in self.conn._synapses_for_segment.items():
            for syn in synapses:
                self.assertIn(syn, self.conn._synapse_data)

    def test_create_segment_non_integer(self):
        with self.assertRaises(TypeError):
            self.conn.createSegment("not_an_int")

    def test_create_synapse_non_numeric_permanence(self):
        seg = self.conn.createSegment(0)
        with self.assertRaises(TypeError):
            self.conn.createSynapse(seg, 1, "high")

    def test_create_synapse_nan_permanence(self):
        import math
        seg = self.conn.createSegment(0)
        with self.assertRaises(ValueError):
            self.conn.createSynapse(seg, 1, float('nan'))

    def test_connections_with_zero_cells(self):
        with self.assertRaises(ValueError):
            _ = Connections(num_cells=0)

    def test_destroy_synapse_removes_everywhere(self):
        segment = self.conn.createSegment(0)
        syn = self.conn.createSynapse(segment, 1, 0.3)
        self.assertIn(syn, self.conn._synapse_data)
        self.assertIn(syn, self.conn._synapses_for_segment[segment])
        self.conn.destroySynapse(syn)
        self.assertNotIn(syn, self.conn._synapse_data)
        self.assertNotIn(syn, self.conn._synapses_for_segment[segment])

    def test_destroy_synapse_nonexistent_raises(self):
        with self.assertRaises(KeyError):
            self.conn.destroySynapse(999)

    def test_update_synapse_permanence_successful(self):
        segment = self.conn.createSegment(0)
        syn = self.conn.createSynapse(segment, 1, 0.25)
        self.conn.setSynapsePermanence(syn, 0.75)
        data = self.conn._synapse_data[syn]
        self.assertEqual(data.presynapticCell, 1)
        self.assertAlmostEqual(data.permanence, 0.75)

    def test_update_synapse_permanence_invalid_synapse(self):
        with self.assertRaises(KeyError):
            self.conn.updateSynapsePermanence(999, 0.2)

    def test_update_synapse_permanence_nan(self):
        segment = self.conn.createSegment(0)
        syn = self.conn.createSynapse(segment, 1, 0.1)
        import math
        with self.assertRaises(ValueError):
            self.conn.setSynapsePermanence(syn, math.nan)

    def test_synapses_for_segment_returns_copy(self):
        segment = self.conn.createSegment(0)
        s1 = self.conn.createSynapse(segment, 1, 0.1)
        s2 = self.conn.createSynapse(segment, 2, 0.2)
        syns = self.conn.synapsesForSegment(segment)
        self.assertEqual(set(syns), {s1, s2})
        syns.clear()
        internal = self.conn._synapses_for_segment[segment]
        self.assertEqual(set(internal), {s1, s2})

    def test_synapses_for_segment_invalid_segment(self):
        with self.assertRaises(KeyError):
            self.conn.synapsesForSegment(1234)

    def test_data_for_synapse_returns_correct_data(self):
        segment = self.conn.createSegment(0)
        syn = self.conn.createSynapse(segment, 7, 0.44)
        data = self.conn.dataForSynapse(syn)
        self.assertEqual(data.presynapticCell, 7)
        self.assertAlmostEqual(data.permanence, 0.44)

    def test_data_for_synapse_invalid(self):
        with self.assertRaises(KeyError):
            self.conn.dataForSynapse(2024)

    def test_no_orphan_synapses_after_destroy(self):
        seg = self.conn.createSegment(0)
        syn = self.conn.createSynapse(seg, 1, 0.5)
        self.conn.destroySynapse(syn)
        for syn_list in self.conn._synapses_for_segment.values():
            self.assertNotIn(syn, syn_list)
        total_tracked = sum(len(s) for s in self.conn._synapses_for_segment.values())
        self.assertEqual(total_tracked, len(self.conn._synapse_data))

    def test_destroy_segment_removes_all_synapses_and_links(self):
        cell = 0
        seg = self.conn.createSegment(cell)
        syn1 = self.conn.createSynapse(seg, 1, 0.5)
        syn2 = self.conn.createSynapse(seg, 2, 0.4)
        self.conn.destroySegment(seg)
        self.assertNotIn(seg, self.conn._cell_for_segment)
        self.assertNotIn(seg, self.conn._segments_for_cell[cell])
        self.assertNotIn(seg, self.conn._synapses_for_segment)
        self.assertNotIn(syn1, self.conn._synapse_data)
        self.assertNotIn(syn2, self.conn._synapse_data)

    def test_destroy_segment_invalid_raises(self):
        with self.assertRaises(KeyError):
            self.conn.destroySegment(999)

    def test_segments_for_cell_valid_and_copy(self):
        cell = 0
        seg1 = self.conn.createSegment(cell)
        seg2 = self.conn.createSegment(cell)
        segs = self.conn.segmentsForCell(cell)
        self.assertEqual(set(segs), {seg1, seg2})
        segs.clear()
        internal = self.conn._segments_for_cell[cell]
        self.assertEqual(set(internal), {seg1, seg2})

    def test_segments_for_cell_invalid_raises(self):
        with self.assertRaises(IndexError):
            self.conn.segmentsForCell(-1)
        with self.assertRaises(IndexError):
            self.conn.segmentsForCell(self.conn.numCells())

    def test_cell_for_segment_valid(self):
        cell = 2
        seg = self.conn.createSegment(cell)
        returned = self.conn.cell_for_segment(seg)
        self.assertEqual(returned, cell)

    def test_cell_for_segment_invalid_raises(self):
        with self.assertRaises(KeyError):
            self.conn.cell_for_segment(1234)

    def test_num_segments_and_synapses_match_expectation(self):
        cell = 1
        seg1 = self.conn.createSegment(cell)
        seg2 = self.conn.createSegment(cell)
        self.conn.createSynapse(seg1, 2, 0.1)
        self.conn.createSynapse(seg1, 3, 0.2)
        self.conn.createSynapse(seg2, 4, 0.3)
        self.assertEqual(self.conn.numSegments(cell), 2)
        self.assertEqual(self.conn.numSynapses(seg1), 2)
        self.assertEqual(self.conn.numSynapses(seg2), 1)

    def test_num_cells_and_segmentFlatListLength(self):
        initial_cells = self.conn.numCells()
        self.assertEqual(initial_cells, 10)
        segs = [self.conn.createSegment(0) for _ in range(3)]
        self.assertEqual(self.conn.segmentFlatListLength(), 3)
        self.conn.destroySegment(segs[1])
        self.assertEqual(self.conn.segmentFlatListLength(), 2)

if __name__ == '__main__':
    unittest.main()

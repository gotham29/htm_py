import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTemporalMemoryLimits(unittest.TestCase):

    def test_add_segment_to_cell_with_fewest_segments(self):
        tm = TemporalMemory((1,), 2, 1, 0.3, 0.5, 1, 5, 0.1, 0.1, 0.0,
                            maxSegmentsPerCell=2, maxSynapsesPerSegment=5, seed=42)

        col = 0
        tm.compute([0, 1], learn=True)  # seed winner cells

        for _ in range(4):
            cell_segment_counts = {
                c: len(tm.connections.segments_for_cell(c))
                for c in tm.cells_for_column(col)
            }
            min_segs = min(cell_segment_counts.values())

            tm.compute([col], learn=True)
            winner = tm.get_winner_cells()[0]
            self.assertEqual(len(tm.connections.segments_for_cell(winner)), min_segs + 1)

    def test_create_segment_destroy_old(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 1, 0.1, 0.1, 0.0,
                            maxSegmentsPerCell=1, maxSynapsesPerSegment=5, seed=42)

        tm.compute([0, 1], learn=True)
        tm.compute([0], learn=True)
        first_seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]
        tm.compute([0], learn=True)
        segs = tm.connections.segments_for_cell(tm.get_winner_cells()[0])
        self.assertNotIn(first_seg, segs)

    def test_destroy_segment_with_too_few_synapses(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 5, 0.1, 0.1, 0.0,
                            maxSegmentsPerCell=5, maxSynapsesPerSegment=3, seed=42)

        tm.compute([0, 1], learn=True)
        tm.compute([0], learn=True)
        cell = tm.get_winner_cells()[0]
        seg = tm.connections.segments_for_cell(cell)[0]

        # Remove all synapses
        for syn in tm.connections.synapses_for_segment(seg)[:]:
            tm.connections.destroy_synapse(syn)

        # Destroy all other segments to force this one to match
        for s in tm.connections.segments_for_cell(cell)[:]:
            if s is not seg:
                tm.connections.destroy_segment(s)

        # Manually grow a new synapse to a valid winner, so it appears as matching
        off_cell = (cell + 1) % tm.numCells  # not in prevActiveCells
        tm.connections.create_synapse(seg, off_cell, permanence=0.01)

        # Force adaptation on the segment with no active presynaptic cells
        tm._adapt_segment(seg, prevActiveCells=set())

        # Now run and trigger adaptation
        tm.compute([0], learn=True)

        # Assert segment got destroyed
        segs = tm.connections.segments_for_cell(cell)
        self.assertNotIn(seg, segs)

    def test_destroy_segments_then_reach_limit(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 1, 0.1, 0.1, 0.0,
                            maxSegmentsPerCell=2, maxSynapsesPerSegment=5, seed=42)

        tm.compute([0, 1], learn=True)
        for _ in range(3):
            tm.compute([0], learn=True)

        segs = tm.connections.segments_for_cell(tm.get_winner_cells()[0])
        self.assertEqual(len(segs), 2)

    def test_recycle_weakest_synapse_to_make_room(self):
        tm = TemporalMemory((1,), 1, 1, 0.3, 0.5, 1, 5, 0.1, 0.1, 0.0,
                            maxSegmentsPerCell=1, maxSynapsesPerSegment=3, seed=42)

        tm.compute([0, 1], learn=True)
        tm.compute([0], learn=True)
        seg = tm.connections.segments_for_cell(tm.get_winner_cells()[0])[0]

        for _ in range(4):
            tm.compute([0, 1], learn=True)

        syns = tm.connections.synapses_for_segment(seg)
        self.assertLessEqual(len(syns), 3)

import unittest
from htm_py.temporal_memory import TemporalMemory


def setup_tm_for_helpers():
    tm = TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.2,
        connectedPermanence=0.2,
        minThreshold=1,
        maxNewSynapseCount=4,
        permanenceIncrement=0.05,
        permanenceDecrement=0.02,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=5,
        seed=42,
    )
    return tm


class TestPhase1Helpers(unittest.TestCase):

    def test_get_matching_segments(self):
        tm = setup_tm_for_helpers()
        tm.prevWinnerCells = {0, 1}
        cell = 0
        seg = tm.connections.createSegment(cell)
        tm.connections.createSynapse(seg, 0, 0.3)
        tm.connections.createSynapse(seg, 1, 0.25)

        matches = tm._get_matching_segments(0)
        self.assertIn((seg, 2), matches)
        self.assertEqual(len(matches), 1)

    def test_adapt_segment(self):
        tm = setup_tm_for_helpers()
        tm.prevWinnerCells = {0}
        seg = tm.connections.createSegment(0)
        syn = tm.connections.createSynapse(seg, 0, 0.2)
        tm.connections.createSynapse(seg, 1, 0.5)

        tm._adapt_segment(seg, tm.prevWinnerCells)
        data0 = tm.connections.dataForSynapse(syn)
        self.assertGreaterEqual(data0["permanence"], 0.25)

    def test_grow_segment_if_needed(self):
        tm = setup_tm_for_helpers()
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)

        tm._grow_segment_if_needed(seg)
        syn_cells = {tm.connections.dataForSynapse(syn)["presynapticCell"]
                     for syn in tm.connections.synapsesForSegment(seg)}
        self.assertIn(1, syn_cells)
        self.assertNotIn(2, syn_cells)  # not in prevWinnerCells

    def test_grow_new_segment(self):
        tm = setup_tm_for_helpers()
        tm.prevWinnerCells = {0, 1}
        tm._grow_new_segment(0)
        segments = tm.connections.segmentsForCell(0)
        self.assertEqual(len(segments), 1)
        syn_cells = {tm.connections.dataForSynapse(syn)["presynapticCell"]
                     for syn in tm.connections.synapsesForSegment(segments[0])}
        self.assertEqual(syn_cells, {0, 1})

    def test_get_least_used_cell(self):
        tm = setup_tm_for_helpers()
        cell = tm._get_least_used_cell(0)
        self.assertIn(cell, range(4))

    def test_cells_for_column(self):
        tm = setup_tm_for_helpers()
        cells = tm._cells_for_column(0)
        self.assertEqual(cells, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

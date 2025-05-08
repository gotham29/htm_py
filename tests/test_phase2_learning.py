import unittest
from htm_py.temporal_memory import TemporalMemory

def setup_tm_phase2():
    return TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.3,
        connectedPermanence=0.2,
        minThreshold=1,
        maxNewSynapseCount=3,
        permanenceIncrement=0.05,
        permanenceDecrement=0.02,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=4,
        seed=42
    )

class TestPhase2Learning(unittest.TestCase):

    def test_segment_grows_on_match(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)  # one match

        tm._adapt_segment(seg)
        tm._grow_segment_if_needed(seg)

        syn_cells = {tm.connections.dataForSynapse(s)["presynapticCell"]
                     for s in tm.connections.synapsesForSegment(seg)}
        self.assertTrue(1 in syn_cells, "Expected new synapse to cell 1")

    def test_new_synapses_added_to_segment(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)  # already present

        tm._grow_segment_if_needed(seg)
        syn_cells = [tm.connections.dataForSynapse(s)["presynapticCell"]
                     for s in tm.connections.synapsesForSegment(seg)]
        self.assertEqual(syn_cells.count(0), 1, "Should not duplicate synapse to cell 0")
        self.assertIn(1, syn_cells, "Should add synapse to new cell 1")

    def test_segment_synapse_growth_limited_by_maxSynapses(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0, 1, 2, 3}
        seg = tm.connections.createSegment(0)
        tm._grow_segment_if_needed(seg)

        syns = tm.connections.synapsesForSegment(seg)
        self.assertLessEqual(len(syns), tm.maxSynapsesPerSegment,
                             f"Should not exceed {tm.maxSynapsesPerSegment} synapses")

    def test_segment_not_grown_if_no_prevWinnerCells(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = set()
        seg = tm.connections.createSegment(0)

        tm._grow_segment_if_needed(seg)
        syns = tm.connections.synapsesForSegment(seg)
        self.assertEqual(len(syns), 0, "No synapses should be created without prevWinnerCells")

    def test_adapt_segment_applies_decay_to_inactive_synapses(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0}  # only cell 0 is active

        seg = tm.connections.createSegment(0)
        syn0 = tm.connections.createSynapse(seg, 0, 0.3)  # active
        syn1 = tm.connections.createSynapse(seg, 1, 0.3)  # inactive

        tm._adapt_segment(seg)
        data0 = tm.connections.dataForSynapse(syn0)
        data1 = tm.connections.dataForSynapse(syn1)

        self.assertGreater(data0["permanence"], 0.3, "Synapse to active cell should be incremented")
        self.assertLess(data1["permanence"], 0.3, "Synapse to inactive cell should be decremented")

    def test_segment_replaced_when_over_segment_limit(self):
        tm = setup_tm_phase2()
        tm.maxSegmentsPerCell = 2
        cell = 0

        # Create and track 2 segments
        seg1 = tm.connections.createSegment(cell)
        seg2 = tm.connections.createSegment(cell)
        tm.lastUsedIterationForSegment[seg1] = 10
        tm.lastUsedIterationForSegment[seg2] = 20

        # Now attempt to create a 3rd segment — should evict seg1
        tm.iteration = 30
        seg3 = tm._grow_new_segment(cell)

        segments = tm.connections.segmentsForCell(cell)
        self.assertEqual(len(segments), 2, "Cell should not exceed maxSegmentsPerCell")
        self.assertIn(seg2, segments, "Newer segment should be preserved")
        self.assertIn(seg3, segments, "New segment should be added")
        self.assertNotIn(seg1, segments, "Oldest (LRU) segment should be removed")

    def test_synapse_destroyed_if_permanence_drops_below_zero(self):
        tm = setup_tm_phase2()
        tm.permanenceDecrement = 0.4  # large enough to cross 0
        tm.prevWinnerCells = {0}  # synapse to 1 will decay

        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 1, 0.3)  # not in prevWinnerCells

        tm._adapt_segment(seg)
        syns = tm.connections.synapsesForSegment(seg)
        self.assertEqual(len(syns), 0, "Synapse with low permanence should be removed")

    def test_no_duplicate_synapse_growth_on_segment(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0, 1}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)  # already present

        tm._grow_segment_if_needed(seg)
        syn_cells = [tm.connections.dataForSynapse(s)["presynapticCell"]
                    for s in tm.connections.synapsesForSegment(seg)]

        self.assertEqual(syn_cells.count(0), 1, "Should not duplicate synapse to cell 0")

    def test_adapt_segment_skips_when_no_prev_winners(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = set()

        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)
        tm._adapt_segment(seg)

        # Permanence should remain unchanged
        p = tm.connections.dataForSynapse(
            tm.connections.synapsesForSegment(seg)[0])["permanence"]
        self.assertEqual(p, 0.3, "Permanence should not change without context")

    def test_last_used_iteration_updated_after_learning(self):
        tm = setup_tm_phase2()
        tm.iteration = 42
        seg = tm.connections.createSegment(0)
        tm.prevWinnerCells = {0}
        tm._adapt_segment(seg)
        tm.lastUsedIterationForSegment[seg] = tm.iteration  # simulate update

        self.assertEqual(tm.lastUsedIterationForSegment[seg], 42,
                        "Segment usage iteration should be updated")

    def test_segment_destroyed_if_all_synapses_pruned(self):
        tm = setup_tm_phase2()
        tm.permanenceDecrement = 0.5  # ensure removal
        tm.prevWinnerCells = {1}  # cell 0 will decay

        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.2)
        tm._adapt_segment(seg)

        segments = tm.connections.segmentsForCell(0)
        self.assertEqual(len(segments), 0, "Segment should be destroyed if no synapses remain")


    def test_adapt_segment_clamps_permanence_bounds(self):
        tm = setup_tm_phase2()
        tm.prevWinnerCells = {0}

        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.99)  # active
        tm.connections.createSynapse(seg, 1, 0.01)  # inactive

        tm.permanenceIncrement = 0.2
        tm.permanenceDecrement = 0.2
        tm._adapt_segment(seg)

        for syn in tm.connections.synapsesForSegment(seg):
            p = tm.connections.dataForSynapse(syn)["permanence"]
            self.assertTrue(0.0 <= p <= 1.0, f"Permanence {p} not clamped")

    def test_segment_not_learned_if_below_activationThreshold(self):
        tm = setup_tm_phase2()
        tm.activationThreshold = 3
        tm.prevActiveCells = {0}
        seg = tm.connections.createSegment(0)
        tm.connections.createSynapse(seg, 0, 0.3)

        tm._learn_segments()

        # Segment shouldn't have grown or been touched
        syns = tm.connections.synapsesForSegment(seg)
        self.assertEqual(len(syns), 1, "Segment should not have grown")

    def test_grow_segment_prunes_if_max_synapses_exceeded(self):
        tm = setup_tm_phase2()
        seg = tm.connections.createSegment(0)
        tm.prevWinnerCells = {0, 1, 2, 3, 4}
        for cell in [0, 1, 2, 3]:
            tm.connections.createSynapse(seg, cell, 0.1)  # 4 syns

        # Now force a grow beyond limit
        tm._grow_segment_if_needed(seg)

        syns = tm.connections.synapsesForSegment(seg)
        self.assertLessEqual(len(syns), tm.maxSynapsesPerSegment,
                            "Segment should prune if max exceeded")




if __name__ == "__main__":
    unittest.main()

import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

class TestTMSynapseGrowth(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0
        )

    def test_new_segment_grows_only_on_winner_cell(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)
        cell = list(self.tm.winnerCells)[0]
        segments = self.tm.connections.segmentsForCell(cell)
        self.assertGreater(len(segments), 0)

    def test_new_synapses_point_only_to_prev_winner_cells(self):
        self.tm.compute([0], learn=True)
        prev_winners = set(self.tm.winnerCells)
        self.tm.compute([1], learn=True)

        cell = list(self.tm.winnerCells)[0]
        segment = self.tm.connections.segmentsForCell(cell)[0]
        syns = self.tm.connections.synapsesForSegment(segment)

        for s in syns:
            syn_cell = self.tm.connections.dataForSynapse(s).presynapticCell
            self.assertIn(syn_cell, prev_winners)

    def test_synapse_growth_caps_at_maxNewSynapseCount(self):
        self.tm.maxNewSynapseCount = 2
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        cell = list(self.tm.winnerCells)[0]
        segment = self.tm.connections.segmentsForCell(cell)[0]
        syn_count = len(self.tm.connections.synapsesForSegment(segment))
        self.assertLessEqual(syn_count, 2)


class TestTMSegmentAdaptation(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0
        )

    def test_adapt_segment_increases_active_synapses(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        cell = list(self.tm.winnerCells)[0]
        segment = self.tm.connections.segmentsForCell(cell)[0]
        active_syns = self.tm.connections.synapsesForSegment(segment)
        before = {s: self.tm.connections.dataForSynapse(s).permanence for s in active_syns}

        self.tm.compute([0], learn=True)
        after = {s: self.tm.connections.dataForSynapse(s).permanence for s in active_syns}

        for s in active_syns:
            self.assertGreaterEqual(after[s], before[s])

    def test_adapt_segment_decreases_inactive_synapses(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        cell = list(self.tm.winnerCells)[0]
        segment = self.tm.connections.segmentsForCell(cell)[0]
        synapses = self.tm.connections.synapsesForSegment(segment)

        inactive_syn = synapses[-1]
        initial_perm = self.tm.connections.dataForSynapse(inactive_syn).permanence
        self.tm.connections.updateSynapsePermanence(inactive_syn, initial_perm)

        self.tm.compute([0], learn=True)
        updated_perm = self.tm.connections.dataForSynapse(inactive_syn).permanence
        self.assertLessEqual(updated_perm, initial_perm)

    def test_adapt_segment_does_not_change_structure(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([1], learn=True)

        cell = list(self.tm.winnerCells)[0]
        segment = self.tm.connections.segmentsForCell(cell)[0]
        syn_before = set(self.tm.connections.synapsesForSegment(segment))

        self.tm.compute([0], learn=True)
        syn_after = set(self.tm.connections.synapsesForSegment(segment))
        self.assertEqual(syn_before, syn_after)

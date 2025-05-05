# tests/test_adapt_segment.py

import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import TemporalMemory

class TestAdaptSegment(unittest.TestCase):

    def setUp(self):
        self.conn = Connections(num_cells=10)
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=10,
            activationThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=5,
            permanenceIncrement=0.1,
            permanenceDecrement=0.05,
            predictedSegmentDecrement=0.0
        )
        self.seg = self.conn.createSegment(0)

        # Create 3 synapses with defined permanence
        self.syn1 = self.conn.createSynapse(self.seg, 1, 0.2)
        self.syn2 = self.conn.createSynapse(self.seg, 2, 0.5)
        self.syn3 = self.conn.createSynapse(self.seg, 3, 0.8)

    def test_positive_reinforcement(self):
        self.tm.adaptSegment(self.conn, self.seg, [self.syn1, self.syn2], True,
                             permanenceIncrement=0.1, permanenceDecrement=0.05)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn1).permanence, 0.3)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn2).permanence, 0.6)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn3).permanence, 0.75)

    def test_negative_reinforcement(self):
        self.tm.adaptSegment(self.conn, self.seg, [self.syn2], False,
                             permanenceIncrement=0.1, permanenceDecrement=0.05)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn2).permanence, 0.45)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn1).permanence, 0.2)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn3).permanence, 0.8)

    def test_permanence_clamping(self):
        self.conn.updateSynapsePermanence(self.syn1, 0.99)
        self.conn.updateSynapsePermanence(self.syn3, 0.01)
        self.tm.adaptSegment(self.conn, self.seg, [self.syn1], True,
                             permanenceIncrement=0.1, permanenceDecrement=0.05)
        self.assertEqual(self.conn.dataForSynapse(self.syn1).permanence, 1.0)
        self.assertAlmostEqual(self.conn.dataForSynapse(self.syn3).permanence, 0.0)

    def test_adapt_empty_segment(self):
        empty_seg = self.conn.createSegment(0)
        try:
            self.tm.adaptSegment(self.conn, empty_seg, [], True)
        except Exception as e:
            self.fail(f"adaptSegment raised unexpectedly on empty segment: {e}")

    def test_adapt_with_invalid_synapse_in_active_list(self):
        seg = self.conn.createSegment(0)
        real_syn = self.conn.createSynapse(seg, 1, 0.2)
        with self.assertRaises(KeyError):
            self.tm.adaptSegment(self.conn, seg, [real_syn, 999], True)

    def test_duplicate_synapse_in_active_list(self):
        seg = self.conn.createSegment(0)
        syn = self.conn.createSynapse(seg, 2, 0.5)
        self.tm.adaptSegment(self.conn, seg, [syn, syn, syn], True, 0.1, 0.1)
        self.assertAlmostEqual(self.conn.dataForSynapse(syn).permanence, 0.6)

    def test_reinforcement_with_no_active_synapses(self):
        seg = self.conn.createSegment(0)
        syn = self.conn.createSynapse(seg, 2, 0.9)
        self.tm.adaptSegment(self.conn, seg, [], True, 0.1, 0.2)
        self.assertAlmostEqual(self.conn.dataForSynapse(syn).permanence, 0.7)

    def test_synapse_count_stable_after_adaptation(self):
        seg = self.conn.createSegment(0)
        syns = [self.conn.createSynapse(seg, i + 1, 0.5) for i in range(5)]
        before = set(self.conn.synapsesForSegment(seg))
        self.tm.adaptSegment(self.conn, seg, syns[:2], True)
        after = set(self.conn.synapsesForSegment(seg))
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()

# tests/test_learn_segments.py

import unittest
from unittest.mock import patch
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

class TestLearnSegments(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=[1],
            cellsPerColumn=4,
            activationThreshold=2,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            minThreshold=1,
            maxNewSynapseCount=10,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0
        )
        self.conn = self.tm.connections

    def test_learn_with_existing_segment_adapts(self):
        cell = self.tm._cells_for_column(0)[0]  # → 0
        seg = self.conn.createSegment(cell)

        # Use valid presynaptic cell indices (0–3)
        self.conn.createSynapse(seg, 1, 0.21)
        self.conn.createSynapse(seg, 2, 0.21)
        self.conn.createSynapse(seg, 3, 0.21)

        prevActive = {1, 2, 3}

        # Register predictive state and winner mapping
        self.tm.prevPredictiveCells.add(cell)
        self.tm.segmentActiveForCell[cell] = seg
        self.tm.winnerCellForColumn[0] = cell

        with patch.object(self.tm, 'adaptSegment', wraps=self.tm.adaptSegment) as mock_adapt:
            self.tm._learn_segments([0], prevActive)
            self.assertGreaterEqual(mock_adapt.call_count, 1)

    def test_learn_creates_new_segment_when_no_match(self):
        cell = self.tm._cells_for_column(0)[0]
        self.tm.winnerCellForColumn[0] = cell
        prevActive = {1, 2}

        # No prior prediction or segment
        with patch.object(self.conn, 'createSegment', wraps=self.conn.createSegment) as mock_create:
            self.tm._learn_segments([0], prevActive)
            self.assertEqual(mock_create.call_count, 1)

    def test_learn_grows_synapses_on_new_segment(self):
        cell = self.tm._cells_for_column(0)[0]
        self.tm.winnerCellForColumn[0] = cell
        prevActive = {0, 1, 2}

        segs_before = set(self.conn.segments())

        self.tm._learn_segments([0], prevActive)

        segs_after = set(self.conn.segments())
        new_seg = (segs_after - segs_before).pop()
        syns = self.conn.synapsesForSegment(new_seg)
        presyn_cells = {self.conn.dataForSynapse(s).presynapticCell for s in syns}

        self.assertEqual(presyn_cells, prevActive)

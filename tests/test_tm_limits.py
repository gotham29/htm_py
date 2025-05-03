import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTMLimits(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=3,
            maxSegmentsPerCell=2,
            maxSynapsesPerSegment=5
        )

    def test_max_segments_per_cell_enforced(self):
        col = 0
        for _ in range(5):
            self.tm.compute([col], learn=True)
        cell = list(self.tm.winnerCells)[-1]
        segments = self.tm.connections.segments_for_cell(cell)
        self.assertLessEqual(len(segments), 2)

    def test_max_synapses_per_segment_enforced(self):
        col = 0
        for _ in range(10):
            self.tm.compute([col], learn=True)
        for cell in self.tm.winnerCells:
            segments = self.tm.connections.segments_for_cell(cell)
            for seg in segments:
                self.assertLessEqual(len(seg.synapses), 5)

    def test_max_new_synapse_count_respected(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=1
        )
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)
        for cell in self.tm.winnerCells:
            for seg in self.tm.connections.segments_for_cell(cell):
                self.assertLessEqual(len(seg.synapses), 1)

    def test_max_synapses_per_segment_after_adaptation(self):
        col = 0
        for _ in range(6):
            self.tm.compute([col], learn=True)
        for cell in self.tm.winnerCells:
            for seg in self.tm.connections.segments_for_cell(cell):
                self.assertLessEqual(len(seg.synapses), 5)

    def test_max_segments_enforced_multiple_cells(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=3,
            maxSegmentsPerCell=1
        )
        for _ in range(6):
            self.tm.compute([0], learn=True)
        for cell in range(4):
            segs = self.tm.connections.segments_for_cell(cell)
            self.assertLessEqual(len(segs), 1)

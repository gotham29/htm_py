# tests/test_tm_learning_behavior.py
import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections


class TestSegmentReuseLogic(unittest.TestCase):

    def test_segment_reuse_on_partial_overlap(self):
        """Test that segment is reused (not grown) when overlap ≥ minThreshold even if context not exact."""
        tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=16,  # 🔧 Ensure we can reference cells like 10, 11
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        cell = 0
        tm.winnerCellForColumn[0] = cell

        # Step 1: learn with full context
        context1 = {10, 11}
        tm.prevWinnerCells = context1
        tm._learn_segments([0], context1)
        seg1 = tm.segmentActiveForCell[cell]

        # Step 2: re-learn with partial context (same cell, overlapping)
        context2 = {10}
        tm.prevWinnerCells = context2
        tm._learn_segments([0], context2)
        seg2 = tm.segmentActiveForCell[cell]

        # ✅ Assert same segment reused
        self.assertEqual(seg1, seg2, "Expected reuse of segment with overlap ≥ minThreshold")

        # ✅ Ensure only one segment exists
        segments = list(tm.connections.segmentsForCell(cell))
        self.assertEqual(len(segments), 1, f"Expected 1 segment, found {len(segments)}")


if __name__ == '__main__':
    unittest.main()
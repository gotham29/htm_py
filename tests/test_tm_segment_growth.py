import unittest
from htm_py.temporal_memory import TemporalMemory

class TestSegmentGrowthLogic:
    
    def test_does_not_create_segment_when_predicted(self):
        """
        Ensure that a segment is NOT created on a cell that was correctly predicted.
        """
        tm = TemporalMemory(columnDimensions=(4,), cellsPerColumn=2, activationThreshold=1,
                            initialPermanence=0.3, connectedPermanence=0.2, minThreshold=1,
                            permanenceIncrement=0.1, permanenceDecrement=0.0)

        # Teach TM: column 0 → column 1
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)

        cell = tm.winnerCellForColumn[1]
        seg_count_before = tm.connections.numSegments(cell)

        # Feed same input again (should be predicted now)
        tm.compute([1], learn=True)
        seg_count_after = tm.connections.numSegments(cell)

        assert seg_count_after == seg_count_before, \
            "Should not grow new segment on predicted cell"

    def test_only_one_segment_per_bursting_column(self):
        tm = TemporalMemory(columnDimensions=(1,), cellsPerColumn=4, activationThreshold=1,
                            initialPermanence=0.3, connectedPermanence=0.2, minThreshold=1,
                            permanenceIncrement=0.1, permanenceDecrement=0.0)

        fixed_winner = 0
        tm.winnerCellForColumn[0] = fixed_winner

        # Set up prior state manually
        tm.prevWinnerCells = set()
        tm.prevPredictiveCells = set()

        for _ in range(3):
            tm.prevWinnerCells = {fixed_winner}
            tm.winnerCellForColumn[0] = fixed_winner
            tm.compute([0], learn=True)

        total_segs = tm.connections.numSegments(fixed_winner)
        assert total_segs == 1, f"Expected 1 segment on cell {fixed_winner}, found {total_segs}"

    def test_synapse_growth_only_on_burst(self):
        """
        Check that synapses are only added when a cell actually bursts.
        """
        tm = TemporalMemory(columnDimensions=(2,), cellsPerColumn=2, activationThreshold=1,
                            initialPermanence=0.3, connectedPermanence=0.2, minThreshold=1,
                            permanenceIncrement=0.1, permanenceDecrement=0.0)

        tm.compute([0], learn=True)
        tm.compute([1], learn=True)

        cell = tm.winnerCellForColumn[1]
        segs = tm.connections.segmentsForCell(cell)
        assert len(segs) == 1

        syns = tm.connections.synapsesForSegment(segs[0])
        assert len(syns) > 0, "Expected synapses to be grown on bursting column"

    def test_debug_segment_growth_trace(self):
        """
        DEBUG: Trace exactly when multiple segments are created on a cell across bursts.
        """
        tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            connectedPermanence=0.2,
            minThreshold=1,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0
        )

        seen_segments = set()
        for step in range(5):
            tm.compute([0], learn=True)
            seg_counts = [tm.connections.numSegments(c) for c in range(tm.numCells)]
            total_segments = sum(seg_counts)

            print(f"\nStep {step} — WinnerCell: {tm.winnerCellForColumn[0]}")
            print(f"  Segments per cell: {seg_counts}")
            print(f"  Total segments: {total_segments}")

            for c in range(tm.numCells):
                for seg in tm.connections.segmentsForCell(c):
                    if seg not in seen_segments:
                        print(f"    [+] Segment {seg} created on cell {c}")
                        seen_segments.add(seg)


if __name__ == "__main__":
    unittest.main()

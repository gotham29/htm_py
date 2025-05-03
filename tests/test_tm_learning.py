import unittest
from htm_py.temporal_memory import TemporalMemory

class TestTMLearning(unittest.TestCase):

    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(4,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.21,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=3
        )

    def test_learning_cells_chosen_for_segments(self):
        active_columns = [1, 2]
        
        # Warm-up phase to ensure prevActiveCells is populated
        self.tm.compute(active_columns, learn=False)

        max_attempts = 5
        for attempt in range(max_attempts):
            self.tm.compute(active_columns, learn=True)

            all_have_segments = True
            for col in active_columns:
                winner_cells = self.tm.winner_cells_for_column(col)
                for cell in winner_cells:
                    segs = self.tm.connections.data_for_cell(cell).segments
                    print(f"[ATTEMPT {attempt}] col={col}, winner_cell={cell}, segments={len(segs)}")
                    if len(segs) < 1:
                        all_have_segments = False

            if all_have_segments:
                break

        # Final check: assert that each winner cell has at least 1 segment
        for col in active_columns:
            winner_cells = self.tm.winner_cells_for_column(col)
            for cell in winner_cells:
                segs = self.tm.connections.data_for_cell(cell).segments
                self.assertGreaterEqual(len(segs), 1, f"Learning cell in column {col} (cell {cell}) should have a segment")

    def test_segment_learning_reinforcement(self):
        active_columns = [0]
        self.tm.compute(active_columns, learn=True)
        self.tm.compute(active_columns, learn=True)

        for cell in self.tm.winner_cells_for_column(0):
            segments = self.tm.connections.data_for_cell(cell).segments
            if segments:
                before = [syn.permanence for syn in segments[0].synapses]
                self.tm.compute(active_columns, learn=True)
                after = [syn.permanence for syn in segments[0].synapses]
                self.assertNotEqual(before, after)

    def test_synapse_growth_respects_max_new(self):
        self.tm = TemporalMemory(
            columnDimensions=(1,),
            cellsPerColumn=4,
            activationThreshold=1,
            initialPermanence=0.3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            predictedSegmentDecrement=0.0,
            connectedPermanence=0.5,
            minThreshold=1,
            maxNewSynapseCount=1
        )
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)

        for cell in self.tm.winner_cells_for_column(0):
            segments = self.tm.connections.data_for_cell(cell).segments
            if segments:
                self.assertLessEqual(len(segments[0].synapses), 1)

    def test_segment_created_if_none_exist(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)
        for cell in self.tm.winner_cells_for_column(0):
            segments = self.tm.connections.data_for_cell(cell).segments
            self.assertGreaterEqual(len(segments), 1)

    def test_synapses_created_for_prev_active_cells(self):
        self.tm.compute([0], learn=True)
        self.tm.compute([0], learn=True)
        prev_active = self.tm.prevActiveCells.copy()
        self.tm.compute([0], learn=True)
        found = False
        for cell in self.tm.winner_cells_for_column(0):
            segments = self.tm.connections.data_for_cell(cell).segments
            for seg in segments:
                connected = {self.tm.connections.presynaptic_cell(s) for s in seg.synapses}
                if connected.intersection(prev_active):
                    found = True
        self.assertTrue(found)

    def test_segments_reinforced_on_correct_prediction(self):
        self.tm.compute([0], learn=True)
        winner = list(self.tm.winnerCells)[0]
        seg = self.tm.connections.create_segment(winner)
        for c in self.tm.prevActiveCells:
            self.tm.connections.create_synapse(seg, c, 0.4)
        before = [s.permanence for s in seg.synapses]
        self.tm.compute([0], learn=True)
        after = [s.permanence for s in seg.synapses]
        self.assertNotEqual(before, after)

    def test_no_synapse_growth_without_learning(self):
        self.tm.compute([1], learn=True)
        self.tm.compute([1], learn=False)
        for cell in self.tm.winner_cells_for_column(1):
            segments = self.tm.connections.data_for_cell(cell).segments
            for seg in segments:
                syn_count = len(seg.synapses)
                self.tm.compute([1], learn=False)
                self.assertEqual(syn_count, len(seg.synapses))


if __name__ == '__main__':
    unittest.main()

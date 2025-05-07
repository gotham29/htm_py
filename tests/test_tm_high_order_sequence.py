# tests/test_tm_high_order_sequence.py

import unittest
from htm_py.temporal_memory import TemporalMemory

import unittest
from htm_py.temporal_memory import TemporalMemory


class TestHighOrderSequence(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(6,),  # 6 cols × 4 cells = 24 total cells
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

    def _run_sequence(self, col_sequence, learn=True):
        for col in col_sequence:
            print(f"== Running input col {col} at t={self.tm._timestep}")
            print(f"   prevWinnerCells: {sorted(self.tm.prevWinnerCells)}")
            self.tm.compute([col], learn=learn)

    def test_high_order_branching_sequence(self):
        # Repeat each sequence 3x to allow full learning
        for _ in range(3):
            self._run_sequence([0, 1, 2])  # A → B → C
            self.tm.reset()
        for _ in range(3):
            self._run_sequence([0, 3, 4])  # A → D → E
            self.tm.reset()

        # Test prediction after A → B
        self._run_sequence([0, 1], learn=False)
        pred_cells_after_B = self.tm.predictiveCells
        pred_cols_B = set(cell // self.tm.cellsPerColumn for cell in pred_cells_after_B)
        print(f"[TEST] Predicted columns after A→B: {sorted(pred_cols_B)}")

        self.tm.reset()

        # Test prediction after A → D
        self._run_sequence([0, 3], learn=False)
        pred_cells_after_D = self.tm.predictiveCells
        pred_cols_D = set(cell // self.tm.cellsPerColumn for cell in pred_cells_after_D)
        print(f"[TEST] Predicted columns after A→D: {sorted(pred_cols_D)}")

        # Assert that column 2 (C) is predicted after A→B
        self.assertIn(2, pred_cols_B, f"Expected C (col 2) to be predicted after A→B, got {pred_cols_B}")
        # Assert that column 4 (E) is predicted after A→D
        self.assertIn(4, pred_cols_D, f"Expected E (col 4) to be predicted after A→D, got {pred_cols_D}")

    def test_high_order_AB_vs_XB_discrimination(self):
        # Train A → B → C
        for _ in range(3):
            self._run_sequence([0, 1, 2])
            self.tm.reset()

        # Train X → B → Y
        for _ in range(3):
            self._run_sequence([3, 1, 4])
            self.tm.reset()

        # Recall A → B and check prediction
        self._run_sequence([0, 1], learn=False)
        preds_ab = set(self.tm.predictiveCells)
        pred_cols_ab = {cell // self.tm.cellsPerColumn for cell in preds_ab}
        self.tm.reset()

        # Recall X → B and check prediction
        self._run_sequence([3, 1], learn=False)
        preds_xb = set(self.tm.predictiveCells)
        pred_cols_xb = {cell // self.tm.cellsPerColumn for cell in preds_xb}
        self.tm.reset()

        # C is col 2, Y is col 4
        self.assertIn(2, pred_cols_ab, f"Expected C (col 2) to be predicted after A→B, got {pred_cols_ab}")
        self.assertIn(4, pred_cols_xb, f"Expected Y (col 4) to be predicted after X→B, got {pred_cols_xb}")
        self.assertNotEqual(pred_cols_ab, pred_cols_xb, "Predictions after A→B and X→B should differ")

    def test_high_order_ABC_vs_XBC_discrimination(self):
        # A=0, B=1, C=2, D=3, X=4, Y=5
        self.tm = TemporalMemory(
            columnDimensions=(6,), cellsPerColumn=4,
            activationThreshold=1, minThreshold=1,
            initialPermanence=0.21, connectedPermanence=0.2,
            permanenceIncrement=0.1, permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        def run(seq, learn=True): self._run_sequence(seq, learn=learn)

        # Train A → B → C → D
        for _ in range(3):
            run([0, 1, 2, 3])
            self.tm.reset()

        # Train X → B → C → Y
        for _ in range(3):
            run([4, 1, 2, 5])
            self.tm.reset()

        # Test A → B → C → predict D
        run([0, 1, 2], learn=False)
        preds = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}

        # Test A → B → C → predict D
        run([0, 1, 2], learn=False)
        preds = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}

        # 🔍 Debug unexpected predictions
        print(f"[DEBUG] Predicted columns after A→B→C: {sorted(preds)}")
        print(f"[DEBUG] Predictive cells: {sorted(self.tm.predictiveCells)}")
        for cell in sorted(self.tm.predictiveCells):
            segs = self.tm.connections.segmentsForCell(cell)
            for seg in segs:
                syns = self.tm.connections.synapsesForSegment(seg)
                srcs = [self.tm.connections.dataForSynapse(s).presynapticCell for s in syns]
                print(f"  [cell {cell}] → segment {seg} from presynaptic cells: {sorted(srcs)}")

        self.assertIn(3, preds, f"Expected D (col 3) after A→B→C, got {preds}")
        self.assertNotIn(5, preds, f"Did not expect Y (col 5) after A→B→C")


        self.tm.reset()

        # Test X → B → C → predict Y
        run([4, 1, 2], learn=False)
        preds = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}
        self.assertIn(5, preds, f"Expected Y (col 5) after X→B→C, got {preds}")
        self.assertNotIn(3, preds, f"Did not expect D (col 3) after X→B→C")


    def test_context_specific_segment_reuse(self):
        """Ensure same cell reuses segment only for identical context."""
        tm = TemporalMemory(
            columnDimensions=(4,),  # 4 columns × 4 cells = 16 cells (indices 0–15)
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        # Force cell 0 to win for column 0
        tm.winnerCellForColumn[0] = 0

        # Context A
        context1 = {10, 11}
        tm.prevWinnerCells = context1
        tm._learn_segments([0], context1)
        seg1 = tm.segmentActiveForCell[0]

        # Reuse same context
        tm._learn_segments([0], context1)
        seg2 = tm.segmentActiveForCell[0]
        assert seg1 == seg2, "Segment should be reused for same context"

        # New context
        context2 = {12, 13}
        tm.prevWinnerCells = context2
        tm._learn_segments([0], context2)
        seg3 = tm.segmentActiveForCell[0]
        assert seg3 != seg1, "Different context should yield new segment"

    def test_partial_context_creates_new_segment_when_context_differs(self):
        """Segments should NOT be reused if context differs, even if overlap ≥ minThreshold."""
        tm = TemporalMemory(
            columnDimensions=(1,), cellsPerColumn=16,
            activationThreshold=1, minThreshold=1,
            initialPermanence=0.21, connectedPermanence=0.2,
            permanenceIncrement=0.1, permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        cell = 0
        tm.winnerCellForColumn[0] = cell

        # Step 1: Learn full context
        context1 = {10, 11}
        tm.prevWinnerCells = context1
        tm._learn_segments([0], context1)
        seg1 = tm.segmentActiveForCell[cell]

        # Step 2: New context overlaps but not identical
        context2 = {10}
        tm.prevWinnerCells = context2
        tm._learn_segments([0], context2)
        seg2 = tm.segmentActiveForCell[cell]

        # ✅ Expect a new segment (context mismatch → no reuse)
        assert seg1 != seg2, "Expected a new segment when context only partially overlaps"

    def test_ABC_only_predicts_D(self):
        """After training on A→B→C→D and X→B→C→Y, ensure A→B→C only predicts D (col 3)."""
        self.tm = TemporalMemory(
            columnDimensions=(6,), cellsPerColumn=4,
            activationThreshold=1, minThreshold=1,
            initialPermanence=0.21, connectedPermanence=0.2,
            permanenceIncrement=0.1, permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        def run(seq, learn=True): self._run_sequence(seq, learn=learn)

        # Train A → B → C → D
        for _ in range(3):
            run([0, 1, 2, 3])
            self.tm.reset()

        # Train X → B → C → Y
        for _ in range(3):
            run([4, 1, 2, 5])
            self.tm.reset()

        # Test A → B → C
        run([0, 1, 2], learn=False)
        predicted_cols = {cell // self.tm.cellsPerColumn for cell in self.tm.predictiveCells}
        print(f"[ASSERT] After A→B→C, predicted columns: {predicted_cols}")

        # ✅ Only D (col 3) should be predicted
        self.assertEqual(predicted_cols, {3}, f"Only D (col 3) should be predicted, got {predicted_cols}")

    def test_segments_on_C_after_branches(self):
        """Inspect segments on column C (2) after training A→B→C→D and X→B→C→Y.
        Ensures separate segments form on different winner cells for C depending on context.
        """
        self.tm = TemporalMemory(
            columnDimensions=(6,), cellsPerColumn=4,
            activationThreshold=1, minThreshold=1,
            initialPermanence=0.21, connectedPermanence=0.2,
            permanenceIncrement=0.1, permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        def run(seq, learn=True):
            self._run_sequence(seq, learn=learn)

        # Train both branches
        for _ in range(3):
            run([0, 1, 2, 3])  # A→B→C→D
            self.tm.reset()
            run([4, 1, 2, 5])  # X→B→C→Y
            self.tm.reset()

        # Get cells for column C (index 2)
        colC_cells = self.tm._cells_for_column(2)

        # Print segments and contexts on each cell
        found = []
        for cell in colC_cells:
            segments = self.tm.connections.segmentsForCell(cell)
            if not segments:
                continue
            for seg in segments:
                syns = self.tm.connections.synapsesForSegment(seg)
                srcs = [self.tm.connections.dataForSynapse(s).presynapticCell for s in syns]
                found.append((cell, seg, sorted(srcs)))
                print(f"[C SEGMENTS] Cell {cell}, Segment {seg}, Context {sorted(srcs)}")

        # Assert we have at least 2 segments on distinct cells (branching)
        unique_cells = {cell for cell, _, _ in found}
        self.assertGreaterEqual(len(unique_cells), 2, "Expected at least 2 unique cells in column C due to branching")

        # Check that at least one segment includes presynaptic cell from A-context, one from X-context
        A_context = set(self.tm._cells_for_column(0))  # Column A
        X_context = set(self.tm._cells_for_column(4))  # Column X

        has_A_branch = any(any(src in A_context for src in ctx) for _, _, ctx in found)
        has_X_branch = any(any(src in X_context for src in ctx) for _, _, ctx in found)

        self.assertTrue(has_A_branch, "Expected one segment on C from A-context")
        self.assertTrue(has_X_branch, "Expected one segment on C from X-context")


if __name__ == '__main__':
    unittest.main()

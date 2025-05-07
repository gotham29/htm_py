# tests/test_tm_high_order_branching.py
import unittest
from htm_py.temporal_memory import TemporalMemory

import unittest
from htm_py.temporal_memory import TemporalMemory


class TestBranchingSegments(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(5,),
            cellsPerColumn=4,
            activationThreshold=1,
            minThreshold=1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            maxNewSynapseCount=5
        )
        self.cols = {"A": 0, "B": 2, "C": 3, "X": 1}
        self.cell_B = 2 * self.tm.cellsPerColumn  # first cell in column B

    def feed_sequence(self, labels):
        for label in labels:
            self.tm.compute([self.cols[label]], learn=True)

    def test_segment_growth_on_branching(self):
        # First high-order context: A → B → C
        self.feed_sequence(["A", "B", "C"])
        self.tm.reset()

        # Second high-order context: X → B → C
        self.feed_sequence(["X", "B", "C"])
        self.tm.reset()

        # Count the number of segments grown on the first cell of column B
        num_segments = self.tm.connections.numSegments(self.cell_B)
        print(f"[TEST] Number of segments on cell_B after branching: {num_segments}")
        self.assertGreaterEqual(
            num_segments, 2,
            "Expected multiple segments on branching cell B due to high-order context divergence"
        )

if __name__ == "__main__":
    unittest.main()

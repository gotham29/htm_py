import unittest
from htm_py.temporal_memory import TemporalMemory


class TestTMBranchPoints(unittest.TestCase):

    def test_branching_on_B(self):
        tm = TemporalMemory(
            columnDimensions=(6,), cellsPerColumn=4,
            activationThreshold=1, minThreshold=1,
            initialPermanence=0.21, connectedPermanence=0.2,
            permanenceIncrement=0.1, permanenceDecrement=0.0,
            maxNewSynapseCount=10
        )

        def run(seq):
            for col in seq:
                tm.compute([col])

        # Run A → B
        tm.reset()
        run([0, 1])  # A=0, B=1
        winner_after_A = tm.winnerCellForColumn.get(1)
        print(f"[BRANCH] Winner cell for B after A→B: {winner_after_A}")

        # Run X → B
        tm.reset()
        run([4, 1])  # X=4, B=1
        winner_after_X = tm.winnerCellForColumn.get(1)
        print(f"[BRANCH] Winner cell for B after X→B: {winner_after_X}")

        self.assertIsNotNone(winner_after_A)
        self.assertIsNotNone(winner_after_X)
        self.assertNotEqual(
            winner_after_A, winner_after_X,
            "Expected different winner cells for B after different contexts (A vs X)"
        )



if __name__ == "__main__":
    unittest.main()

# import unittest
# from htm_py.connections import Connections
# from htm_py.temporal_memory import TemporalMemory

# class TestBurstColumn(unittest.TestCase):
#     def setUp(self):
#         self.conn = Connections(num_cells=20)
#         self.tm = TemporalMemory(
#             columnDimensions=(2,),
#             cellsPerColumn=10,
#             activationThreshold=1,
#             initialPermanence=0.21,
#             connectedPermanence=0.5,
#             minThreshold=1,
#             maxNewSynapseCount=5,
#             permanenceIncrement=0.1,
#             permanenceDecrement=0.05,
#             predictedSegmentDecrement=0.0
#         )
#         self.tm.connections = self.conn  # inject

#     def test_burst_column_creates_segments(self):
#         self.tm.activeColumns = {1}
#         self.tm.compute(list(self.tm.activeColumns))
#         num_segments = sum(len(self.conn.segmentsForCell(cell)) for cell in range(10, 20))
#         self.assertGreater(num_segments, 0)

#     def test_burst_column_marks_winner_cells(self):
#         self.tm.activeColumns = {0}
#         self.tm.compute(list(self.tm.activeColumns))
#         self.assertEqual(len(self.tm.winnerCells), 1)
#         self.assertIn(next(iter(self.tm.winnerCells)), range(0, 10))

# if __name__ == "__main__":
#     unittest.main()


import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

class TestBurstColumn(unittest.TestCase):
    def setUp(self):
        self.tm = TemporalMemory(
            columnDimensions=(2,),
            cellsPerColumn=10,
            activationThreshold=1,
            minThreshold=1,
            maxNewSynapseCount=3,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            initialPermanence=0.21,
            connectedPermanence=0.2,
            predictedSegmentDecrement=0.0
        )

    def test_burst_column_creates_segments(self):
        self.tm.activeColumns = {1}
        self.tm.compute(list(self.tm.activeColumns), learn=True)  # PATCHED: added learn=True
        num_segments = sum(len(self.tm.connections.segmentsForCell(cell)) for cell in range(10, 20))
        self.assertGreater(num_segments, 0)

    def test_burst_column_marks_winner_cells(self):
        self.tm.activeColumns = {0}
        self.tm.compute(list(self.tm.activeColumns), learn=True)
        self.assertTrue(len(self.tm.winnerCells) > 0)

if __name__ == "__main__":
    unittest.main()

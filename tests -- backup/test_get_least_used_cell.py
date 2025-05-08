# tests/test_get_least_used_cell.py

import unittest
from htm_py.connections import Connections
from htm_py.temporal_memory import getLeastUsedCell

class TestGetLeastUsedCell(unittest.TestCase):

    def setUp(self):
        self.conn = Connections(num_cells=10)
        self.column = [2, 3, 4]

    def test_returns_cell_with_fewest_segments(self):
        self.conn.createSegment(2)
        self.conn.createSegment(2)
        self.conn.createSegment(3)
        result = getLeastUsedCell(self.conn, self.column)
        self.assertEqual(result, 4)  # has 0 segments

    def test_tie_breaker_returns_first(self):
        self.conn.createSegment(3)
        self.conn.createSegment(4)
        result = getLeastUsedCell(self.conn, self.column)
        self.assertEqual(result, 2)  # all equal (0), 2 is first

    def test_single_cell_column(self):
        result = getLeastUsedCell(self.conn, [5])
        self.assertEqual(result, 5)

    def test_empty_column_raises(self):
        with self.assertRaises(ValueError):
            getLeastUsedCell(self.conn, [])

    def test_non_integer_cell_raises(self):
        with self.assertRaises(ValueError):
            getLeastUsedCell(self.conn, [2, "3"])

    def test_out_of_bounds_cell_raises(self):
        with self.assertRaises(ValueError):
            getLeastUsedCell(self.conn, [2, 11])


if __name__ == "__main__":
    unittest.main()

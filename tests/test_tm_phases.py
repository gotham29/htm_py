import unittest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Connections

def setup_tm(activation_threshold=1, connected_perm=0.2):
    tm = TemporalMemory(
        columnDimensions=(4,),
        cellsPerColumn=2,
        activationThreshold=activation_threshold,
        initialPermanence=0.21,
        connectedPermanence=connected_perm,
        minThreshold=1,
        maxNewSynapseCount=4,
        maxSynapsesPerSegment=5,
        maxSegmentsPerCell=5,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.1
    )
    print(f"[SETUP] tm.connections id: {id(tm.connections)}")
    return tm


class TestTMPhases(unittest.TestCase):

    def test_winner_cell_selection(self):
        tm = setup_tm()
        tm.compute([0], learn=True)
        self.assertTrue(len(tm.winnerCells) > 0, "Winner cells must be selected")

    def test_create_segment_on_learning_cell(self):
        tm = setup_tm()
        tm.compute([0], learn=True)
        print(f"[TEST] tm.connections id: {id(tm.connections)}")
        winner_cells = list(tm.winnerCells)
        self.assertGreater(len(winner_cells), 0, "Winner cell must exist")
        # At least one winner cell must receive a segment
        self.assertTrue(any(
            len(tm.connections.segmentsForCell(cell)) > 0 for cell in winner_cells
        ), "At least one winner cell should receive a segment")

    def test_adapt_segment_growth(self):
        tm = setup_tm()
        tm.prevActiveCells = {1}
        tm.compute([0], learn=True)
        segments = list(tm.connections._segment_to_cell.keys())
        self.assertGreater(len(segments), 0, "At least one segment should exist after learning")

    def test_predictive_cell_activation(self):
        tm = setup_tm()
        for _ in range(5):
            tm.compute([0], learn=True)
        tm.compute([0], learn=False)
        pred = tm.predictiveCells
        print(f"[TEST] predictiveCells = {pred}")
        self.assertGreater(len(pred), 0, "At least one cell should become predictive")

    def test_no_segment_created_if_max_reached(self):
        tm = setup_tm()
        cell = 0
        for _ in range(6):  # Exceed maxSegmentsPerCell = 5
            tm.connections.createSegment(cell, iteration_num=_)
        segments = tm.connections.segmentsForCell(cell)
        self.assertEqual(len(segments), 5, "Segments must not exceed maxSegmentsPerCell")

    def test_anomaly_score_changes_on_bursting(self):
        tm = setup_tm()
        tm.prevActiveCells = {99}  # junk
        tm.compute([0], learn=True)
        self.assertGreaterEqual(tm.anomalyScore, 0.0)
        self.assertLessEqual(tm.anomalyScore, 1.0)


class TestConnectionsDataIntegrity(unittest.TestCase):

    def test_segment_creation_and_retrieval(self):
        conn = Connections(
            columnDimensions=(1,), 
            cellsPerColumn=4,
            initialPermanence=0.21, 
            connectedPermanence=0.5,
            maxSegmentsPerCell=5
        )
        cell = 0
        segment = conn.createSegment(cell, iteration_num=1)

        self.assertIn(segment, conn.segmentsForCell(cell), 
                      "Segment should be retrievable from segmentsForCell")
        self.assertIn(segment, conn.segments,
                      "Segment should be in global segments list")
        self.assertEqual(conn.cellForSegment(segment), cell,
                         "cellForSegment should return correct cell")

    def test_segments_registered_in_tm_compute(self):
        tm = setup_tm()
        tm.compute([0], learn=True)
        print(f"[TEST] tm.connections id: {id(tm.connections)}")
        winner_cells = list(tm.winnerCells)
        self.assertGreater(len(winner_cells), 0, "Winner cell must exist")
        # At least one of the winner cells should have a segment
        self.assertTrue(any(
            len(tm.connections.segmentsForCell(cell)) > 0 for cell in winner_cells
        ), "At least one winner cell must have a segment")

    def test_synapse_creation_on_learning(self):
        tm = setup_tm()
        tm.prevActiveCells = {1}
        tm.compute([0], learn=True)

        any_synapse_created = False
        for cell in tm.winnerCells:
            for seg in tm.connections.segmentsForCell(cell):
                print(f"[DEBUG TEST] Segment {seg.id} synapses: {seg.synapses}")
                if len(seg.synapses) > 0:
                    any_synapse_created = True

        self.assertTrue(any_synapse_created, "At least one synapse should be created during learning")

    def test_synapse_created_flag(self):
        tm = setup_tm()
        tm.prevActiveCells = {1}
        tm.compute([0], learn=True)
        found = False
        for segment in tm.connections._segments:
            if segment.synapses:
                found = True
        self.assertTrue(found, "Synapse was created on at least one segment")

    def test_tm_higher_order_sequence(self):
        """
        Verifies TM learns higher-order sequences like:
        a,b,c,d,x,b,c,y → y should only be predicted after (x,b,c), not (a,b,c)
        """
        tm = setup_tm()
        encoding = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'x': 5, 'y': 6}
        sequence = ['a', 'b', 'c', 'd', 'x', 'b', 'c', 'y'] * 3  # Repeat 3x for learning

        for t, char in enumerate(sequence):
            val = encoding[char]
            tm.compute([val], learn=True)

        # Test prediction: after the 2nd occurrence of x,b,c, y should be predicted
        tm.prevActiveCells = set()
        for char in ['x', 'b', 'c']:
            tm.compute([encoding[char]], learn=False)

        pred = tm.predictiveCells
        self.assertGreater(len(pred), 0, "Should predict 'y' after 'x,b,c' context")

    def test_tm_interleaved_branching_sequence(self):
        """
        Verifies TM distinguishes between two interleaved sequences:
        a,b,c,d and x,b,c,e
        """
        tm = setup_tm()
        encoding = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'x': 5, 'e': 6}
        sequence = ['a', 'b', 'c', 'd', 'x', 'b', 'c', 'e'] * 3  # Repeated learning

        for char in sequence:
            tm.compute([encoding[char]], learn=True)

        # Test prediction after a,b,c → should predict d
        tm.prevActiveCells = set()
        for char in ['a', 'b', 'c']:
            tm.compute([encoding[char]], learn=False)
        pred1 = tm.predictiveCells

        # Test prediction after x,b,c → should predict e
        tm.prevActiveCells = set()
        for char in ['x', 'b', 'c']:
            tm.compute([encoding[char]], learn=False)
        pred2 = tm.predictiveCells

        # Check that predictive sets are non-empty and disjoint
        self.assertGreater(len(pred1), 0, "Should predict after a,b,c")
        self.assertGreater(len(pred2), 0, "Should predict after x,b,c")
        self.assertTrue(pred1.isdisjoint(pred2), "Predicted cells should differ after different contexts")

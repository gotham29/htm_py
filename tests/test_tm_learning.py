import pytest
from htm_py.temporal_memory import TemporalMemory
from htm_py.connections import Segment


@pytest.fixture
def tm():
    return TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=1,
        activationThreshold=1,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=1,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=1,
        maxSynapsesPerSegment=5,
        maxNewSynapseCount=3,
    )


def test_adapt_segment_behavior(tm):
    con = tm.connections
    seg = con.createSegment(0, iteration_num=0)
    con.createSynapse(seg, 1, 0.4)
    con.createSynapse(seg, 2, 0.4)
    con.createSynapse(seg, 3, 0.4)

    con.adaptSegment(
        seg,
        prev_active_dense=[False, True, True, False],  # instead of {1, 2}
        permanenceInc=0.1,
        permanenceDec=0.1
    )

    perms = {s.presynaptic_cell: s.permanence for s in con.synapsesForSegment(seg)}
    assert abs(perms[1] - 0.5) < 1e-6
    assert abs(perms[2] - 0.5) < 1e-6
    assert abs(perms[3] - 0.3) < 1e-6


def test_segment_growth_on_bursting_column(tm):
    tm.prevActiveCells = {0, 1}
    tm.compute(activeColumns=[0], learn=True, iteration=1)
    segments = tm.connections.segmentsForCell(0)
    assert len(segments) == 1


def test_segment_eviction_when_max_segments_reached(tm):
    tm.prevActiveCells = {0, 1}
    cell = 0
    seg0 = tm.connections.createSegment(cell, iteration_num=0)
    assert seg0 in tm.connections.segmentsForCell(cell)

    tm.compute(activeColumns=[0], learn=True, iteration=1)
    segments = tm.connections.segmentsForCell(cell)

    assert len(segments) == 1
    assert seg0 not in segments


def test_get_best_matching_segment_logic(tm):
    con = tm.connections
    cell = 0
    seg = con.createSegment(cell, iteration_num=0)
    con.createSynapse(seg, 1, 0.6)
    con.createSynapse(seg, 2, 0.4)

    dense = [False] * 10
    dense[1] = True
    dense[2] = True

    best = con.getBestMatchingSegment(cell, dense, min_threshold=1)
    assert best is not None
    assert best.id == seg.id


def test_segment_not_created_if_limit_disabled():
    tm = TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=1,
        activationThreshold=1,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=1,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=0,  # disabled
        maxSynapsesPerSegment=5,
        maxNewSynapseCount=3,
    )
    tm.prevActiveCells = {0, 1}
    tm.compute(activeColumns=[0], learn=True, iteration=1)
    segments = tm.connections.segmentsForCell(0)
    assert len(segments) == 0


def test_best_matching_cell_selection():
    tm = TemporalMemory(
        columnDimensions=(1,),
        cellsPerColumn=2,
        activationThreshold=1,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=1,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=5,
        maxSynapsesPerSegment=5,
        maxNewSynapseCount=3,
    )
    tm.prevActiveCells = {0}
    dense = tm._to_dense(tm.prevActiveCells)

    seg = tm.connections.createSegment(1, iteration_num=0)
    tm.connections.createSynapse(seg, 0, 0.6)

    best = tm._best_matching_cell(0, dense)
    assert best == 1


def test_no_learning_mode_does_not_create_segment(tm):
    tm.prevActiveCells = {0, 1}
    tm.compute(activeColumns=[0], learn=False, iteration=1)
    segments = tm.connections.segmentsForCell(0)
    assert len(segments) == 0

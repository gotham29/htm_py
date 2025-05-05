import pytest
from htm_py.temporal_memory import TemporalMemory

@pytest.fixture
def tm_config():
    return {
        'columnDimensions': (1,),
        'cellsPerColumn': 1,
        'activationThreshold': 1,
        'initialPermanence': 0.21,
        'connectedPermanence': 0.5,
        'minThreshold': 1,
        'permanenceIncrement': 0.1,
        'permanenceDecrement': 0.1,
        'predictedSegmentDecrement': 0.0,
        'maxSegmentsPerCell': 5,
        'maxSynapsesPerSegment': 5,
        'maxNewSynapseCount': 2,
    }

def test_no_synapse_growth_when_presynaptic_empty(tm_config):
    tm = TemporalMemory(**tm_config)
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm._grow_synapses_to_segment(seg, set())
    syns = tm.connections.synapsesForSegment(seg)
    assert len(syns) == 0

def test_synapse_growth_respects_maxNewSynapseCount(tm_config):
    tm = TemporalMemory(**tm_config)
    tm.prevActiveCells = {0, 1, 2, 3, 4}
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm._grow_synapses_to_segment(seg, tm.prevActiveCells)
    syns = tm.connections.synapsesForSegment(seg)
    assert len(syns) == tm.maxNewSynapseCount

def test_synapse_growth_does_not_duplicate_existing(tm_config):
    tm = TemporalMemory(**tm_config)
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm.connections.createSynapse(seg, 1, 0.5)
    tm.prevActiveCells = {1, 2}
    tm._grow_synapses_to_segment(seg, tm.prevActiveCells)
    syns = tm.connections.synapsesForSegment(seg)
    presyn_ids = [s.presynaptic_cell for s in syns]
    assert presyn_ids.count(1) == 1

def test_segment_adaptation_increments_and_decrements(tm_config):
    tm = TemporalMemory(**tm_config)
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm.connections.createSynapse(seg, 0, 0.4)  # active
    tm.connections.createSynapse(seg, 1, 0.6)  # inactive
    prev_dense = [True, False]
    tm.connections.adaptSegment(seg, prev_dense, tm.permanenceIncrement, tm.permanenceDecrement)
    syns = tm.connections.synapsesForSegment(seg)
    assert any(s.presynaptic_cell == 0 and s.permanence > 0.4 for s in syns)
    assert any(s.presynaptic_cell == 1 and s.permanence < 0.6 for s in syns)

def test_learning_reinforces_existing_segment_if_threshold_met(tm_config):
    tm = TemporalMemory(**tm_config)
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm.connections.createSynapse(seg, 0, 0.5)
    tm.prevActiveCells = {0}
    tm.compute([0], learn=True, iteration=1)
    assert seg in tm.connections.segmentsForCell(0)

def test_no_segment_growth_when_minThreshold_not_met(tm_config):
    tm = TemporalMemory(**tm_config)
    tm.minThreshold = 2  # override
    seg = tm.connections.createSegment(0, iteration_num=0)
    tm.connections.createSynapse(seg, 0, 0.5)
    tm.prevActiveCells = {0}
    tm.compute([0], learn=True, iteration=1)
    segments = tm.connections.segmentsForCell(0)
    assert len(segments) == 1  # no new segment created

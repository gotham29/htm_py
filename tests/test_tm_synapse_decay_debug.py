# File: tests/test_tm_synapse_decay_debug.py

import pytest
from htm_py.temporal_memory import TemporalMemory

def test_synapse_decay_and_pruning():
    tm = TemporalMemory(
        columnDimensions=(10,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.3,
        connectedPermanence=0.2,
        minThreshold=1,
        maxNewSynapseCount=4,
        permanenceIncrement=0.1,
        permanenceDecrement=0.05,
        predictedSegmentDecrement=0.0
    )

    # Step 1: Learn [0] → [1]
    tm.compute([0], learn=True)
    tm.compute([1], learn=True)

    # Extract segment + synapses on column 1
    cell = tm._cells_for_column(1)[0]
    segs = tm.connections.segmentsForCell(cell)
    assert segs, "No segment created on column 1"
    segment = segs[0]
    syns = tm.connections.synapsesForSegment(segment)
    assert syns, "No synapses grown"
    original_perms = {s: tm.connections.dataForSynapse(s).permanence for s in syns}

    # Step 2: Decay - simulate disuse
    for _ in range(10):
        tm.compute([2], learn=True)
        tm.prevPredictiveCells = set()  # No correct prediction
        tm.segmentActiveForCell.pop(cell, None)  # Avoid reuse of segment

    print("\n[Decay Test] Synapse permanence values after repeated learning:")
    for s in syns:
        perm = tm.connections.dataForSynapse(s).permanence
        print(f"  Synapse {s}: permanence = {perm:.3f} (was {original_perms[s]:.3f})")

    # Assert decay occurred
    assert any(tm.connections.dataForSynapse(s).permanence < original_perms[s] for s in syns), \
        "No synapse permanence decayed after repeated disuse"


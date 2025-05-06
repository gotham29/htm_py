# File: tests/test_tm_synapse_decay_debug.py

import unittest
from htm_py.temporal_memory import TemporalMemory

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

    cell = tm._cells_for_column(1)[0]
    segs = tm.connections.segmentsForCell(cell)
    assert segs, "No segment created on column 1"
    segment = segs[0]
    syns = tm.connections.synapsesForSegment(segment)
    assert syns, "No synapses grown"
    original_perms = {s: tm.connections.dataForSynapse(s).permanence for s in syns}

    # Get connected synapses and their presynaptic cells
    connected_syn_data = [
        (s, tm.connections.dataForSynapse(s).presynapticCell)
        for s in syns
        if tm.connections.dataForSynapse(s).permanence >= tm.connectedPermanence
    ]
    assert connected_syn_data, "Segment should have at least one connected synapse"

    # Step 2: Reuse segment with active synapses but one forced inactive
    for _ in range(10):
        # Deactivate exactly one connected synapse
        inactive_syn, inactive_cell = connected_syn_data[0]
        active_cells = {cell for _, cell in connected_syn_data if cell != inactive_cell}

        tm.activeCells = active_cells
        tm.prevActiveCells = active_cells

        tm.segmentActiveForCell[cell] = segment
        tm.winnerCellForColumn[1] = cell
        tm.prevWinnerCells = {tm._cells_for_column(9)[0]}  # wrong winner to prevent growth

        tm._adapt_segment(
            connections=tm.connections,
            segment=segment,
            activePresynapticCells=tm.prevActiveCells,
            newSynapseCount=0,
            increment=0.0,
            decrement=tm.permanenceDecrement
        )

    print("\n[Decay Test] Synapse permanence values after repeated learning:")
    for s in syns:
        perm = tm.connections.dataForSynapse(s).permanence
        print(f"  Synapse {s}: permanence = {perm:.3f} (was {original_perms[s]:.3f})")

    # Assert decay occurred
    assert any(tm.connections.dataForSynapse(s).permanence < original_perms[s] for s in syns), \
        "No synapse permanence decayed after repeated disuse"


if __name__ == "__main__":
    unittest.main()



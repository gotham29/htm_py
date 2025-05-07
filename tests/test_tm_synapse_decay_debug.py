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

    tm.compute([0], learn=True)
    tm.compute([1], learn=True)

    cell = tm._cells_for_column(1)[0]
    segs = tm.connections.segmentsForCell(cell)
    assert segs, "No segment created on column 1"
    segment = segs[0]
    syns = tm.connections.synapsesForSegment(segment)
    assert syns, "No synapses grown"

    for _ in range(10):
        connected_syn_data = [
            (s, tm.connections.dataForSynapse(s).presynapticCell)
            for s in syns
            if s in tm.connections._synapse_data and
               tm.connections.dataForSynapse(s).permanence >= tm.connectedPermanence
        ]
        if not connected_syn_data:
            break

        inactive_syn, inactive_cell = connected_syn_data[0]
        active_cells = {c for _, c in connected_syn_data if c != inactive_cell}

        tm.activeCells = active_cells
        tm.prevActiveCells = active_cells

        tm.segmentActiveForCell[cell] = segment
        tm.winnerCellForColumn[1] = cell
        tm.prevWinnerCells = {tm._cells_for_column(9)[0]}

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
        if s in tm.connections._synapse_data:
            perm = tm.connections.dataForSynapse(s).permanence
            print(f"Syn {s} -> {perm}")
        else:
            print(f"Syn {s} was pruned")

if __name__ == "__main__":
    unittest.main()



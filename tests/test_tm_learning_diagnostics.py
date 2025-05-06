import pytest
from htm_py.temporal_memory import TemporalMemory

def test_segment_and_synapse_growth_debug():
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

    pattern = [[0], [1]] * 3

    print("\n=== Training Phase ===")
    for i, input_col in enumerate(pattern):
        print(f"\n--- Step {i+1}: compute({input_col}) ---")
        tm.compute(input_col, learn=True)

    print("\n=== Segment/Synapse Structure After Training ===")
    for col in [0, 1]:
        col_cells = tm._cells_for_column(col)
        print(f"\nColumn {col}:")
        for c in col_cells:
            segments = tm.connections.segmentsForCell(c)
            print(f"  Cell {c} has {len(segments)} segment(s)")
            for seg in segments:
                syns = tm.connections.synapsesForSegment(seg)
                print(f"    Segment {seg} has {len(syns)} synapse(s):")
                for s in syns:
                    syn_data = tm.connections.dataForSynapse(s)
                    print(f"      Synapse {s}: presyn={syn_data.presynapticCell}, perm={syn_data.permanence:.3f}")

    print("\n=== Inference Test ===")
    tm.compute([0], learn=False)
    print(f"Predictive cells after compute([0]): {sorted(tm.predictiveCells)}")

    # Assert that a predictive cell exists in column 1
    col1_cells = set(tm._cells_for_column(1))
    predicted_in_col1 = col1_cells & tm.predictiveCells
    print(f"Column 1 predicted cells: {predicted_in_col1}")
    assert predicted_in_col1, "No cell in column 1 predicted after [0]"

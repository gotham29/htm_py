# File: tests/test_tm_prediction_thresholds.py

from htm_py.temporal_memory import TemporalMemory

def test_prediction_threshold_and_overlap():
    tm = TemporalMemory(
        columnDimensions=(10,),
        cellsPerColumn=4,
        activationThreshold=1,
        initialPermanence=0.3,
        connectedPermanence=0.2,
        minThreshold=1,
        permanenceIncrement=0.1,
        permanenceDecrement=0.05,
        predictedSegmentDecrement=0.0,
        maxNewSynapseCount=4
    )

    # Train a simple sequence [0] → [1] multiple times
    for _ in range(5):
        tm.compute([0], learn=True)
        tm.compute([1], learn=True)

    # Now compute([0]) in inference mode
    tm.compute([0], learn=False)

    print("\n[Predictive Threshold Test]")
    print(f"Winner cells: {sorted(tm.winnerCells)}")
    print(f"Predictive cells: {sorted(tm.predictiveCells)}")

    # Must have a predictive cell in column 1
    col1_cells = set(tm._cells_for_column(1))
    predicted = tm.predictiveCells & col1_cells
    print(f"Predicted column 1 cells: {predicted}")

    assert predicted, "No column 1 cell became predictive after [0]"

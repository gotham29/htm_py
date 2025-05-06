import numpy as np
from typing import Union, Set, List
from htm_py.connections import Connections
from htm_py.temporal_memory import TemporalMemory
import pytest


# Diagnostic Test: Check if winner cells rotate after bursting multiple times
def test_winner_cell_rotation():
    tm = TemporalMemory(
        columnDimensions=(4,),
        cellsPerColumn=4,
        activationThreshold=1,
        connectedPermanence=0.2,
        initialPermanence=0.3,
        permanenceIncrement=0.1,
        permanenceDecrement=0.05,
        minThreshold=1,
        maxNewSynapseCount=4,
    )

    col = [0]
    winner_history = []

    for i in range(5):
        tm.compute(col, learn=True)
        winner = tm.winnerCellForColumn.get(0)
        print(f"Step {i+1}: Winner = {winner}")
        winner_history.append(winner)

    print(f"Winner cell history: {winner_history}")
    assert len(set(winner_history)) > 1, "Winner cell did not change across timesteps"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

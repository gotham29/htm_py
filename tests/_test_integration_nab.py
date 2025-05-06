import pytest
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder
import datetime
import csv
import numpy as np
import pandas as pd


def load_nab_art_daily_jumpsup():
    with open("data/NAB_art_daily_jumpsup.csv", 'r') as f:
        reader = csv.DictReader(f)
        timestamps, values = [], []
        for row in reader:
            timestamps.append(datetime.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"))
            values.append(float(row["value"]))
    return timestamps, values

def test_integration_pipeline_alignment_to_nab():
    timestamps, values = load_nab_art_daily_jumpsup()

    rdse = RDSE(min_val=0, max_val=144.4, n=150, w=21)
    date_enc = DateEncoder(timeOfDay=(21, 9.49))
    encoder = MultiEncoder({
        "timestamp": date_enc,
        "value": rdse,
    })

    model = HTMModel(encoder)

    scores = []
    for t, v in zip(timestamps, values):
        encoded = encoder.encode({"timestamp": t, "value": v})
        score, _ = model.run(encoded, learn=True)
        scores.append(score)

    # Load Numenta’s expected anomaly scores if available
    expected_scores = pd.read_csv("results/NAB_art_daily_jumpsup_numentaTM.csv")

    assert len(scores) == len(expected_scores)
    mae = np.mean(np.abs(np.array(scores) - expected_scores))
    assert mae < 0.05, f"Mean Absolute Error too high: {mae}"


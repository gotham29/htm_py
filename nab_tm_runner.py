# nab_tm_runner.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("htm_debug.log", mode='w'),
        logging.StreamHandler()
    ]
)


import pandas as pd
import csv

from htm_py.htm_model import HTMModel

def load_nab_file(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def write_output(filepath, timestamps, scores, counts):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "anomaly_score", "prediction_count"])
        writer.writerows(zip(timestamps, scores, counts))

def main():
    fname = "art_daily_jumpsup"
    input_csv = f"data/{fname}.csv"
    output_csv = f"htmpy_{fname}.csv"
    df = load_nab_file(input_csv)

    # Provide only configs
    enc_params = {
        "rdse": {"size": 563, "resolution": 0.88, "seed": 42},
        "date": {"timeOfDay": (21, 9.49), "weekend": 1}
    }
    sp_params = {
        "inputDimensions": (563,),         # Match encoder output size
        "columnDimensions": (2048,),
        "potentialPct": 0.85,
        "globalInhibition": True,
        "numActiveColumnsPerInhArea": 40,
        "synPermActiveInc": 0.03,
        "synPermInactiveDec": 0.008,
        "synPermConnected": 0.1,
        "boostStrength": 0.0,
        "seed": 42
    }
    tm_params = {
        "columnDimensions": (563,),
        "cellsPerColumn": 32,
        "activationThreshold": 13, #from 7
        "initialPermanence": 0.21,
        "connectedPermanence": 0.50,
        "minThreshold": 10, #from 5
        "maxNewSynapseCount": 20,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.1,
        "predictedSegmentDecrement": 0.0,
        "maxSegmentsPerCell": 255,
        "maxSynapsesPerSegment": 255,
        "seed": 42
    }

    model = HTMModel(enc_params, sp_params, tm_params)

    limit = 200
    timestamps, scores, counts = [], [], []
    for _, row in df[:limit].iterrows():
        input_dict = {"value": row["value"], "timestamp": row["timestamp"]}
        anomaly_score, prediction_count = model.compute(input_dict, learn=True, iteration=_)
        timestamps.append(row["timestamp"])
        scores.append(anomaly_score)
        counts.append(prediction_count)
        if _%1 == 0:
            print(f"time={_}; ascore={round(anomaly_score,3)}; pcount={round(prediction_count,3)}")

    write_output(output_csv, timestamps, scores, counts)
    print(f"✓ Saved results to {output_csv}")


if __name__ == "__main__":
    main()

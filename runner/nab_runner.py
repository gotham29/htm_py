import os
import csv
import numpy as np
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder
import datetime

def load_dataset(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        timestamps = []
        values = []
        for row in reader:
            timestamps.append(datetime.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"))
            values.append(float(row["value"]))
    return timestamps, values

def main():
    timestamps, values = load_dataset("data/NAB_art_daily_jumpsup.csv")

    rdse = RDSE(min_val=0, max_val=114.4, resolution=0.88, n=130, w=21)
    date_enc = DateEncoder(timeOfDay=(21, 9.49))
    encoder = MultiEncoder({
        "timestamp": date_enc,
        "value": rdse,
    })

    model = HTMModel(encoder)

    output_path = "../results/NAB_art_daily_jumpsup_HTMPY.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "value", "anomaly_score", "prediction_count"])

        for t, v in zip(timestamps, values):
            encoded = encoder.encode({"timestamp": t, "value": v})
            anomaly_score, prediction_count = model.run(encoded, learn=True)
            print(f"time={t} value={v:.2f} anomaly_score={anomaly_score:.3f} pred_count={prediction_count:.3f}")
            writer.writerow([t, v, anomaly_score, prediction_count])

if __name__ == "__main__":
    main()

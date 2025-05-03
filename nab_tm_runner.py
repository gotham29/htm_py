# import os
# import csv
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from htm_py.encoders.date import DateEncoder
# from htm_py.encoders.rdse import RDSE
# from htm_py.encoders.combine import combine_encodings
# from htm_py.temporal_memory import TemporalMemory
# from htm_py.connections import Connections

# # Load the NAB dataset
# limit = 100
# path_in = os.path.join("data", "art_daily_jumpsup.csv")
# path_out_csv = os.path.join("results", "htmpy_art_daily_jumpsup.csv")
# path_out_png = os.path.join("results", "htmpy_art_daily_jumpsup.png")
# df = pd.read_csv(path_in)
# values = df["value"].values
# timestamps = pd.to_datetime(df["timestamp"])

# # Numenta-style encoders
# date_encoder = DateEncoder(timeOfDay=(21, 9.49), weekend=1)
# rdse = RDSE(resolution=0.88, seed=1960)  # resolution from NAB's art_daily_jumpsup config

# # Determine encoding size
# date_encoder_size = date_encoder.size
# rdse_size = rdse.size
# input_size = date_encoder_size + rdse_size

# # Initialize TM with Numenta parameters
# connections = Connections(
#     num_cells=2048 * 32,
#     maxSegmentsPerCell=128,
#     maxSynapsesPerSegment=128
# )

# tm = TemporalMemory(
#     columnDimensions=(2048,),
#     cellsPerColumn=32,
#     activationThreshold=20,
#     initialPermanence=0.24,
#     connectedPermanence=0.5,
#     minThreshold=13,
#     maxNewSynapseCount=31,
#     permanenceIncrement=0.04,
#     permanenceDecrement=0.008,
#     predictedSegmentDecrement=0.001,
#     seed=1960,
#     maxSegmentsPerCell=128,
#     maxSynapsesPerSegment=128
# )

# # Prepare inputs and scores
# records = []
# anomaly_scores = []
# num_predicteds = []
# for i in range(len(df)):
#     if i > limit:
#         continue
#     ts = timestamps[i]
#     val = values[i]

#     enc_date = date_encoder.encode(ts)
#     enc_val = rdse.encode(val)
#     input_vector = combine_encodings(enc_date, enc_val)

#     # Binarize inputs as expected by TM (indexes of nonzero elements)
#     active_columns = np.where(input_vector > 0)[0].tolist()

#     tm.compute(active_columns, learn=True)

#     # Anomaly score: fraction of active columns not predicted
#     num_predicted = len(set(tm.get_predictive_cells()) & set(tm.get_active_cells()))
#     anomaly_score = 1.0 - (num_predicted / len(tm.get_active_cells()) if tm.get_active_cells() else 0)
#     anomaly_scores.append(anomaly_score)
#     num_predicteds.append(num_predicted)
#     print(f"t={i}; ascore={round(anomaly_score,3)}; pcount={round(num_predicted,3)}")
    
#     records.append([ts, val, round(anomaly_score, 6), round(num_predicted, 6)])


# # Write to output CSV
# with open(path_out_csv, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["timestamp", "value", "anomaly_score", "num_predicted"])
#     writer.writerows(records)


# # Plot
# plt.figure(figsize=(12, 4))
# # plt.plot(df["timestamp"][:limit], values[:limit], label="value")
# plt.plot(df["timestamp"][:limit], anomaly_scores[:limit], label="anomaly score")
# plt.legend()
# plt.title("HTM Anomaly Score")
# plt.savefig(path_out_png)
# plt.show()


# # import os
# # import csv
# # from htm_py.temporal_memory import TemporalMemory
# # from htm_py.connections import Connections
# # from htm_py.encoders.rdse_encoder import RDSE
# # from htm_py.encoders.date_encoder import DateEncoder
# # import numpy as np
# # import pandas as pd

# # # Load dataset
# # df = pd.read_csv(os.path.join("data", "art_daily_jumpsup.csv"))

# # df["timestamp"] = pd.to_datetime(df["timestamp"])
# # values = df["value"].values
# # timestamps = df["timestamp"].values

# # # Encoders
# # value_encoder = RDSE(n=100, w=21, minval=0.0, maxval=100.0)
# # date_encoder = DateEncoder(timeOfDay=(21, 9.49), weekend=1)

# # input_size = value_encoder.n + date_encoder.n

# # # Model setup
# # cells_per_column = 32
# # column_dims = (input_size,)
# # num_cells = input_size * cells_per_column

# # connections = Connections(
# #     num_cells=num_cells,
# #     maxSegmentsPerCell=255,
# #     maxSynapsesPerSegment=32
# # )

# # tm = TemporalMemory(
# #     columnDimensions=column_dims,
# #     cellsPerColumn=cells_per_column,
# #     activationThreshold=13,
# #     initialPermanence=0.21,
# #     connectedPermanence=0.50,
# #     minThreshold=10,
# #     maxNewSynapseCount=20,
# #     permanenceIncrement=0.10,
# #     permanenceDecrement=0.10,
# #     predictedSegmentDecrement=0.0,
# #     seed=42,
# #     maxSegmentsPerCell=255,
# #     maxSynapsesPerSegment=32
# # )

# # # Output tracking
# # records = []

# # # Run HTM over input
# # tm.reset()
# # for i in range(len(values)):
# #     value = values[i]
# #     timestamp = timestamps[i]

# #     val_bits = value_encoder.encode(value)
# #     date_bits = date_encoder.encode(timestamp)

# #     input_bits = val_bits + date_bits
# #     active_columns = [i for i, b in enumerate(input_bits) if b == 1]

# #     tm.compute(active_columns, learn=True)
# #     num_predicted = sum(1 for c in tm.get_active_cells() if c in tm.get_predictive_cells())
# #     anomaly_score = 1.0 - (num_predicted / float(len(tm.get_active_cells()) or 1))

# #     records.append([timestamp, value, round(anomaly_score, 6)])

# # # Write to output CSV
# # with open("htmpy_art_daily_jumpsup.csv", "w", newline="") as f:
# #     writer = csv.writer(f)
# #     writer.writerow(["timestamp", "value", "anomaly_score"])
# #     writer.writerows(records)



import pandas as pd
import numpy as np
import time
import argparse
import pickle
from pathlib import Path

from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.htm_model import HTMModel


def combine_encodings(*encodings):
    return np.concatenate(encodings).astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help="CSV file from the /data directory")
    parser.add_argument('--limit', type=int, default=None, help="Number of rows to process")
    parser.add_argument('--verbose', action='store_true', help="Print per-row debug output")
    args = parser.parse_args()

    # Load dataset
    data_path = Path("data") / args.file
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    if args.limit:
        df = df.iloc[:args.limit]

    # Set up encoders
    date_encoder = DateEncoder(timeOfDay=(21, 9.49), weekend=1)
    rdse = RDSE(resolution=0.88, seed=42, size=2048)

    # Setup model
    enc_params = {
        "rdse": {
            "size": 2048,
            "resolution": 0.88,
            "seed": 42
        },
        "date": {
            "timeOfDay": (21, 9.49),  # 21 active bits, 24 hours/day → π width
            "weekend": 1
        }
    }

    tm_params = {
        "columnDimensions": (2048,),
        "cellsPerColumn": 32,
        "activationThreshold": 13,
        "initialPermanence": 0.21,
        "connectedPermanence": 0.50,
        "minThreshold": 10,
        "maxNewSynapseCount": 20,
        "permanenceIncrement": 0.10,
        "permanenceDecrement": 0.10,
        "predictedSegmentDecrement": 0.0,
        "maxSegmentsPerCell": 255,
        "maxSynapsesPerSegment": 255,
        "seed": 42
    }
    # tm_params = dict(
    #     activationThreshold=20,
    #     cellsPerColumn=32,
    #     columnDimensions=(2048,),
    #     initialPermanence=0.24,
    #     connectedPermanence=0.5,
    #     maxSegmentsPerCell=128,
    #     maxSynapsesPerSegment=128,
    #     minThreshold=13,
    #     maxNewSynapseCount=31,
    #     permanenceDecrement=0.008,
    #     permanenceIncrement=0.04,
    #     predictedSegmentDecrement=0.001,
    #     seed=1960,
    # )

    model = HTMModel(enc_params=enc_params, tm_params=tm_params)

    results = []
    for i, row in df.iterrows():
        ts = row["timestamp"]
        val = row["value"]

        enc_date = date_encoder.encode(ts)
        enc_val = rdse.encode(val)
        input_vector = combine_encodings(enc_date, enc_val)

        start_time = time.time()
        anomaly_score, pred_count = model.compute(input_vector, learn=True)
        latency = time.time() - start_time

        results.append({
            "timestamp": ts,
            "value": val,
            "anomaly_score": anomaly_score,
            "prediction_count": pred_count,
            "processing_time": latency
        })

        if args.verbose:
            print(f"{i:04d} | score: {anomaly_score:.4f}, pred: {pred_count}, latency: {latency*1000:.2f} ms")

    results_df = pd.DataFrame(results)
    out_file = f"results/htmpy_{args.file}"
    results_df.to_csv(out_file, index=False)
    print(f"\n✅ Results written to {out_file}")

    # Save TM state
    with open("results/htmpy_tm_connections.pkl", "wb") as f:
        pickle.dump(model.tm.connections, f)
        print("🧠 TM connections saved to results/htmpy_tm_connections.pkl")


if __name__ == "__main__":
    main()

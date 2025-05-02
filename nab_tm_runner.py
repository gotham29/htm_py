import argparse
import pandas as pd
import matplotlib.pyplot as plt
from htm_py.htm_model import HTMModel

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="Run HTM on NAB dataset")
parser.add_argument('-l', '--limit', type=int, default=None,
                    help='Limit number of timesteps to process')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable verbose debugging output')
args = parser.parse_args()
limit = args.limit
verbose = args.verbose

# --- Load data ---
df = pd.read_csv("data/art_daily_jumpsup.csv")
if limit:
    df = df.iloc[:limit]

# --- Initialize HTM model ---
model = HTMModel(
    enc_params={
        "resolution": 0.88,
        "n": 2048,
        "w": 40,
        "minval": df["value"].min(),
        "maxval": df["value"].max(),
        "seed": 42
    },
    tm_params = {
        "columnDimensions": (2048,),
        "cellsPerColumn": 32,
        "activationThreshold": 12,
        "initialPermanence": 0.21,
        "connectedPermanence": 0.5,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.1,
        "minThreshold": 8,
        "maxNewSynapseCount": 20
    },
    seed=42,
)

# --- Run HTM over data ---
ascores = []
pcounts = []

for i, row in df.iterrows():
    record = row.to_dict()
    outputs = model.compute(record, learn=True, verbose=verbose)
    ascores.append(outputs["anomaly_score"])
    pcounts.append(outputs["prediction_count"])

    if i % 100 == 0:
        print(f"time={i} anomaly_score={outputs['anomaly_score']:.3f} pred_count={outputs['prediction_count']:.3f}")

# --- Plot results ---
plt.figure(figsize=(12, 6))
plt.plot(ascores, label="HTM Anomaly Score")
plt.title("NAB: art_daily_jumpsup (HTM Anomaly Detection)")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
plt.savefig("ascores.png")

plt.figure(figsize=(12, 6))
plt.plot(pcounts, label="HTM Prediction Count")
plt.title("NAB: art_daily_jumpsup (HTM Anomaly Detection)")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Score")
plt.legend()
plt.tight_layout()
plt.savefig("pcounts.png")

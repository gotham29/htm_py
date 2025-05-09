import pandas as pd
from datetime import datetime, timedelta
from htm_model import HTMModel
import matplotlib.pyplot as plt

# === Create Dummy Config ===
config = {
    "encoder": {
        "rdse_features": [{"name": "value", "min_val": 0, "max_val": 100, "n": 100, "w": 21}],
        "timeOfDay": {"n": 21, "rotation": 9.49}
    },
    "use_sp": False,
    "tm": {
        "column_dimensions": [32],
        "cells_per_column": 4,
        "activation_threshold": 3,
        "initial_permanence": 0.21,
        "connected_permanence": 0.5,
        "min_threshold": 2,
        "max_new_synapse_count": 5,
        "permanence_increment": 0.1,
        "permanence_decrement": 0.1,
        "predicted_segment_decrement": 0.0,
        "seed": 42,
        "max_segments_per_cell": 5,
        "max_synapses_per_segment": 20,
        "check_inputs": True
    }
}

# === Create Dummy Data (Simulate a pattern with a known anomaly) ===
timestamps = [datetime.now() + timedelta(seconds=i) for i in range(200)]
values = [10] * 50 + [50] * 10 + [10] * 140  # Spike between steps 50–60 to simulate anomaly

df = pd.DataFrame({"timestamp": timestamps, "value": values})

# === Initialize Model ===
model = HTMModel(config)

# === Run Model and Collect Results ===
results = []
for i, row in df.iterrows():
    input_row = {"value": row["value"], "timestamp": row["timestamp"]}
    anomaly_score, prediction_count = model.compute(input_row, learn=True)
    results.append((i, anomaly_score, prediction_count))

# === Convert Results to DataFrame ===
results_df = pd.DataFrame(results, columns=["step", "anomaly_score", "prediction_count"])

# === Plot Results ===
plt.figure(figsize=(12, 6))
plt.plot(results_df["step"], results_df["anomaly_score"], label="Anomaly Score", linewidth=2)
plt.plot(results_df["step"], results_df["prediction_count"], label="Prediction Count", linewidth=2, linestyle='--')
plt.axvline(50, color="red", linestyle=":", label="Anomaly Injected")
plt.xlabel("Timestep")
plt.ylabel("Score")
plt.title("HTMModel Validation - Anomaly Score & Prediction Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Summary ===
print(results_df.head(10))
print("✅ Validation complete. Check the plot for anomaly detection behavior.")

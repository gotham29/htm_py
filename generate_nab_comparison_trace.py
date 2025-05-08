import pandas as pd
import yaml
from datetime import datetime
from htm_py.htm_model import HTMModel
import matplotlib.pyplot as plt

limit = 200

# === Load reference scores ===
numenta_df = pd.read_csv("results/NAB_art_daily_jumpsup_NumentaTM.csv")
ref_scores = numenta_df["raw_score"].tolist()  #anomaly_score

# === Load dataset and config ===
df = pd.read_csv("data/NAB_art_daily_jumpsup.csv")[:limit]
with open("config/NAB_tm.yaml", "r") as f:
    config = yaml.safe_load(f)

model = HTMModel(config)

# === Compute your model scores ===
rows = []
for i, row in df.iterrows():
    if i >= len(ref_scores): break

    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
    input_row = {"value": row["value"], "timestamp": ts}
    result = model.compute(input_row, learn=True)

    active = set(result["active_cells"])
    pred = set(result["predictive_cells"])
    overlap = len(active & pred)
    match_fraction = overlap / len(active) if active else 0.0

    if i < 10:
        print(f"[t={i}] norm_pred_count={result['normalized_prediction_count']:.2f}")

    rows.append({
        "step": i,
        "timestamp": row["timestamp"],
        "value": row["value"],
        "numenta_score": ref_scores[i],
        "your_score": result["anomaly_score"],
        "abs_diff": abs(result["anomaly_score"] - ref_scores[i]),
        "prediction_count": result["prediction_count"],
        "normalized_prediction_count": result["normalized_prediction_count"],
        "segment_count": sum(len(model.tm.connections.segmentsForCell(c)) for c in range(model.tm.numCells)),
        "active_overlap": overlap,
        "match_fraction": match_fraction,
    })

# === Save trace as CSV ===
trace_df = pd.DataFrame(rows)
trace_df.to_csv("nab_comparison_trace.csv", index=False)
print("✅ Wrote trace to nab_comparison_trace.csv")

# === Plot side-by-side ===
plt.figure(figsize=(12, 5))
plt.plot(trace_df["step"], trace_df["numenta_score"], label="Numenta", linewidth=2)
plt.plot(trace_df["step"], trace_df["your_score"], label="Your HTM", linewidth=2, linestyle='--')
plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.title("NAB Score Comparison — art_daily_jumpsup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

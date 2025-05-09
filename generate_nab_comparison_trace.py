import pandas as pd
import yaml
from datetime import datetime
from htm_py.htm_model import HTMModel
import matplotlib.pyplot as plt

limit = 1000
dataset = "art_daily_jumpsup"

# === Load reference scores ===
numenta_df = pd.read_csv(f"results/NAB_{dataset}_NumentaTM.csv")
ref_scores = numenta_df["raw_score"].tolist()  #anomaly_score

# === Load dataset and config ===
df = pd.read_csv(f"data/NAB_{dataset}.csv")[:limit]
with open(f"config/NAB_{dataset}.yaml", "r") as f:
    config = yaml.safe_load(f)

model = HTMModel(config)

# === Compute your model scores ===
rows = []
for i, row in df.iterrows():
    if i >= len(ref_scores): break

    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
    input_row = {"value": row["value"], "timestamp": ts}
    anomaly_score, pred_count = model.compute(input_row, learn=True)

    print(f"time={i} htm_py: {round(anomaly_score,3)}; numenta: {round(ref_scores[i],3)}")

    rows.append({
        "step": i,
        "timestamp": row["timestamp"],
        "value": row["value"],
        "numenta_score": ref_scores[i],
        "your_score": anomaly_score,
        "abs_diff": abs(anomaly_score - ref_scores[i]),
        "prediction_count": pred_count,
        "segment_count": sum(len(model.tm.connections.segmentsForCell(c)) for c in range(model.tm.numCells)),
        # "normalized_prediction_count": result["normalized_prediction_count"],
        # "active_overlap": overlap,
        # "match_fraction": match_fraction,
    })

# === Save trace as CSV ===
trace_df = pd.DataFrame(rows)
trace_df.to_csv(f"results/NAB_comparison_trace--{dataset}.csv", index=False)
print(f"✅ Wrote trace to NAB_comparison_trace--{dataset}.csv")

# === Plot side-by-side ===
plt.figure(figsize=(12, 5))
plt.plot(trace_df["step"], trace_df["numenta_score"], label="Numenta", linewidth=2)
plt.plot(trace_df["step"], trace_df["your_score"], label="Your HTM", linewidth=2, linestyle='--')
plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.title(f"NAB Score Comparison — {dataset}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

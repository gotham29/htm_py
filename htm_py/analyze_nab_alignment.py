import pandas as pd
import matplotlib.pyplot as plt

# Load your debug trace
debug_trace = pd.read_csv("results/nab_alignment_debug_trace.csv")

# Load Numenta's official NAB results
numenta_trace = pd.read_csv("results/NAB_art_daily_jumpsup_NumentaTM.csv")

plt.figure(figsize=(14, 6))

# Plot Anomaly Scores
plt.plot(debug_trace["timestep"], debug_trace["anomaly_score"], label="Your HTM Anomaly Score", linewidth=2)
plt.plot(numenta_trace["timestamp"][:len(debug_trace)], numenta_trace["raw_score"][:len(debug_trace)], 
         label="Numenta Anomaly Score", linewidth=2, linestyle="--")

plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.title("NAB Anomaly Score Alignment Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Inspect Prediction Count vs Predictive Cell Ambiguity
plt.figure(figsize=(14, 4))
plt.plot(debug_trace["timestep"], debug_trace["prediction_count"], label="Prediction Count", color="orange", linestyle="--")
plt.xlabel("Timestep")
plt.ylabel("Prediction Count")
plt.title("Prediction Ambiguity (Prediction Count)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(debug_trace["timestep"], debug_trace["num_predictive_cells"], 
         label="Num Predictive Cells", color="purple")
plt.xlabel("Timestep")
plt.ylabel("Predictive Cells")
plt.title("Predictive Cells Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

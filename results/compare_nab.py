import os
import pandas as pd
import matplotlib.pyplot as plt

# Load results
path_in_py = os.path.join("results", "htmpy_art_daily_jumpsup.csv")
path_in_numenta = os.path.join("results", "numentaTM_art_daily_jumpsup.csv")

htmpy_df = pd.read_csv(path_in_py)
numenta_df = pd.read_csv(path_in_numenta)

# Truncate to same length
min_len = min(len(htmpy_df), len(numenta_df))
htmpy_df = htmpy_df.iloc[:min_len]
numenta_df = numenta_df.iloc[:min_len]

# Correlation
correlation = htmpy_df["anomaly_score"].corr(numenta_df["raw_score"]) #since anomaly_score called raw_score and likelihood call anomaly_score

# Plot
plt.figure(figsize=(10, 4))
plt.plot(htmpy_df["anomaly_score"], label="HTMPy", alpha=0.7)
plt.plot(numenta_df["anomaly_score"], label="Numenta", alpha=0.7)
plt.title(f"Anomaly Score Comparison - Pearson r = {correlation:.4f}")
plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

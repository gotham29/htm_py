import os
import pandas as pd
import matplotlib.pyplot as plt

# Load results
path_htmpy = os.path.join("results", "NAB_art_daily_jumpsup_HTMPY.csv")
path_numenta = os.path.join("results", "NAB_art_daily_jumpsup_NumentaTM.csv")

htmpy_df = pd.read_csv(path_htmpy)
numenta_df = pd.read_csv(path_numenta)

# Truncate to same length
min_len = min(len(htmpy_df), len(numenta_df))
htmpy_df = htmpy_df.iloc[:min_len]
numenta_df = numenta_df.iloc[:min_len]

# Correlation (compare our anomaly_score to Numenta's raw_score)
correlation = htmpy_df["anomaly_score"].corr(numenta_df["raw_score"])

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(htmpy_df["anomaly_score"], label="HTMPy Anomaly Score", alpha=0.8)
plt.plot(numenta_df["raw_score"], label="Numenta Raw Score", alpha=0.8)
plt.title(f"Anomaly Score Comparison\nPearson r = {correlation:.4f}")
plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# === Paths to your result CSVs ===
htmpy_csv = "htmpy_art_daily_jumpsup.csv"
numenta_csv = "numentaTM_art_daily_jumpsup.csv"

# === Load them ===
htmpy = pd.read_csv(htmpy_csv)
numenta = pd.read_csv(numenta_csv)

# === Align based on timestamp (or index if no timestamp) ===
if 'timestamp' in htmpy.columns and 'timestamp' in numenta.columns:
    merged = pd.merge(htmpy, numenta, on='timestamp', suffixes=('_htmpy', '_numenta'))
else:
    merged = pd.concat([htmpy, numenta], axis=1)
    merged.columns = ['anomalyScore_htmpy', 'anomalyScore_numenta']

# === Compute Error ===
mae = mean_absolute_error(merged['anomalyScore_numenta'], merged['anomalyScore_htmpy'])

print(f"\nMean Absolute Error (HTM.py vs Numenta): {mae:.5f}\n")

# === Plot ===
plt.figure(figsize=(15, 6))
plt.plot(merged['anomalyScore_numenta'], label="Numenta", linewidth=2)
plt.plot(merged['anomalyScore_htmpy'], label="HTM.py", linewidth=2, linestyle='--')
plt.legend()
plt.title("Anomaly Scores: Numenta vs HTM.py")
plt.xlabel("Timestep")
plt.ylabel("Anomaly Score")
plt.grid(True)
plt.show()

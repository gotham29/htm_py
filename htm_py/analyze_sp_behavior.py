import pandas as pd
import matplotlib.pyplot as plt

# Load SP active columns trace
sp_trace = pd.read_csv("results/sp_active_columns_trace.csv")

plt.figure(figsize=(12, 5))
plt.plot(sp_trace["timestep"], sp_trace["num_active_columns"], marker="o", linestyle="-", color="blue")
plt.axhline(y=40, color="red", linestyle="--", label="Expected Active Columns (40)")
plt.xlabel("Timestep")
plt.ylabel("Active Columns")
plt.title("SP Active Columns per Timestep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

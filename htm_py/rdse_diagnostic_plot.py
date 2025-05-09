import numpy as np
import matplotlib.pyplot as plt
from encoders.rdse import RDSE

# === Configurable RDSE Parameters ===
min_val = 0
max_val = 100
resolution = 0.88  # Target resolution
w = 21

# === Initialize RDSE Using Resolution ===
rdse = RDSE(min_val=min_val, max_val=max_val, resolution=resolution, w=w)

# Sweep test values across min to max
test_values = np.linspace(min_val, max_val, 200)
sdr_activations = []

for v in test_values:
    sdr = rdse.encode(v)
    sdr_activations.append(sdr)

sdr_activations = np.array(sdr_activations)

# === Plot SDR Activation Heatmap ===
plt.figure(figsize=(14, 6))
plt.imshow(sdr_activations.T, aspect='auto', cmap='Greys', interpolation='nearest')
plt.colorbar(label="Activation (1 = Active Bit)")
plt.xlabel("Input Value Sweep")
plt.ylabel("SDR Bit Index")
plt.title(f"RDSE Activation Map (Resolution: {resolution}, w: {w})")
plt.show()

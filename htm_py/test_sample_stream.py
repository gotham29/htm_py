
import sys
import pandas as pd
from .utils import load_config

def main(config_path, data_path):
    config = load_config(config_path)
    df = pd.read_csv(data_path)

    print("Streaming data...")
    for timestep, row in df.iterrows():
        result = config.compute(row)
        print(f"Timestep {timestep}: {result}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

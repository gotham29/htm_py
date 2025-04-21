import csv
import sys
import os
from htm_py.utils import load_config
from htm_py.htm_model import HTMModel

def main(config_path, csv_path):
    # Load YAML config
    config = load_config(config_path)

    # Initialize HTMModel
    htm_model = HTMModel(config)

    # Get feature list from config
    features = list(config["features"].keys())

    # Open and stream the CSV
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                input_data = {key: float(row[key]) for key in features}
            except ValueError:
                print(f"[Warning] Row {i} has invalid data: {row}")
                continue

            output = htm_model.compute(input_data)
            print(f"Step {i}: {output}")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python runner.py path/to/config.yaml path/to/data.csv")
        sys.exit(1)

    config_file = sys.argv[1]
    csv_file = sys.argv[2]

    main(config_file, csv_file)

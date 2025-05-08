# nab_runner.py

import pandas as pd
from htm_py.htm_model import HTMModel
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_nab_csv(path):
    df = pd.read_csv(path)
    return df["value"].tolist()

def run_htm_on_nab(csv_path, config_path):
    config = load_config(config_path)
    model = HTMModel(config)

    values = load_nab_csv(csv_path)
    results = []

    for i, v in enumerate(values):
        output = model.compute(v, learn=True)
        results.append({
            "time": i,
            "value": v,
            **output
        })

    return pd.DataFrame(results)

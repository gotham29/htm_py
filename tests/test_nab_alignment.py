import unittest
import pandas as pd
import yaml
from datetime import datetime
from htm_py.htm_model import HTMModel
import csv
import os

TRACE_FILE = "nab_trace.csv"

class TestNABAlignment(unittest.TestCase):

    def load_numenta_scores(self, path):
        df = pd.read_csv(path)
        return df["anomaly_score"].tolist()

    def run_model(self, csv_path, config_path, start_i, end_i):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model = HTMModel(config)
        df = pd.read_csv(csv_path)

        # ✅ Set up CSV logging (overwrite each run)
        with open(TRACE_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestep", "value", "anomaly_score"])

            scores = []
            for i, row in df[:end_i].iterrows():
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                input_row = {
                    "value": row['value'],
                    "timestamp": timestamp
                }
                result = model.compute(input_row, learn=True)
                if i >= start_i:
                    score = result["anomaly_score"]
                    scores.append(score)
                    # ✅ Log to CSV
                    writer.writerow([i, input_row['value'], score])

        return scores

    def test_alignment_to_numenta(self):
        start_i = 0
        end_i = 100
        ref_scores = self.load_numenta_scores("results/NAB_art_daily_jumpsup_NumentaTM.csv")[start_i:end_i]
        my_scores = self.run_model("data/NAB_art_daily_jumpsup.csv", "config/NAB_tm.yaml", start_i, end_i)

        mae = sum(abs(r - m) for r, m in zip(ref_scores, my_scores)) / len(ref_scores)
        print(f"MAE vs Numenta: {mae:.5f}")
        self.assertLess(mae, 0.05, f"MAE too high: {mae:.5f}")


if __name__ == "__main__":
    unittest.main()

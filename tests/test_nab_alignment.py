# import unittest
# import pandas as pd
# import yaml
# from datetime import datetime
# from htm_py.htm_model import HTMModel
# import csv
# import os

# TRACE_FILE = "nab_trace.csv"

# class TestNABAlignment(unittest.TestCase):

#     def load_numenta_scores(self, path):
#         df = pd.read_csv(path)
#         return df["anomaly_score"].tolist()

#     def run_model(self, csv_path, config_path, start_i, end_i):
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)

#         model = HTMModel(config)
#         df = pd.read_csv(csv_path)

#         # ✅ Set up CSV logging (overwrite each run)
#         with open(TRACE_FILE, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["timestep", "value", "anomaly_score"])

#             scores = []
#             for i, row in df[:end_i].iterrows():
#                 timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
#                 input_row = {
#                     "value": row['value'],
#                     "timestamp": timestamp
#                 }
#                 result = model.compute(input_row, learn=True)
#                 if i >= start_i:
#                     score = result["anomaly_score"]
#                     scores.append(score)
#                     # ✅ Log to CSV
#                     writer.writerow([i, input_row['value'], score])

#         return scores

#     def test_alignment_to_numenta(self):
#         start_i = 0
#         end_i = 100
#         ref_scores = self.load_numenta_scores("results/NAB_art_daily_jumpsup_NumentaTM.csv")[start_i:end_i]
#         my_scores = self.run_model("data/NAB_art_daily_jumpsup.csv", "config/NAB_tm.yaml", start_i, end_i)

#         mae = sum(abs(r - m) for r, m in zip(ref_scores, my_scores)) / len(ref_scores)
#         print(f"MAE vs Numenta: {mae:.5f}")
#         self.assertLess(mae, 0.05, f"MAE too high: {mae:.5f}")


# if __name__ == "__main__":
#     unittest.main()


import yaml
import unittest
import numpy as np
import pandas as pd
from htm_py.htm_model import HTMModel


class TestNabAlignment(unittest.TestCase):

    def test_nab_alignment_mae_and_prediction(self):
        # Set limit, paths
        limit = 100
        path_nab = "data/NAB_art_daily_jumpsup.csv"
        path_numenta = "results/NAB_art_daily_jumpsup_NumentaTM.csv"
        path_config = "config/NAB_art_daily_jumpsup.yaml"
        with open(path_config, "r") as file:
            config = yaml.safe_load(file)

        # Load NAB dataset
        df = pd.read_csv(path_nab)[:limit]
        timestamps, values = df['timestamp'].to_list(), df['value'].to_list()

        # Load Numenta 
        df_numenta = pd.read_csv(path_numenta)[:limit]
        numenta_scores = df_numenta["anomaly_score"].tolist()

        # Init Model
        model = HTMModel(config=config)  # or pass decoded YAML dict

        # Run Model
        ours_scores = []
        ours_preds = []
        for t, (ts, val) in enumerate(zip(timestamps, values)):
            row = {"timestamp": ts, "value": val}
            anomaly_score, pred_count = model.compute(row, learn=True)
            ours_scores.append(anomaly_score)
            ours_preds.append(pred_count)

        ours_scores = np.array(ours_scores)
        ours_preds = np.array(ours_preds)

        # Compare anomaly scores to Numenta
        mae_score = np.mean(np.abs(ours_scores - numenta_scores))
        mae_pred = np.mean(np.abs(ours_preds[:-1] - ours_preds[1:]))  # delta predictions

        print(f"MAE (Anomaly Score): {mae_score:.5f}")
        print(f"MAE (Prediction Delta): {mae_pred:.5f}")
        print(f"Pearson Corr (Scores): {np.corrcoef(ours_scores, numenta_scores)[0, 1]:.3f}")

        assert mae_score < 0.03, "Anomaly score MAE too high"
        assert mae_pred < 1.0, "Prediction fluctuation too high"

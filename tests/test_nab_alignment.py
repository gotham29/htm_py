import yaml
import unittest
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from htm_py.htm_model import HTMModel

class TestNabAlignment(unittest.TestCase):

    def test_nab_alignment_mae_and_prediction(self):
        limit = 100
        path_nab = "data/NAB_art_daily_jumpsup.csv"
        path_numenta = "results/NAB_art_daily_jumpsup_NumentaTM.csv"
        path_config = "config/NAB_art_daily_jumpsup.yaml"
        path_trace = f"results/NAB_alignment_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(path_config, "r") as file:
            config = yaml.safe_load(file)

        df = pd.read_csv(path_nab)[:limit]
        timestamps = df["timestamp"].tolist()
        values = df["value"].tolist()

        df_numenta = pd.read_csv(path_numenta)[:limit]
        numenta_scores = df_numenta["raw_score"].tolist()

        model = HTMModel(config=config)
        ours_scores = []
        ours_preds = []

        with open(path_trace, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestep", "value", "numenta_score", "our_score", "abs_error", "pred_count"])

            for t, (ts, val) in enumerate(zip(timestamps, values)):
                input_row = {"timestamp": ts, "value": val}
                anomaly_score, pred_count = model.compute(input_row, learn=True)
                ours_scores.append(anomaly_score)
                ours_preds.append(pred_count)

                abs_err = abs(anomaly_score - numenta_scores[t])
                writer.writerow([t, val, numenta_scores[t], anomaly_score, abs_err, pred_count])

        ours_scores = np.array(ours_scores)
        ours_preds = np.array(ours_preds)

        mae_score = np.mean(np.abs(ours_scores - numenta_scores))
        mae_pred = np.mean(np.abs(ours_preds[:-1] - ours_preds[1:]))
        max_error = np.max(np.abs(ours_scores - numenta_scores))
        corr = np.corrcoef(ours_scores, numenta_scores)[0, 1]

        print(f"MAE (Anomaly Score): {mae_score:.5f}")
        print(f"Max Error: {max_error:.5f}")
        print(f"MAE (Prediction Delta): {mae_pred:.5f}")
        print(f"Pearson Corr (Scores): {corr:.3f}")
        print(f"Trace file saved to: {path_trace}")

        assert mae_score < 0.03, "Anomaly score MAE too high"
        assert max_error < 0.10, "Max anomaly score error too high"
        assert mae_pred < 2.0, "Prediction fluctuation too high"
        assert corr > 0.9, "Score correlation too low"

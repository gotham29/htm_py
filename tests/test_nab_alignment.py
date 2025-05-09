import yaml
import unittest
import numpy as np
import pandas as pd
from htm_py.htm_model import HTMModel

class TestNabAlignment(unittest.TestCase):

    def test_nab_alignment_mae_and_prediction(self):
        limit = 100
        path_nab = "data/NAB_art_daily_jumpsup.csv"
        path_numenta = "results/NAB_art_daily_jumpsup_NumentaTM.csv"
        path_config = "config/NAB_art_daily_jumpsup.yaml"
        # path_trace = f"results/NAB_alignment_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(path_config, "r") as file:
            config = yaml.safe_load(file)

        df = pd.read_csv(path_nab)[:limit]
        timestamps = df["timestamp"].tolist()
        values = df["value"].tolist()

        df_numenta = pd.read_csv(path_numenta)[:limit]
        numenta_scores = df_numenta["raw_score"].tolist()

        model = HTMModel(config=config)

        scores = []
        for t, v in zip(timestamps, values):
            input_data = {"timestamp": t, "value": v}
            anomaly_score, _ = model.compute(input_data, learn=True)
            scores.append(anomaly_score)

        assert len(scores) == len(numenta_scores), "Length mismatch between predicted and reference scores"

        # === Compute Mean Absolute Error
        mae = np.mean(np.abs(np.array(scores) - numenta_scores))

        print(f"MAE: {mae:.4f}")
        assert mae < 0.05, f"MAE too high! Found {mae:.4f}, expected < 0.05"

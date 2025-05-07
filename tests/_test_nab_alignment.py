import unittest
import os
import pandas as pd
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder

class TestNABAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set limit on rumber of rows 
        limit = 500

        # Load NAB dataset
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(root, "data", "NAB_art_daily_jumpsup.csv")
        numenta_path = os.path.join(root, "results", "NAB_art_daily_jumpsup_NumentaTM.csv")

        df = pd.read_csv(data_path)[:limit]
        cls.timestamps = pd.to_datetime(df["timestamp"])
        cls.values = df["value"].tolist()

        numenta_df = pd.read_csv(numenta_path)[:limit]
        cls.numenta_scores = numenta_df["raw_score"].tolist()

        # Setup HTM Model
        rdse = RDSE(min_val=0, max_val=114.4, resolution=0.88, n=130, w=21)
        date_enc = DateEncoder(timeOfDay=(21, 9.49))
        encoder = MultiEncoder({"timestamp": date_enc, "value": rdse})
        cls.model = HTMModel(encoder)

    def test_stream_and_score(self):
        scores = []

        for t, v in zip(self.timestamps, self.values):
            encoded = self.model.encoder.encode({"timestamp": t, "value": v})
            score, _ = self.model.run(encoded, learn=True)
            scores.append(score)

        self.assertEqual(len(scores), len(self.numenta_scores))

        # --- Evaluation ---
        import numpy as np
        from scipy.stats import pearsonr
        from sklearn.metrics import mean_squared_error

        htm = np.array(scores)
        nupic = np.array(self.numenta_scores)

        corr, _ = pearsonr(htm, nupic)
        rmse = mean_squared_error(nupic, htm, squared=False)

        print(f"Pearson Correlation: {corr:.4f}")
        print(f"RMSE: {rmse:.4f}")

        self.assertGreater(corr, 0.90)
        self.assertLess(rmse, 0.2)

        # Check decay in steady-state region (last 300 timesteps)
        tail_scores = htm[-300:]
        self.assertLess(np.mean(tail_scores), 0.2)

        # Check high anomaly near known jump (~timestamp 5000)
        spike_index = 5000
        self.assertGreater(htm[spike_index], 0.8)

if __name__ == "__main__":
    unittest.main()

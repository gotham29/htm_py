import os
import unittest
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder


class TestNABAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        limit = 225

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(root, "data", "NAB_art_daily_jumpsup.csv")
        numenta_path = os.path.join(root, "results", "NAB_art_daily_jumpsup_NumentaTM.csv")

        df = pd.read_csv(data_path)[:limit]
        cls.timestamps = pd.to_datetime(df["timestamp"])
        cls.values = df["value"].tolist()

        numenta_df = pd.read_csv(numenta_path)[:limit]
        cls.numenta_scores = numenta_df["raw_score"].tolist()

    def test_anomaly_score_alignment_with_nab(self):
        # NAB-compliant encoder
        rdse = RDSE(min_val=0, max_val=114.4, resolution=0.88, n=130, w=21)
        date_enc = DateEncoder(timeOfDay=(21, 9.49))
        encoder = MultiEncoder({"timestamp": date_enc, "value": rdse})

        model = HTMModel(encoder, use_sp=False)

        # Patch logging into adaptSegment and createSegment
        orig_adapt = model.tm.adaptSegment
        orig_create = model.tm.connections.createSegment

        def wrapped_adapt(*args, **kwargs):
            print(f"[WRAP] adaptSegment called with segment {args[1]}")
            return orig_adapt(*args, **kwargs)

        def wrapped_create(*args, **kwargs):
            seg = orig_create(*args, **kwargs)
            print(f"[WRAP] createSegment created segment {seg}")
            return seg

        model.tm.adaptSegment = wrapped_adapt
        model.tm.connections.createSegment = wrapped_create

        our_scores = []
        detailed_log = []

        # Bootstrap learning with one clean step
        model.tm.prevPredictiveCells.clear()
        model.run({"timestamp": self.timestamps[0], "value": self.values[0]}, learn=True)

        for i in range(1, len(self.timestamps)):
            t, v = self.timestamps[i], self.values[i]
            input_data = {"timestamp": t, "value": v}

            anomaly_score, _ = model.run(input_data, learn=True)
            our_scores.append(anomaly_score)

            if i < 10 or i % 50 == 0:
                print(f"\n🧪 Step {i}")
                print(f"Timestamp: {t}, Value: {v}")
                print(f"WinnerCells: {sorted(model.tm.winnerCells)}")
                print(f"PrevPredictiveCells: {sorted(model.tm.prevPredictiveCells)}")
                print(f"Anomaly Score: {anomaly_score:.3f} vs NuPIC {self.numenta_scores[i]:.3f}")

                detailed_log.append({
                    "i": i,
                    "t": t,
                    "val": v,
                    "score": anomaly_score,
                    "nupic": self.numenta_scores[i],
                    "winner": sorted(model.tm.winnerCells),
                    "predict": sorted(model.tm.prevPredictiveCells),
                })

        print("Our anomaly scores:", our_scores[:10])
        ref_scores = self.numenta_scores[1:]
        print("NAB anomaly scores:", ref_scores[:10])
        assert len(our_scores) >= 2 and len(ref_scores) >= 2, "Too few scores to compare"

        print("\n🔬 Sample logs:")
        for row in detailed_log:
            print(f"[{row['i']:3}] ts={row['t']} val={row['val']:6.2f} ours={row['score']:.3f} nupic={row['nupic']:.3f} "
                  f"winner={row['winner']} pred={row['predict']}")

        htm = np.array(our_scores[199:])
        nupic = np.array(self.numenta_scores[200:])

        if np.std(htm) == 0:
            print("⚠️ All HTM scores are constant. No correlation possible.")
            corr = float('nan')
        else:
            corr, _ = pearsonr(htm, nupic)

        mse = mean_squared_error(nupic, htm)
        rmse = mse ** 0.5
        mae = np.mean(np.abs(nupic - htm))

        print("\n🧪 NAB Raw Score Alignment:")
        print(f"Pearson Correlation: {corr:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        self.assertGreater(corr, 0.90, "HTM vs NAB correlation too low")
        self.assertLess(rmse, 0.15, "HTM vs NAB RMSE too high")
        self.assertLess(mae, 0.10, "HTM vs NAB MAE too high")


if __name__ == "__main__":
    unittest.main()

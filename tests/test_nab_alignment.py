import os
import unittest
import numpy as np
import pandas as pd
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder


class TestNABAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Limit rows for speed
        limit = 500

        # Paths
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(root, "data", "NAB_art_daily_jumpsup.csv")
        numenta_path = os.path.join(root, "results", "NAB_art_daily_jumpsup_NumentaTM.csv")

        df = pd.read_csv(data_path)[:limit]
        cls.timestamps = pd.to_datetime(df["timestamp"])
        cls.values = df["value"].tolist()

        numenta_df = pd.read_csv(numenta_path)[:limit]
        cls.numenta_scores = numenta_df["raw_score"].tolist()

        # Encoder & model
        rdse = RDSE(min_val=0, max_val=114.4, resolution=0.88, n=130, w=21)
        date_enc = DateEncoder(timeOfDay=(21, 9.49))
        encoder = MultiEncoder({"timestamp": date_enc, "value": rdse})
        cls.model = HTMModel(encoder)

    def test_tm_sequence_learning_art_jumpsup(self):
        pred_counts = []
        total_segments_created = 0
        total_synapses_created = 0
        segment_synapse_counts = []

        for t, v in zip(self.timestamps, self.values):
            encoded = self.model.encoder.encode({"timestamp": t, "value": v})
            anomaly_score, pred_count = self.model.run(encoded, learn=True)
            pred_counts.append(pred_count)

            tm = self.model.tm
            segment_synapse_counts += [
                len(tm.connections.synapsesForSegment(seg))
                for seg in tm.connections.segments
            ]
            total_segments_created = len(tm.connections.segments)
            total_synapses_created = sum(segment_synapse_counts)

        # Assertions
        self.assertGreater(np.mean(pred_counts), 0.2, "Predictive count too low — TM not learning sequences")
        self.assertGreater(total_segments_created, 50, "Too few segments created — learning not progressing")
        self.assertGreater(total_synapses_created, 200, "Too few synapses — underfitting or encoding problem")
        self.assertGreaterEqual(np.mean(segment_synapse_counts), 4, "Each segment should have at least 4 synapses on average")

        print("✅ test_tm_sequence_learning_art_jumpsup passed.")

    def test_anomaly_score_alignment_with_nab(self):
        our_scores = []

        for t, v in zip(self.timestamps, self.values):
            encoded = self.model.encoder.encode({"timestamp": t, "value": v})
            anomaly_score, _ = self.model.run(encoded, learn=True)
            our_scores.append(anomaly_score)

        # Convert to NumPy for vectorized math
        htm = np.array(our_scores)
        nupic = np.array(self.numenta_scores)

        # --- Evaluation ---
        from scipy.stats import pearsonr
        from sklearn.metrics import mean_squared_error

        corr, _ = pearsonr(htm, nupic)
        rmse = mean_squared_error(nupic, htm, squared=False)
        mae = np.mean(np.abs(nupic - htm))

        print("\n🧪 Line-by-Line NAB Alignment:")
        print(f"Pearson Correlation: {corr:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        self.assertGreater(corr, 0.90, "HTM vs NAB correlation too low")
        self.assertLess(rmse, 0.15, "HTM vs NAB RMSE too high")
        self.assertLess(mae, 0.10, "HTM vs NAB MAE too high")

        # Optional: check that anomaly spike occurs near known event
        max_index_nupic = np.argmax(nupic)
        max_index_htm = np.argmax(htm)
        offset = abs(max_index_nupic - max_index_htm)
        print(f"Peak anomaly at index: Numenta={max_index_nupic}, Ours={max_index_htm}, offset={offset}")
        self.assertLess(offset, 5, "Peak anomaly index deviates too far from NAB")

        print("✅ test_anomaly_score_alignment_with_nab passed.")


if __name__ == "__main__":
    unittest.main()

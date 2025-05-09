from htm_py.encoders.multi import MultiEncoder
from htm_py.spatial_pooler import SpatialPooler
from htm_py.temporal_memory import TemporalMemory
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder


class HTMModel:
    def __init__(self, config, encoder=None):
        enc_cfg = config["encoder"]
        if isinstance(encoder, MultiEncoder):
            self.encoder = encoder
        else:
            encoders = {}
            for feature in enc_cfg.get("rdse_features", []):
                encoders[feature["name"]] = RDSE(
                    min_val=feature["min_val"],
                    max_val=feature["max_val"],
                    n=feature["n"],
                    w=feature["w"]
                )
            if "timeOfDay" in enc_cfg:
                encoders["timestamp"] = DateEncoder(
                    timeOfDay=(enc_cfg["timeOfDay"]["n"], enc_cfg["timeOfDay"]["rotation"])
                )
            self.encoder = MultiEncoder(encoders)

        self.use_sp = config.get("use_sp", False)
        self.sp = SpatialPooler(**config["sp"]) if self.use_sp else None
        self.tm = TemporalMemory(**config["tm"])

    def compute(self, input_data, learn=True):
        """
        Compute the anomaly score and prediction ambiguity for a single timestep.

        Args:
            input_data (dict): Raw input features to encode.
            learn (bool): Whether the model should learn at this step.

        Returns:
            (float, float): (Anomaly Score, Prediction Count)
        """
        encoded = self.encoder.encode(input_data)

        active_columns = (
            self.sp.compute(encoded, learn=learn) if self.use_sp
            else [i for i, bit in enumerate(encoded) if bit == 1]
        )

        anomaly_score, prediction_count = self.tm.compute(active_columns, learn=learn)

        return anomaly_score, prediction_count

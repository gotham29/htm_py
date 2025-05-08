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
        if self.use_sp:
            self.sp = SpatialPooler(**config["sp"])
        else:
            self.sp = None

        self.tm = TemporalMemory(**config["tm"])

    def compute(self, input_data, learn=True):
        encoded = self.encoder.encode(input_data)

        if self.use_sp:
            active_columns = self.sp.compute(encoded, learn=learn)
        else:
            # treat each active bit index as a column
            active_columns = [i for i, bit in enumerate(encoded) if bit == 1]

        anomaly_score, pred_count = self.tm.compute(active_columns, learn=learn)
        return anomaly_score, pred_count

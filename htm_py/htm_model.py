import numpy as np
from htm_py.temporal_memory import TemporalMemory
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder


class HTMModel:
    def __init__(self, encoder_params, tm_params):
        self.encoder = self._build_encoder(encoder_params)
        self.tm = TemporalMemory(**tm_params)

    def _build_encoder(self, encoder_params):
        encoders = {}
        if "rdse" in encoder_params:
            rdse_cfg = encoder_params["rdse"]
            encoders["rdse"] = RDSE(
                size=rdse_cfg.get("size", 2048),
                resolution=rdse_cfg.get("resolution", 0.88),
                seed=rdse_cfg.get("seed", 42),
                min_val=rdse_cfg.get("min_val", 0.0),
                max_val=rdse_cfg.get("max_val", 100.0),
            )
        if "date" in encoder_params:
            date_cfg = encoder_params["date"]
            encoders["date"] = DateEncoder(
                timeOfDay=date_cfg.get("timeOfDay", (21, 9.49)),
                weekend=date_cfg.get("weekend", 1)
            )
        return MultiEncoder(encoders)

    def compute(self, input_dict, learn=True):
        # Expect input_dict = {"value": float, "timestamp": datetime}
        encoding_input = {
            "rdse": input_dict["value"],
            "date": input_dict["timestamp"]
        }
        sdr = self.encoder.encode(encoding_input)
        active_columns = np.flatnonzero(sdr)

        anomaly_score, prediction_count = self.tm.compute(active_columns, learn)
        return anomaly_score, prediction_count

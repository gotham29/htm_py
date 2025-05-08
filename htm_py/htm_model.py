import numpy as np
from htm_py.temporal_memory import TemporalMemory
from htm_py.spatial_pooler import SpatialPooler
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder
import logging

logger = logging.getLogger("HTMModel")


class HTMModel:
    def __init__(self, config):
        self.config = config
        self.use_sp = config.get("use_sp", True)

        self.encoder = self._build_encoder(config["encoder"])
        if self.use_sp:
            self.sp = SpatialPooler(**config["sp"])
        self.tm = TemporalMemory(**config["tm"])

    def _build_encoder(self, encoder_config):
        encoders = {}
        rdse_features = encoder_config.get("rdse_features", [])
        assert len(rdse_features) > 0, "At least one RDSE feature must be defined"

        for feat in rdse_features:
            # WDND: build RDSE using fixed resolution of 0.88
            encoders[feat["name"]] = RDSE(
                min_val=feat["min_val"],
                max_val=feat["max_val"],
                n=feat["n"],
                w=feat["w"]
            )

        if "timeOfDay" in encoder_config:
            tod = encoder_config["timeOfDay"]
            encoders["timestamp"] = DateEncoder(timeOfDay=(tod["n"], tod["rotation"]))

        return MultiEncoder(encoders)

    def compute(self, input_value, learn=True):
        encoded = self.encoder.encode(input_value)

        if self.use_sp:
            active_columns = self.sp.compute(encoded, learn=learn)
        else:
            active_columns = np.nonzero(encoded)[0]  # 🛠 FIXED: Nonzero indices

        anomaly_score, prediction_count = self.tm.compute(active_columns, learn=learn)

        return {
            "anomaly_score": anomaly_score,
            "prediction_count": prediction_count,
            "normalized_prediction_count": self.tm.getNormalizedPredictionCount(),
            "predictive_cells": list(self.tm.predictiveCells),
            "active_cells": list(self.tm.activeCells),
        }

    def reset(self):
        self.tm.reset()

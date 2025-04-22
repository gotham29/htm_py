import numpy as np
from .encoders import ScalarEncoder
from .spatial.spatial_pooler import SpatialPooler
from .temporal.temporal_memory import TemporalMemory


class HTMModel:
    def __init__(
        self,
        encoders,
        model_type="single",
        sp_params=None,
        tm_params=None,
    ):
        self.model_type = model_type
        self.encoders = encoders
        self.sp_params = sp_params or self._default_sp_params()
        self.tm_params = tm_params or self._default_tm_params()

        self.models = {}

        if model_type in ["single", "shared"]:
            self.models["combined"] = self._create_model()
        elif model_type == "separate":
            for feature in encoders:
                self.models[feature] = self._create_model()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _default_sp_params(self):
        return {
            "inputDimensions": 100,
            "columnDimensions": 2048,
            "potentialPct": 0.85,
            "synPermActiveInc": 0.04,
            "synPermInactiveDec": 0.005,
            "synPermConnected": 0.1,
            "boostStrength": 0.0,
            "seed": 42,
        }

    def _default_tm_params(self):
        return {
            "columnDimensions": 2048,
            "cellsPerColumn": 32,
            "activationThreshold": 12,
            "initialPermanence": 0.21,
            "connectedPermanence": 0.5,
            "minThreshold": 9,
            "maxNewSynapseCount": 20,
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.1,
            "seed": 42,
        }

    def _create_model(self):
        return {
            "sp": SpatialPooler(**self.sp_params),
            "tm": TemporalMemory(**self.tm_params),
            "anomaly_scores": [],
            "prediction_counts": [],
        }

    def compute(self, input_row):
        if self.model_type in ["single", "shared"]:
            return self._compute_shared(input_row)
        else:
            return self._compute_multi(input_row)

    def _compute_shared(self, input_row):
        encoded = []
        for feature_name, encoder in self.encoders.items():
            val = float(input_row[feature_name])
            encoded.extend(encoder.encode(val))

        sp = self.models["combined"]["sp"]
        tm = self.models["combined"]["tm"]

        active_columns = sp.compute(encoded)
        prediction_count, anomaly_score = tm.compute(active_columns)

        self.models["combined"]["anomaly_scores"].append(anomaly_score)
        self.models["combined"]["prediction_counts"].append(prediction_count)

        return anomaly_score, prediction_count

    def _compute_multi(self, input_row):
        results = {}
        for feature_name in self.encoders:
            val = float(input_row[feature_name])
            encoded = self.encoders[feature_name].encode(val)

            sp = self.models[feature_name]["sp"]
            tm = self.models[feature_name]["tm"]

            active_columns = sp.compute(encoded)
            prediction_count, anomaly_score = tm.compute(active_columns)

            self.models[feature_name]["anomaly_scores"].append(anomaly_score)
            self.models[feature_name]["prediction_counts"].append(prediction_count)

            results[feature_name] = (anomaly_score, prediction_count)

        return results

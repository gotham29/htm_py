import numpy as np
from .encoders import ScalarEncoder

class HTMModel:
    def __init__(self, encoders, model_type="single", sp_params=None, tm_params=None):
        self.model_type = model_type
        self.encoders = encoders
        self.sp_params = sp_params or {}
        self.tm_params = tm_params or {}

        self.models = {}

        if model_type in ["single", "shared"]:
            self.models["combined"] = self._create_model()
        elif model_type == "separate":
            for feature in encoders:
                self.models[feature] = self._create_model()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _create_model(self):
        # Placeholder for SP and TM; will later be filled with real logic
        return {
            "sp": None,
            "tm": None,
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

        anomaly_score = float(np.random.rand())  # Placeholder
        prediction_count = int(np.random.randint(1, 10))  # Placeholder

        self.models["combined"]["anomaly_scores"].append(anomaly_score)
        self.models["combined"]["prediction_counts"].append(prediction_count)

        return anomaly_score, prediction_count

    def _compute_multi(self, input_row):
        results = {}
        for feature_name, encoder in self.encoders.items():
            val = float(input_row[feature_name])
            encoded = encoder.encode(val)

            anomaly_score = float(np.random.rand())  # Placeholder
            prediction_count = int(np.random.randint(1, 10))  # Placeholder

            self.models[feature_name]["anomaly_scores"].append(anomaly_score)
            self.models[feature_name]["prediction_counts"].append(prediction_count)

            results[feature_name] = (anomaly_score, prediction_count)

        return results

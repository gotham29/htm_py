from htm.encoders.scalar import ScalarEncoder
from htm.bindings.sdr import SDR
from htm.algorithms.temporal_memory import TemporalMemory
import numpy as np


class HTMmodel:
    def __init__(self, config):
        self.shared = config.get("shared", True)
        self.models = {}

        # Load TM parameters (with defaults if not specified)
        tm_params = config.get("tm_params", {})
        self.tm_args = {
            "columnDimensions": tm_params.get("column_dimensions", [2048]),
            "cellsPerColumn": tm_params.get("cells_per_column", 32),
            "activationThreshold": tm_params.get("activation_threshold", 13),
            "initialPermanence": tm_params.get("initial_permanence", 0.21),
            "connectedPermanence": tm_params.get("connected_permanence", 0.5),
            "permanenceIncrement": tm_params.get("permanence_increment", 0.1),
            "permanenceDecrement": tm_params.get("permanence_decrement", 0.1),
            "minThreshold": tm_params.get("min_threshold", 10),
            "maxNewSynapseCount": tm_params.get("max_new_synapse_count", 20),
        }

        if self.shared:
            self.models["shared"] = {
                "encoder": self._init_encoder(config["features"]),
                "tm": TemporalMemory(**self.tm_args)
            }
        else:
            for feature in config["features"]:
                encoder = ScalarEncoder(
                    n=feature.get("n", 100),
                    w=feature.get("w", 21),
                    minval=feature["minval"],
                    maxval=feature["maxval"],
                    clipInput=True
                )
                self.models[feature["name"]] = {
                    "encoder": encoder,
                    "tm": TemporalMemory(**self.tm_args)
                }

    def _init_encoder(self, features):
        total_n = sum(f.get("n", 100) for f in features)
        encoders = []
        for f in features:
            enc = ScalarEncoder(
                n=f.get("n", 100),
                w=f.get("w", 21),
                minval=f["minval"],
                maxval=f["maxval"],
                clipInput=True
            )
            encoders.append(enc)
        return encoders

    def compute(self, input_data, learn=True):
        results = {}

        if self.shared:
            sdr = SDR(self.models["shared"]["encoder"][0].n * len(input_data))
            encoded = []
            for i, val in enumerate(input_data):
                bits = self.models["shared"]["encoder"][i].encode(val)
                encoded.extend(bits)
            sdr.sparse = np.nonzero(encoded)[0].tolist()
            tm = self.models["shared"]["tm"]
            tm.compute(sdr, learn)
            results["anomaly_score"] = tm.anomaly
            results["prediction_count"] = len(tm.getPredictiveCells())
        else:
            anomaly_scores = []
            prediction_counts = []
            for i, (name, model) in enumerate(self.models.items()):
                enc = model["encoder"]
                tm = model["tm"]
                sdr = SDR(enc.n)
                sdr.sparse = enc.encode(input_data[i]).nonzero()[0].tolist()
                tm.compute(sdr, learn)
                anomaly_scores.append(tm.anomaly)
                prediction_counts.append(len(tm.getPredictiveCells()))
            results["anomaly_score"] = np.mean(anomaly_scores)
            results["prediction_count"] = np.mean(prediction_counts)

        return results

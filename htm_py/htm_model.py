import os
from htm_py.encoders.multi import MultiEncoder
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.rdse import RDSE
from htm_py.spatial_pooler import SpatialPooler
from htm_py.temporal_memory import TemporalMemory

class HTMModel:
    def __init__(self, config, encoder=None):
        enc_cfg = config["encoder"]

        # === Encoder Setup ===
        if isinstance(encoder, MultiEncoder):
            self.encoder = encoder
        else:
            encoders = {}
            for feature in enc_cfg.get("rdse_features", []):
                encoders[feature["name"]] = RDSE(
                    min_val=feature["min_val"],
                    max_val=feature["max_val"],
                    n=feature.get("n"),
                    w=feature.get("w", 21),
                    resolution=feature.get("resolution"),
                    # seed=feature.get("seed", None)
                )
            if "timeOfDay" in enc_cfg:
                time_cfg = enc_cfg["timeOfDay"]
                encoders["timestamp"] = DateEncoder(
                    timeOfDay=(time_cfg["n"], time_cfg["rotation"])
                )
            self.encoder = MultiEncoder(encoders)

        # === Spatial Pooler Setup ===
        self.use_sp = config.get("use_sp", False)
        sp_cfg = config.get("sp", {})

        if self.use_sp:
            # Ensure inputWidth is correctly set to encoder output
            if sp_cfg.get("inputWidth", 0) == 0:
                sp_cfg["inputWidth"] = self.encoder.output_width

            self.sp = SpatialPooler(
                inputDimensions=[sp_cfg["inputWidth"]],
                columnDimensions=[sp_cfg["columnCount"]],
                potentialPct=sp_cfg.get("potentialPct", 0.8),
                # globalInhibition=sp_cfg.get("globalInhibition", True),
                # numActiveColumnsPerInhArea=sp_cfg.get("numActiveColumnsPerInhArea", 40),
                synPermActiveInc=sp_cfg.get("synPermActiveInc", 0.003),
                synPermInactiveDec=sp_cfg.get("synPermInactiveDec", 0.0005),
                synPermConnected=sp_cfg.get("synPermConnected", 0.2),
                boostStrength=sp_cfg.get("boostStrength", 0.0),
                seed=sp_cfg.get("seed", 1956)
            )
        else:
            self.sp = None

        # === Temporal Memory Setup ===
        self.tm = TemporalMemory(**config["tm"])

    def compute(self, input_data, learn=True):
        """
        Compute anomaly score and prediction count for a single timestep.

        Args:
            input_data (dict): Input data values for encoding.
            learn (bool): Whether the model should learn.

        Returns:
            (float, float): (Anomaly Score, Prediction Count)
        """
        encoded = self.encoder.encode(input_data)

        active_columns = (
            self.sp.compute(encoded, learn=learn)
            if self.use_sp else 
            [i for i, bit in enumerate(encoded) if bit == 1]
        )

        if self.use_sp:
            log_path = "results/sp_active_columns_trace.csv"
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("timestep,num_active_columns\n")
            with open(log_path, "a") as f:
                f.write(f"{self.tm.iteration},{len(active_columns)}\n")

        anomaly_score, prediction_count = self.tm.compute(active_columns, learn=learn)
        return anomaly_score, prediction_count


# sp_diag_log = "results/sp_active_columns_trace.csv"
# if not os.path.exists(sp_diag_log):
#     with open(sp_diag_log, "w") as f:
#         f.write("timestep,num_active_columns\n")

# with open(sp_diag_log, "a") as f:
#     f.write(f"{self.tm.iteration},{len(active_columns)}\n")


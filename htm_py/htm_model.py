import numpy as np
from htm_py.encoders.multi import MultiEncoder
from htm_py.spatial_pooler import SpatialPooler
from htm_py.temporal_memory import TemporalMemory

from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder

class HTMModel:
    def __init__(self, encoder: MultiEncoder):
        self.encoder = encoder  #self._build_encoder(encoder_params)

        self.sp = SpatialPooler(
            inputDimensions=(self.encoder.getWidth(),),
            columnDimensions=(2048,),
            potentialPct=0.8,
            numActiveColumnsPerInhArea=40,
            synPermActiveInc=0.003,
            synPermInactiveDec=0.0005,
            synPermConnected=0.2,
            globalInhibition=True,
            seed=1956,
            boostStrength=0.0,
        )

        self.tm = TemporalMemory(
            columnDimensions=(2048,),
            cellsPerColumn=32,
            activationThreshold=20,
            minThreshold=13,
            initialPermanence=0.24,
            connectedPermanence=0.2,
            permanenceIncrement=0.04,
            permanenceDecrement=0.008,
            predictedSegmentDecrement=0.001,
            seed=1960,
        )

    # def _build_encoder(self, encoder_params):
    #     encoders = {}
    #     if "rdse" in encoder_params:
    #         rdse_cfg = encoder_params["rdse"]
    #         encoders["rdse"] = RDSE(
    #             n=rdse_cfg.get("n", 150),
    #             w=rdse_cfg.get("w", 21),
    #             min_val=rdse_cfg.get("min_val", 0.0),
    #             max_val=rdse_cfg.get("max_val", 114.4),
    #         )
    #     if "date" in encoder_params:
    #         date_cfg = encoder_params["date"]
    #         encoders["date"] = DateEncoder(
    #             timeOfDay=date_cfg.get("timeOfDay", (21, 9.49)),
    #             weekend=date_cfg.get("weekend", (2, 1))
    #         )
    #     return MultiEncoder(encoders)

    def reset(self):
        self.tm.reset()

    def run(self, encoded_input, learn=True):
        output = np.zeros(self.sp.numColumns, dtype=np.int8)
        self.sp.compute(encoded_input, learn=learn, output=output)
        active_columns = np.nonzero(output)[0]
        self.tm.compute(active_columns, learn=learn)

        anomaly_score = self.tm.calculate_anomaly_score()
        prediction_count = self.tm.calculate_prediction_count()
        return anomaly_score, prediction_count

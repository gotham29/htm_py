from htm_py.temporal_memory import TemporalMemory
from htm_py.encoders.rdse import RDSE
from htm_py.spatial_pooler import SpatialPooler  # if you use SP

class HTMModel:
    def __init__(self, config):
        self.config = config
        self.rdse = RDSE(**config["rdse"])
        self.sp = SpatialPooler(**config["sp"])
        self.tm = TemporalMemory(**config["tm"])

    def compute(self, input_value, learn=True):
        encoded = self.rdse.encode(input_value)
        active_columns = self.sp.compute(encoded, learn=learn)
        anomaly_score, prediction_count = self.tm.compute(active_columns, learn=learn)

        return {
            "anomaly_score": anomaly_score,
            "prediction_count": prediction_count,
            "normalized_prediction_count": self.tm.getNormalizedPredictionCount(),
        }



# import os
# import logging

# log_path = os.getenv("HTM_TRACE_LOG", "htm_trace.log")

# # 🧼 Clear handlers to prevent duplicate logs on re-run
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)
# logger.debug("Logger initialized. Writing to %s", log_path)

# import numpy as np
# from htm_py.spatial_pooler import SpatialPooler
# from htm_py.temporal_memory import TemporalMemory


# class HTMModel:
#     def __init__(self, encoder, use_sp=True):
#         self.encoder = encoder
#         self.use_sp = use_sp

#         self.sp = SpatialPooler(
#             inputDimensions=(self.encoder.getWidth(),),
#             columnDimensions=(2048,),
#             potentialPct=0.8,
#             numActiveColumnsPerInhArea=40,
#             synPermActiveInc=0.003,
#             synPermInactiveDec=0.0005,
#             synPermConnected=0.2,
#             globalInhibition=True,
#             seed=1956,
#             boostStrength=0.0,
#         )

#         self.tm = TemporalMemory(
#             columnDimensions=(2048,),
#             cellsPerColumn=32,
#             activationThreshold=20,
#             minThreshold=13,
#             initialPermanence=0.24,
#             connectedPermanence=0.2,
#             permanenceIncrement=0.04,
#             permanenceDecrement=0.008,
#             predictedSegmentDecrement=0.001,
#             seed=1960,
#         )

#     def run(self, input_data, learn=True):
#         encoded = self.encoder.encode(input_data)

#         if self.use_sp:
#             active_columns = np.zeros(2048, dtype=np.int32)
#             self.sp.compute(encoded, learn=learn, activeArray=active_columns)
#             active_cols = np.flatnonzero(active_columns)
#         else:
#             # Directly interpret encoder bits as column activity (1 bit = 1 column)
#             active_cols = np.flatnonzero(encoded)

#         self.tm.compute(active_cols.tolist(), learn=learn)

#         anomaly_score = self.tm.calculate_anomaly_score()
#         prediction_count = self.tm.calculate_prediction_count()

#         return anomaly_score, prediction_count

#     def reset(self):
#         self.tm.reset()

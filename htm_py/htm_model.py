from .encoders.rdse import RDSE
from .encoders.date import DateEncoder
from .encoders.multi import MultiEncoder
from .spatial_pooler import SpatialPooler
from .temporal_memory import TemporalMemory

import numpy as np

class HTMModel:
    def __init__(self, encoder_params, sp_params, tm_params):
        self.encoder = self._build_encoder(encoder_params)
        self.input_dimensions = (self.encoder.size,)
        
        self.sp = SpatialPooler(
            inputDimensions=self.input_dimensions,
            columnDimensions=(tm_params.get("columnDimensions", 2048),),
            potentialRadius=sp_params.get("potentialRadius", 2048),
            potentialPct=sp_params.get("potentialPct", 0.85),
            globalInhibition=sp_params.get("globalInhibition", True),
            # localAreaDensity=sp_params.get("localAreaDensity", -1.0),
            numActiveColumnsPerInhArea=sp_params.get("numActiveColumnsPerInhArea", 40),
            stimulusThreshold=sp_params.get("stimulusThreshold", 0),
            # synPermInactiveDec=sp_params.get("synPermInactiveDec", 0.008),
            # synPermActiveInc=sp_params.get("synPermActiveInc", 0.05),
            synPermConnected=sp_params.get("synPermConnected", 0.1),
            # minPctOverlapDutyCycles=sp_params.get("minPctOverlapDutyCycles", 0.001),
            # dutyCyclePeriod=sp_params.get("dutyCyclePeriod", 1000),
            # boostStrength=sp_params.get("boostStrength", 0.0),
            seed=sp_params.get("seed", 42),
            # spVerbosity=sp_params.get("spVerbosity", 0),
            # wrapAround=sp_params.get("wrapAround", True)
        )

        self.tm = TemporalMemory(**tm_params)
        self.last_active_columns = []

    def _build_encoder(self, encoder_params):
        encoders = {}
        if "rdse" in encoder_params:
            rdse_cfg = encoder_params["rdse"]
            encoders["rdse"] = RDSE(
                n=rdse_cfg.get("n", 150),
                w=rdse_cfg.get("w", 21),
                min_val=rdse_cfg.get("min_val", 0.0),
                max_val=rdse_cfg.get("max_val", 114.4),
            )
        if "date" in encoder_params:
            date_cfg = encoder_params["date"]
            encoders["date"] = DateEncoder(
                timeOfDay=date_cfg.get("timeOfDay", (21, 9.49)),
                # weekend=date_cfg.get("weekend", 1)
            )
        return MultiEncoder(encoders)

    def compute(self, input_data, learn=True,  iteration: int = 0):
        # Encode input
        encoding = self.encoder.encode(input_data)  #TODO: debug error: 'float' object cannot be interpreted as an integer
        encoding = np.array(encoding, dtype=np.uint32)

        # SP compute
        active_columns = np.zeros(self.tm.columnDimensions[0], dtype=np.uint32)
        self.sp.compute(encoding, False, active_columns, iteration)

        self.last_active_columns = active_columns.nonzero()[0].tolist()

        # print(f"\nencoding = {encoding}")
        # print(f"active_columns (SP input) = {active_columns}")
        # print(f"self.last_active_columns (SP output) = {self.last_active_columns}")

        # TM compute
        anomaly_score, prediction_count = self.tm.compute(self.last_active_columns, learn)
        return anomaly_score, prediction_count

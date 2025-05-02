# htm_model.py
import numpy as np
from .rdse_encoder import RDSE
from .connections import Connections
from .temporal_memory import TemporalMemory


class HTMModel:
    def __init__(self, enc_params, tm_params, seed=42):
        self.seed = seed

        # Initialize RDSE Encoder
        self.encoder = RDSE(
            resolution=enc_params.get("resolution", 0.1),
            n=enc_params.get("n", 2048),
            w=enc_params.get("w", 40),
            seed=seed,
        )

        # Setup TM using the same n and w as the encoder
        # self.n_columns = enc_params.get("n", 2048)
        # self.cells_per_column = tm_params.get("cellsPerColumn", 32)
        # total_cells = self.n_columns * self.cells_per_column

        # # Setup Connections
        # self.connections = Connections(
        #     num_cells=total_cells,
        # )

        # Setup TM
        self.tm = TemporalMemory(
            **tm_params
        #     connections=self.connections,
        #     params=tm_params,
        )

    def compute(self, record, learn=True, verbose=False):
        # Extract value field for encoding
        value = record["value"] if isinstance(record, dict) else record
        timestamp = record["timestamp"] if isinstance(record, dict) else None

        # Encode input
        sdr = self.encoder.encode(value) #self.dateencoder.encode(timestamp)

        # Active columns are simply the indices of True bits in the SDR
        active_columns = np.flatnonzero(sdr)

        # Compute TM
        outputs = self.tm.compute(active_columns, learn=learn, verbose=verbose)

        return outputs

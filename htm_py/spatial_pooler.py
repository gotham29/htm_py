import numpy as np

class SpatialPooler:
    def __init__(self,
                 inputDimensions,
                 columnDimensions,
                 potentialRadius=None,
                 potentialPct=1.0,
                 globalInhibition=True,
                 numActiveColumnsPerInhArea=40,
                 synPermConnected=0.1,
                 stimulusThreshold=0,
                 seed=None):
        self.inputDimensions = inputDimensions
        self.columnDimensions = columnDimensions
        self.numInputs = np.prod(inputDimensions)
        self.numColumns = np.prod(columnDimensions)
        self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
        self.stimulusThreshold = stimulusThreshold
        self.synPermConnected = synPermConnected
        self.globalInhibition = globalInhibition

        np.random.seed(seed)

        # Initialize a random fixed potential pool mask
        self.permanences = np.random.rand(self.numColumns, self.numInputs) * 0.2
        self.connectedSynapses = self.permanences >= self.synPermConnected

    def compute(self, input_vector):
        input_vector = np.asarray(input_vector, dtype=np.int8)
        if input_vector.ndim > 1:
            input_vector = input_vector.ravel()

        overlaps = np.zeros(self.numColumns, dtype=np.float64)

        # Compute overlaps: dot product of input with connected synapses
        for i in range(self.numColumns):
            connected = self.connectedSynapses[i]
            overlaps[i] = np.dot(connected.astype(np.int8), input_vector)

        # Apply stimulus threshold
        overlaps[overlaps < self.stimulusThreshold] = 0

        # Sort by (overlap descending, index ascending) for deterministic ties
        sorted_indices = sorted(
            range(self.numColumns),
            key=lambda i: (-overlaps[i], i)
        )

        top_k = sorted_indices[:self.numActiveColumnsPerInhArea]
        active_columns = [i for i in top_k if overlaps[i] > 0]

        return active_columns
    



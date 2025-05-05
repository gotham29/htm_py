import numpy as np

class SpatialPooler:
    def __init__(
        self,
        inputDimensions,
        columnDimensions,
        potentialRadius=2048,
        potentialPct=0.85,
        globalInhibition=True,
        localAreaDensity=-1.0,
        numActiveColumnsPerInhArea=40,
        stimulusThreshold=0,
        synPermInactiveDec=0.008,
        synPermActiveInc=0.05,
        synPermConnected=0.1,
        minPctOverlapDutyCycles=0.001,
        dutyCyclePeriod=1000,
        boostStrength=0.0,
        seed=42,
        spVerbosity=0,
        wrapAround=True
    ):
        self.inputDimensions = inputDimensions
        self.columnDimensions = columnDimensions
        self.numInputs = int(np.prod(self.inputDimensions))
        self.numColumns = columnDimensions[0]

        self.synPermConnected = synPermConnected
        self.synPermActiveInc = synPermActiveInc
        self.synPermInactiveDec = synPermInactiveDec
        self.stimulusThreshold = stimulusThreshold
        self.numActiveColumns = numActiveColumnsPerInhArea

        self.permanences = np.random.rand(self.numColumns, self.numInputs) * 0.2
        self.connected_synapses = self.permanences > self.synPermConnected

        self.overlaps = np.zeros(self.numColumns)
        self.boosts = np.ones(self.numColumns)

    def compute(self, input_vector, learn, active_columns_out):
        input_vector = np.array(input_vector, dtype=bool)

        # Calculate overlaps
        self.connected_synapses = self.permanences > self.synPermConnected
        overlaps = np.sum(self.connected_synapses * input_vector, axis=1)
        overlaps[overlaps < self.stimulusThreshold] = 0
        overlaps *= self.boosts

        self.overlaps = overlaps

        # Select top N columns
        active_columns = overlaps.argsort()[::-1][:self.numActiveColumns]
        active_columns_out[:] = 0
        active_columns_out[active_columns] = 1

        if learn:
            for i in range(self.numColumns):
                for j in range(self.numInputs):
                    if input_vector[j]:
                        self.permanences[i, j] += self.synPermActiveInc
                    else:
                        self.permanences[i, j] -= self.synPermInactiveDec

            self.permanences = np.clip(self.permanences, 0.0, 1.0)

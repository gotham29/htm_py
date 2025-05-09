import numpy as np
import logging

logger = logging.getLogger("SpatialPooler")

class SpatialPooler:
    def __init__(
        self,
        inputDimensions,
        columnDimensions,
        potentialPct=0.85,
        synPermActiveInc=0.003,
        synPermInactiveDec=0.0005,
        synPermConnected=0.2,
        boostStrength=0.0,
        seed=42,
    ):
        self.inputDimensions = inputDimensions
        self.columnDimensions = columnDimensions
        self.numInputs = np.prod(inputDimensions)
        self.numColumns = np.prod(columnDimensions)
        self.potentialPct = potentialPct
        self.synPermActiveInc = synPermActiveInc
        self.synPermInactiveDec = synPermInactiveDec
        self.synPermConnected = synPermConnected
        self.boostStrength = boostStrength
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # For each column, select a potential pool of input bits
        self.potentialPools = np.array([
            self.rng.choice(self.numInputs, size=int(self.numInputs * potentialPct), replace=False)
            for _ in range(self.numColumns)
        ])

        # Initialize permanence values randomly in [0, 0.4)
        self.permanences = np.array([
            self.rng.uniform(0.0, 0.4, size=len(pool)) for pool in self.potentialPools
        ])

        # Initialize boost and duty cycles
        self.boostFactors = np.ones(self.numColumns)
        self.activeDutyCycles = np.zeros(self.numColumns)
        self.minDutyCycles = np.zeros(self.numColumns)

    def compute(self, inputVector, learn=True):
        inputVector = np.array(inputVector).astype(np.float32)
        overlaps = np.zeros(self.numColumns)

        for i in range(self.numColumns):
            pool = self.potentialPools[i]
            perms = self.permanences[i]
            connected = perms >= self.synPermConnected
            overlaps[i] = np.sum(inputVector[pool][connected])

        # FIX: Use fixed-k inhibition instead of top-N%
        k = 40  # Numenta typically uses 40
        if k >= self.numColumns:
            active_columns = np.arange(self.numColumns)
        else:
            top_k_indices = np.argpartition(overlaps, -k)[-k:]
            # Optionally sort them to have the strongest overlaps first
            active_columns = top_k_indices[np.argsort(-overlaps[top_k_indices])]

        if learn:
            self._adapt_permanences(inputVector, active_columns)
            self._update_duty_cycles(active_columns)
            self._update_boost_factors()

        return active_columns

    def _adapt_permanences(self, inputVector, active_columns):
        for i in active_columns:
            pool = self.potentialPools[i]
            perms = self.permanences[i]
            inputBits = inputVector[pool]
            perms += self.synPermActiveInc * inputBits
            perms -= self.synPermInactiveDec * (1 - inputBits)
            self.permanences[i] = np.clip(perms, 0.0, 1.0)

    def _update_duty_cycles(self, active_columns):
        decay = 0.99
        self.activeDutyCycles *= decay
        self.activeDutyCycles[active_columns] += (1.0 - decay)

    def _update_boost_factors(self):
        # No boosting if boostStrength is zero
        if self.boostStrength == 0.0:
            self.boostFactors[:] = 1.0
            return

        maxDuty = np.max(self.activeDutyCycles)
        for i in range(self.numColumns):
            if self.activeDutyCycles[i] > self.minDutyCycles[i]:
                self.boostFactors[i] = 1.0
            else:
                self.boostFactors[i] = np.exp(-self.boostStrength * (self.minDutyCycles[i] - self.activeDutyCycles[i]))

    def get_permanences(self):
        return self.permanences

    def get_connected_synapses(self):
        return [
            pool[perms >= self.synPermConnected]
            for pool, perms in zip(self.potentialPools, self.permanences)
        ]

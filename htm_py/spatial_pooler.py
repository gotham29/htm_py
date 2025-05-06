import numpy as np


class SpatialPooler:
    def __init__(
        self,
        inputDimensions=(8,),
        columnDimensions=(4,),
        potentialPct=1.0,
        globalInhibition=True,
        localAreaDensity=None,
        numActiveColumnsPerInhArea=2,
        stimulusThreshold=0,
        synPermConnected=0.05,          # Lowered to ensure more connections
        synPermActiveInc=0.03,
        synPermInactiveDec=0.008,
        boostStrength=0.0,
        seed=42,
        wrapAround=True
    ):
        # Store all parameters
        self.inputDimensions = inputDimensions
        self.columnDimensions = columnDimensions
        self.potentialPct = potentialPct
        self.globalInhibition = globalInhibition
        self.localAreaDensity = localAreaDensity
        self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
        self.stimulusThreshold = stimulusThreshold
        self.synPermConnected = synPermConnected
        self.synPermActiveInc = synPermActiveInc
        self.synPermInactiveDec = synPermInactiveDec
        self.boostStrength = boostStrength
        self.seed = seed
        self.wrapAround = wrapAround
        self.numInputs = np.prod(self.inputDimensions)
        self.numColumns = np.prod(self.columnDimensions)
        self._activeDutyCycles = np.zeros(self.numColumns)
        self._boostFactors = np.ones(self.numColumns)
        self._minDutyCycles = np.zeros(self.numColumns)
        self._overlapDutyCycles = np.zeros(self.numColumns)

        np.random.seed(seed)

        # All inputs are potential by default
        self._potential_inputs = np.ones((self.numColumns, self.numInputs), dtype=bool)

        # Init permanence above synPermConnected to ensure some connected synapses
        self._permanences = np.random.uniform(
            low=synPermConnected - 0.05,   #+ 0.01
            high=synPermConnected + 0.15,   #+ 0.15
            size=(self.numColumns, self.numInputs)
        )

        # Init boost factors and connected synapses
        self._boostFactors = np.ones(self.numColumns)
        self.connectedSynapses = self._permanences >= self.synPermConnected

    def compute(self, inputVector, learn=True, output=None, iteration=None):
        """
        Executes one timestep of Spatial Pooler computation.

        Args:
            inputVector (np.ndarray): Binary input vector of size `numInputs`.
            learn (bool): Whether to adapt synapses based on active columns.
            output (np.ndarray): Optional array to write active columns into (1 = active).
            iteration (int): Optional timestep index (unused but accepted for compatibility).
        """
        overlaps = np.zeros(self.numColumns)

        for i in range(self.numColumns):
            connected = self._permanences[i] > self.synPermConnected
            connected_inputs = np.logical_and(connected, inputVector)
            overlap = np.sum(connected_inputs)

            if overlap < self.stimulusThreshold:
                overlaps[i] = 0
            else:
                overlaps[i] = overlap * self._boostFactors[i]

        # Inhibition
        if self.globalInhibition:
            active_columns = self._inhibitColumnsGlobal(overlaps)
        else:
            active_columns = self._inhibitColumnsLocal(overlaps)

        # Learning
        if learn:
            for i in active_columns:
                inc = np.logical_and(inputVector == 1, self._potential_inputs[i])
                dec = np.logical_and(inputVector == 0, self._potential_inputs[i])
                self._permanences[i][inc] += self.synPermActiveInc
                self._permanences[i][dec] -= self.synPermInactiveDec
                self._permanences[i] = np.clip(self._permanences[i], 0.0, 1.0)

        # Write to output
        if output is not None:
            output.fill(0)
            output[active_columns] = 1

        # Update duty cycles
        alpha = 0.005  # Numenta default
        self._activeDutyCycles *= (1 - alpha)
        self._activeDutyCycles[active_columns] += alpha

        # Compute minDutyCycles
        self._minDutyCycles = 0.01 * np.max(self._activeDutyCycles)

        # Boosting
        for i in range(self.numColumns):
            if self._activeDutyCycles[i] < self._minDutyCycles:
                self._boostFactors[i] += 0.01  # small boost increment
            else:
                self._boostFactors[i] = 1.0

        # Update overlap duty cycles (pre-inhibition)
        for i in range(self.numColumns):
            if overlaps[i] > self.stimulusThreshold:
                self._overlapDutyCycles[i] = (1 - alpha) * self._overlapDutyCycles[i] + alpha
            else:
                self._overlapDutyCycles[i] *= (1 - alpha)

        return active_columns

    def _inhibitColumnsGlobal(self, overlaps):
        """
        Select top-N columns globally based on overlap score.

        Args:
            overlaps (np.ndarray): Overlap scores for each column.

        Returns:
            np.ndarray: Indices of winning columns.
        """
        numActive = self.numActiveColumnsPerInhArea
        if numActive is None:
            raise ValueError("numActiveColumnsPerInhArea must be set for global inhibition.")

        if numActive >= self.numColumns:
            return np.arange(self.numColumns)

        # Pair each column with its overlap score and index (for deterministic tie-breaking)
        indexed = [(i, overlaps[i]) for i in range(self.numColumns)]
        # Sort by overlap descending, index ascending
        sorted_cols = sorted(indexed, key=lambda x: (-x[1], x[0]))
        winners = [i for i, overlap in sorted_cols if overlap > 0][:numActive]

        return np.array(winners, dtype=np.int32)

    def _inhibitColumnsLocal(self, overlaps):
        """
        Selects active columns using local inhibition (1D only for now).
        Returns indices of winning columns.
        """
        num_active = self.numActiveColumnsPerInhArea
        if num_active is None:
            raise ValueError("numActiveColumnsPerInhArea must be set for local inhibition.")

        # For now, treat local inhibition the same as global (no topological neighbors implemented)
        top_indices = np.argsort(overlaps)[::-1]
        active_columns = top_indices[:num_active]
        return active_columns[overlaps[active_columns] > 0]

    def _adapt_permanences(self, column_index, input_vector):
        """
        Adjust permanences for a single column based on input activity.

        If an input is active and part of the potential pool, increment its permanence.
        If inactive, decrement its permanence. Then clip and update connected synapses.
        """
        potential = self._potential_inputs[column_index]
        perm = self._permanences[column_index]

        # Update permanence values
        perm += self.synPermActiveInc * (input_vector & potential)
        perm -= self.synPermInactiveDec * (~input_vector & potential)

        # Clip to [0.0, 1.0]
        perm = np.clip(perm, 0.0, 1.0)
        self._permanences[column_index] = perm

        # Recalculate connected synapses
        self.connectedSynapses[column_index] = perm >= self.synPermConnected

from htm_py.connections import Connections
import numpy as np
import random


class TemporalMemory:
    def __init__(
        self,
        columnDimensions,
        cellsPerColumn,
        activationThreshold,
        initialPermanence,
        connectedPermanence,
        minThreshold,
        maxNewSynapseCount,
        permanenceIncrement,
        permanenceDecrement,
        predictedSegmentDecrement,
        maxSegmentsPerCell=255,
        maxSynapsesPerSegment=255,
        seed=42,
        checkInputs=True,
    ):
        # Core params
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.maxNewSynapseCount = maxNewSynapseCount
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.predictedSegmentDecrement = predictedSegmentDecrement
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.checkInputs = checkInputs

        self.seed = seed
        self.rng = random.Random(seed)

        self.numColumns = np.prod(columnDimensions)
        self.numCells = self.numColumns * cellsPerColumn

        self.connections = Connections(self.numCells)

        # Runtime state
        self.activeCells = set()
        self.winnerCells = set()
        self.predictiveCells = set()
        self.activeSegments = set()
        self.matchingSegments = set()

        self.prevActiveCells = set()
        self.prevWinnerCells = set()

        self.iteration = 0
        self.lastUsedIterationForSegment = dict()

    def reset(self):
        self.activeCells.clear()
        self.winnerCells.clear()
        self.predictiveCells = set()
        self.activeSegments.clear()
        self.matchingSegments.clear()

        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)

    def getWinnerCells(self):
        return self.winnerCells

    def getActiveCells(self):
        return self.activeCells

    def getPredictiveCells(self):
        return self.predictiveCells

    def compute(self, activeColumns, learn=True):
        self._activate_columns(activeColumns, learn)
        if learn:
            self._learn_segments()
        self._predict_next()
        self._update_state()
        anomaly_score = self._calculate_anomaly_score()
        prediction_count = self._calculate_prediction_count()
        return anomaly_score, prediction_count

    def _calculate_anomaly_score(self):
        if not self.activeCells:
            return 0.0
        matched = len(self.activeCells & self.predictiveCells)
        return 1.0 - (matched / len(self.activeCells))

    def _calculate_prediction_count(self):
        return len(self.activeCells & self.predictiveCells)

    def getNormalizedPredictionCount(self):
        """
        Returns: prediction count / (activeCells / cellsPerColumn),
        representing how many full columns were successfully predicted.
        """
        if not self.activeCells:
            return 0.0
        active_columns = len(self.activeCells) / self.cellsPerColumn
        if active_columns == 0:
            return 0.0
        return self._calculate_prediction_count() / active_columns

    def _activate_columns(self, activeColumns, learn):
        self.activeCells.clear()
        self.winnerCells.clear()

        for col in activeColumns:
            col_cells = self._cells_for_column(col)
            matching_segments = []

            for cell in col_cells:
                segments = self.connections.segmentsForCell(cell)
                for seg in segments:
                    active_synapses = sum(
                        1 for syn in self.connections.synapsesForSegment(seg)
                        if self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevWinnerCells
                    )
                    if active_synapses >= self.minThreshold:
                        matching_segments.append((seg, active_synapses))

            if matching_segments:
                best_seg, _ = max(matching_segments, key=lambda s: s[1])
                winner_cell = self.connections.cellForSegment(best_seg)
            else:
                winner_cell = self._get_least_used_cell(col)

                if learn and self.prevWinnerCells:
                    seg = self.connections.createSegment(winner_cell)
                    for cell in sorted(self.prevWinnerCells):
                        self.connections.createSynapse(seg, cell, self.initialPermanence)

            self._mark_column_active(col)
            self.winnerCells.add(winner_cell)

    def _learn_segments(self):
        for cell in self.activeCells:
            for seg in self.connections.segmentsForCell(cell):
                active_syn_count = sum(
                    1 for syn in self.connections.synapsesForSegment(seg)
                    if self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevActiveCells
                )
                if active_syn_count >= self.activationThreshold:
                    self._adapt_segment(seg)
                    self._grow_segment_if_needed(seg)
                    self.lastUsedIterationForSegment[seg] = self.iteration
        self.iteration += 1

    def _update_state(self):
        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)

    def _cells_for_column(self, col):
        start = col * self.cellsPerColumn
        return [start + i for i in range(self.cellsPerColumn)]

    def _get_least_used_cell(self, col):
        min_segments = float("inf")
        candidates = []
        for cell in self._cells_for_column(col):
            n = self.connections.numSegments(cell)
            if n < min_segments:
                min_segments = n
                candidates = [cell]
            elif n == min_segments:
                candidates.append(cell)
        return self.rng.choice(candidates)

    def _mark_column_active(self, col):
        self.activeCells.update(self._cells_for_column(col))

    def _activate_columns(self, activeColumns, learn):
        self.activeCells.clear()
        self.winnerCells.clear()

        for col in activeColumns:
            matching = self._get_matching_segments(col)
            if matching:
                best = self._best_matching_segment(matching)
                winner_cell = self.connections.cellForSegment(best)
                if learn:
                    self._adapt_segment(best)
                    self._grow_segment_if_needed(best)
            else:
                winner_cell = self._get_least_used_cell(col)
                if learn and self.prevWinnerCells:
                    self._grow_new_segment(winner_cell)

            self._mark_column_active(col)
            self.winnerCells.add(winner_cell)

    def _get_matching_segments(self, col):
        matching = []
        for cell in self._cells_for_column(col):
            for seg in self.connections.segmentsForCell(cell):
                count = sum(
                    1 for syn in self.connections.synapsesForSegment(seg)
                    if self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevWinnerCells
                )
                if count >= self.minThreshold:
                    matching.append((seg, count))
        return matching

    def _best_matching_segment(self, segments_with_counts):
        return max(segments_with_counts, key=lambda s: s[1])[0]

    def _adapt_segment(self, segment):
        if not self.prevWinnerCells:
            return

        for syn in list(self.connections.synapsesForSegment(segment)):
            data = self.connections.dataForSynapse(syn)
            cell = data["presynapticCell"]
            if cell in self.prevWinnerCells:
                new_perm = min(data["permanence"] + self.permanenceIncrement, 1.0)
            else:
                new_perm = max(data["permanence"] - self.permanenceDecrement, 0.0)

            if new_perm < 0.0001:
                self.connections.destroySynapse(syn)
            else:
                self.connections.updateSynapsePermanence(syn, new_perm)

        if len(self.connections.synapsesForSegment(segment)) == 0:
            self.connections.destroySegment(segment)

    def _grow_segment_if_needed(self, segment):
        existing = {
            self.connections.dataForSynapse(syn)["presynapticCell"]
            for syn in self.connections.synapsesForSegment(segment)
        }
        new_cells = sorted(set(self.prevWinnerCells) - existing)
        to_add = new_cells[:self.maxNewSynapseCount - len(existing)]
        for cell in to_add:
            self.connections.createSynapse(segment, cell, self.initialPermanence)

    def _grow_new_segment(self, cell):
        segments = self.connections.segmentsForCell(cell)
        if len(segments) >= self.maxSegmentsPerCell:
            oldest = min(segments, key=lambda seg: self.lastUsedIterationForSegment.get(seg, 0))
            self.connections.destroySegment(oldest)

        seg = self.connections.createSegment(cell)
        self.lastUsedIterationForSegment[seg] = self.iteration
        for cell in sorted(self.prevWinnerCells)[:self.maxNewSynapseCount]:
            self.connections.createSynapse(seg, cell, self.initialPermanence)
        return seg

    def _get_least_used_cell(self, col):
        min_count = float("inf")
        candidates = []
        for cell in self._cells_for_column(col):
            count = self.connections.numSegments(cell)
            if count < min_count:
                min_count = count
                candidates = [cell]
            elif count == min_count:
                candidates.append(cell)
        return self.rng.choice(candidates)

    def _predict_next(self):
        self.predictiveCells.clear()

        for cell in range(self.numCells):
            for seg in self.connections.segmentsForCell(cell):
                active_syns = sum(
                    1 for syn in self.connections.synapsesForSegment(seg)
                    if self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevActiveCells
                )
                if active_syns >= self.activationThreshold:
                    self.predictiveCells.add(cell)
                    self.lastUsedIterationForSegment[seg] = self.iteration

    def getAnomalyScore(self):
        return self._calculate_anomaly_score()

    def getPredictionCount(self):
        return self._calculate_prediction_count()

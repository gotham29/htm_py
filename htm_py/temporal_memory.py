from htm_py.connections import Connections
import numpy as np
import random

import logging
logger = logging.getLogger("HTMModel")

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
        self.activeColumns = set(activeColumns)  # ✅ Capture early for fidelity
        if self.iteration == 0:
            self.prevActiveCells.clear()
            self.prevWinnerCells.clear()

        self._activate_columns(activeColumns, learn)

        if learn:
            self._learn_segments()

        self._predict_next()
        if learn:
            self._punish_incorrect_predictions()
        self._update_state()
        anomaly_score = self._calculate_anomaly_score()
        prediction_count = self._calculate_prediction_count()

        return anomaly_score, prediction_count

    def _activate_columns(self, activeColumns, learn):
        self.activeCells.clear()
        self.winnerCells.clear()

        for col in activeColumns:
            matching = self._get_matching_segments(col)

            if matching:
                best = self._best_matching_segment(matching)
                winner_cell = self.connections.cellForSegment(best)
            else:
                winner_cell = self._get_least_used_cell(col)

                if learn:
                    self._grow_new_segment(winner_cell)

            self._mark_column_active(col)
            self.winnerCells.add(winner_cell)

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

    def _learn_segments(self):
        for col in self.activeColumns:  # ✅ Use captured set
            for cell in self._cells_for_column(col):
                segments = self.connections.segmentsForCell(cell)
                matching_segments = [
                    (seg, len(self.connections.active_synapses(seg, self.prevActiveCells)))
                    for seg in segments
                    if len(self.connections.active_synapses(seg, self.prevActiveCells)) >= self.minThreshold
                ]

                if matching_segments:
                    best_seg = self._best_matching_segment(matching_segments)
                    self._adapt_segment(best_seg, self.prevActiveCells)
                    self._grow_segment_if_needed(best_seg)
                    self.lastUsedIterationForSegment[best_seg] = self.iteration
                else:
                    if not self.prevWinnerCells:
                        continue
                    seg = self._grow_new_segment(cell)
                    if seg is not None:
                        self.lastUsedIterationForSegment[seg] = self.iteration

        self.iteration += 1
        self._decrement_predicted_segments()

    def _punish_incorrect_predictions(self):
        """
        Decrease permanence on all segments that were predicting,
        but whose cells did not become active (false positives).
        """
        for cell in range(self.numCells):
            if cell not in self.activeCells and cell in self.predictiveCells:
                for seg in self.connections.segmentsForCell(cell):
                    # Only segments that contributed to a false positive
                    active_syns = [
                        syn for syn in self.connections.synapsesForSegment(seg)
                        if (
                            self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevActiveCells and
                            self.connections.dataForSynapse(syn)["permanence"] >= self.connectedPermanence
                        )
                    ]
                    if len(active_syns) >= self.activationThreshold:
                        for syn in active_syns:
                            perm = self.connections.dataForSynapse(syn)["permanence"]
                            new_perm = max(0.0, perm - self.predictedSegmentDecrement)
                            self.connections.updateSynapsePermanence(syn, new_perm)

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

    def _adapt_segment(self, segment, learning_cells):
        if not learning_cells: #self.prevWinnerCells:
            return

        for syn in list(self.connections.synapsesForSegment(segment)):
            data = self.connections.dataForSynapse(syn)
            cell = data["presynapticCell"]
            if cell in learning_cells:
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
        existing_synapses = self.connections.synapsesForSegment(segment)
        existing_cells = {
            self.connections.dataForSynapse(syn)["presynapticCell"]
            for syn in existing_synapses
        }

        available_cells = sorted(set(self.prevWinnerCells) - existing_cells)
        remaining_capacity = self.maxSynapsesPerSegment - len(existing_synapses)

        to_add = available_cells[:min(self.maxNewSynapseCount, remaining_capacity)]

        for cell in to_add:
            self.connections.createSynapse(segment, cell, self.initialPermanence)

    def _grow_new_segment(self, cell):
        if not self.prevWinnerCells:
            logger.debug(f"[t={self.iteration}] Skipping segment creation: no prevWinnerCells")
            return None

        if self.iteration <= 2:
            logger.debug(f"[t={self.iteration}] DEFERRING segment creation to allow burn-in")
            return None

        # ✅ WDND: context must be winner AND active cells
        context = self.prevWinnerCells & self.prevActiveCells
        if not context:
            logger.debug(f"[t={self.iteration}] Skipping segment: no active winner context")
            return None

        segments = self.connections.segmentsForCell(cell)
        if len(segments) >= self.maxSegmentsPerCell:
            oldest = min(segments, key=lambda seg: self.lastUsedIterationForSegment.get(seg, 0))
            self.connections.destroySegment(oldest)

        seg = self.connections.createSegment(cell)
        self.lastUsedIterationForSegment[seg] = self.iteration

        initial_perm = min(self.initialPermanence, self.connectedPermanence - 0.01)
        for src in sorted(context)[:self.maxNewSynapseCount]:
            logger.debug(f"[t={self.iteration}] Synapse: {src} → {seg}, perm={initial_perm:.3f}")
            self.connections.createSynapse(seg, src, initial_perm)
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

        if not self.prevActiveCells:
            return  # 🔒 can't form predictions without context

        for cell in range(self.numCells):
            for seg in self.connections.segmentsForCell(cell):
                active_syns = 0
                for syn in self.connections.synapsesForSegment(seg):
                    data = self.connections.dataForSynapse(syn)
                    if (
                        data["presynapticCell"] in self.prevActiveCells and
                        data["permanence"] >= self.connectedPermanence
                    ):
                        active_syns += 1

                if active_syns >= self.activationThreshold:
                    self.predictiveCells.add(cell)
                    self.lastUsedIterationForSegment[seg] = self.iteration
                elif self.predictedSegmentDecrement > 0.0:
                    # 🔒 Penalize segments that tried to predict but failed
                    for syn in self.connections.synapsesForSegment(seg):
                        perm = self.connections.dataForSynapse(syn)["permanence"]
                        new_perm = max(0.0, perm - self.predictedSegmentDecrement)
                        self.connections.updateSynapsePermanence(syn, new_perm)

    def _decrement_predicted_segments(self):
        """
        Apply predictedSegmentDecrement to all previously predictive segments
        that did not become active. This matches Numenta's Phase 2 decay logic.
        """
        for cell in range(self.numCells):
            for seg in self.connections.segmentsForCell(cell):
                if self._was_predicted_last_step(seg) and not self._is_segment_active(seg):
                    for syn in self.connections.synapsesForSegment(seg):
                        data = self.connections.dataForSynapse(syn)
                        new_perm = max(data["permanence"] - self.predictedSegmentDecrement, 0.0)
                        self.connections.updateSynapsePermanence(syn, new_perm)

    def _was_predicted_last_step(self, segment):
        """
        Check if a segment was predicted last step: enough synapses connected to prevActiveCells.
        """
        syn_count = sum(
            1 for syn in self.connections.synapsesForSegment(segment)
            if (
                self.connections.dataForSynapse(syn)["presynapticCell"] in self.prevActiveCells and
                self.connections.dataForSynapse(syn)["permanence"] >= self.connectedPermanence
            )
        )
        return syn_count >= self.activationThreshold

    def _is_segment_active(self, segment):
        """
        Check if segment is active now (connected synapses from current prevActiveCells).
        """
        return any(
            self.connections.dataForSynapse(syn)["presynapticCell"] in self.activeCells and
            self.connections.dataForSynapse(syn)["permanence"] >= self.connectedPermanence
            for syn in self.connections.synapsesForSegment(segment)
        )

    def getAnomalyScore(self):
        return self._calculate_anomaly_score()

    def getPredictionCount(self):
        return self._calculate_prediction_count()

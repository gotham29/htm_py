import numpy as np
from typing import Set, List, Tuple, Union
from collections import defaultdict
from htm_py.connections import Connections

import logging

logger = logging.getLogger(__name__)


class TemporalMemory:
    def __init__(self,
                seed=None,
                 columnDimensions=(2048,),
                 cellsPerColumn=32,
                 activationThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.5,
                 minThreshold=10,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.1,
                 predictedSegmentDecrement=0,
                 maxNewSynapseCount=20,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255):
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
        self.seed = seed
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment

        self._timestep = 0  # Internal counter
        self.lastUsedCell = -1
        self.numColumns = 1
        for dim in self.columnDimensions:
            self.numColumns *= dim
        self.numCells = self.numColumns * self.cellsPerColumn

        self.connections = Connections(num_cells=self.numCells)

        self.activeCells = set()
        self.winnerCells = set()
        self.predictiveCells = set()

        self.prevActiveCells = set()
        self.prevWinnerCells = set()
        self.prevPredictiveCells = set()

        self.activeSegments = set()
        self.segmentActiveForCell = {}
        self.winnerCellForColumn = {}
        self.predictedSegmentForCell = {}

        self.contextMemory = defaultdict(dict)  # context → {cell: segment}

        # logger.info("TemporalMemory initialized with %d columns, %d cells total",
        #             self.numColumns, self.numCells)


    def compute(self, activeColumns: Union[list[int], np.ndarray], learn: bool = True):        
        if isinstance(activeColumns, np.ndarray):
            if not np.issubdtype(activeColumns.dtype, np.integer):
                raise TypeError("activeColumns must be a list or array of integers")
            activeColumns = activeColumns.tolist()
        elif isinstance(activeColumns, list):
            if not all(isinstance(i, int) for i in activeColumns):
                raise TypeError("activeColumns must be a list or array of integers")
        else:
            raise TypeError("activeColumns must be a list or array of integers")
        # print(f"\n[TM] >>> compute(activeColumns={activeColumns}, learn={learn})")
        self.learn = learn
        self.activeColumns = activeColumns

        # print("[DEBUG] _activate_columns is:", self._activate_columns)
        # print("[DEBUG] type:", type(self._activate_columns))

        self.activeCells, self.winnerCells = self._activate_columns(activeColumns)
        # print(f"[TM] ActiveCells after _activate_columns: {sorted(self.activeCells)}")
        # print(f"[TM] WinnerCells after _activate_columns: {sorted(self.winnerCells)}")

        if learn:
            self._learn_segments(self.activeColumns, self.prevWinnerCells)

        # 🔥 Fix: move this BEFORE prediction
        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)

        # print(f"[DEBUG] prevActiveCells before prediction: {sorted(self.prevActiveCells)}")
        self._predict_cells()
        logger.debug(f"\n[TIMESTEP] = { self._timestep}")
        logger.debug(f"[TM] PredictiveCells after _predict_cells: {sorted(self.predictiveCells)}")

        anomaly_score = self.calculate_anomaly_score()
        prediction_count = self.calculate_prediction_count()
        logger.debug(f"[TM] Anomaly Score: {anomaly_score:.3f}, Prediction Count: {prediction_count:.3f}")

        # Finalize timestep state
        self.prevPredictiveCells = set(self.predictiveCells)
        # print("[DEBUG] active_synapses function found? ->", hasattr(self.connections, "active_synapses"))

        self._timestep += 1

        return anomaly_score, prediction_count

    def calculate_anomaly_score(self):
        if not self.winnerCells:
            logger.debug("[ANOMALY] No winner cells → anomaly score = 0.0")
            return 0.0

        correctly_predicted = self.winnerCells & self.prevPredictiveCells
        score = 1.0 - len(correctly_predicted) / len(self.winnerCells)
        # print(f"[ANOMALY] WinnerCells: {self.winnerCells}, PrevPredicted: {self.prevPredictiveCells}")
        # print(f"[ANOMALY] CorrectlyPredicted: {correctly_predicted} → anomaly score = {score:.3f}")
        return score

    def calculate_prediction_count(self):
        if not self.activeColumns:
            return 0.0  # Avoid divide-by-zero when no active columns
        return len(self.prevPredictiveCells) / len(self.activeColumns)

    def _cells_for_column(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    # def _activate_columns(self, activeColumns: List[int]) -> Tuple[Set[int], Set[int]]:
    #     if not activeColumns:
    #         return set(), set()

    #     winnerCells = set()
    #     activeCells = set()

    #     for col in activeColumns:
    #         predictedCells = [
    #             c for c in self._cells_for_column(col)
    #             if c in self.prevPredictiveCells
    #         ]
    #         if predictedCells:
    #             winnerCell = predictedCells[0]
    #             bestSeg, overlap = getBestMatchingSegment(
    #                 self.connections, winnerCell, set(self.prevActiveCells),
    #                 self.activationThreshold, self.connectedPermanence, return_overlap=True
    #             )
    #             # if bestSeg is not None:
    #             #     print(f"[ACTIVATE] Best segment for cell {winnerCell}: {bestSeg} from sources {[self.connections.dataForSynapse(s).presynapticCell for s in self.connections.synapsesForSegment(bestSeg)]} with overlap {overlap}")
    #             activeCells.add(winnerCell)  #bestCell
    #             winnerCells.add(winnerCell)  #bestCell
    #             self.segmentActiveForCell[winnerCell] = bestSeg  #[bestCell]
    #             self.winnerCellForColumn[col] = winnerCell  #bestCell
    #         else:
    #             cells = self._cells_for_column(col)
    #             activeCells.update(cells)
    #             winnerCell = getLeastUsedCell(self.connections, cells, self.lastUsedCell)
    #             winnerCells.add(winnerCell)
    #             self.winnerCellForColumn[col] = winnerCell

    #             self.lastUsedCell = winnerCell

    #     return activeCells, winnerCells

    def _activate_columns(self, activeColumns: List[int]) -> Tuple[Set[int], Set[int]]:
        if not activeColumns:
            return set(), set()

        winnerCells = set()
        activeCells = set()

        for col in activeColumns:
            columnCells = self._cells_for_column(col)
            predictiveCells = [c for c in columnCells if c in self.prevPredictiveCells]

            if predictiveCells:
                # ✅ Use best matching predictive cell as winner
                winnerCell = getBestMatchingCell(
                    self.connections, predictiveCells, self.prevActiveCells, self.minThreshold
                )

                # ✅ Activate only that cell
                activeCells.add(winnerCell)
                winnerCells.add(winnerCell)

                bestSeg, _ = getBestMatchingSegment(
                    self.connections, winnerCell, self.prevActiveCells,
                    self.activationThreshold, self.connectedPermanence, return_overlap=True
                )

                self.segmentActiveForCell[winnerCell] = bestSeg
                self.winnerCellForColumn[col] = winnerCell
            else:
                # Burst column: all cells become active
                activeCells.update(columnCells)

                # Pick least-used cell as winner
                winnerCell = getLeastUsedCell(self.connections, columnCells, self.lastUsedCell)
                winnerCells.add(winnerCell)
                self.winnerCellForColumn[col] = winnerCell
                self.lastUsedCell = winnerCell

        return activeCells, winnerCells

    def _learn_segments(self, activeColumns: List[int], prevWinnerCells: Set[int]) -> None:
        context_key = frozenset(prevWinnerCells)

        for col in activeColumns:
            learningCell = self.winnerCellForColumn.get(col)
            if learningCell is None or not prevWinnerCells:
                continue

            # Allow learning even if cell was not predicted
            if learningCell not in self.prevPredictiveCells:
                logger.debug(f"[LEARN] Cell {learningCell} was not predicted — learning on burst cell")

            # 1️⃣ Exact context match → reuse known segment
            if learningCell in self.contextMemory[context_key]:
                segment = self.contextMemory[context_key][learningCell]
                logger.debug(f"[REUSE] Exact context → reusing segment {segment} on cell {learningCell}")
                active_synapses = [
                    s for s in self.connections.synapsesForSegment(segment)
                    if self.connections.dataForSynapse(s).presynapticCell in prevWinnerCells
                ]
                self.adaptSegment(
                    self.connections, segment, active_synapses,
                    positiveReinforcement=True,
                    permanenceIncrement=self.permanenceIncrement,
                    permanenceDecrement=self.permanenceDecrement
                )
                self.segmentActiveForCell[learningCell] = segment
                continue

            # 2️⃣ Best-matching segment with sufficient overlap → reuse even if context differs
            bestSegment, overlap = getBestMatchingSegment(
                self.connections, learningCell, prevWinnerCells,
                self.minThreshold, self.connectedPermanence, return_overlap=True
            )

            if bestSegment is not None and overlap >= self.minThreshold:
                logger.debug(f"[REUSE] Partial context → reusing segment {bestSegment} (overlap={overlap}) on cell {learningCell}")
                active_synapses = [
                    s for s in self.connections.synapsesForSegment(bestSegment)
                    if self.connections.dataForSynapse(s).presynapticCell in prevWinnerCells
                ]
                self.adaptSegment(
                    self.connections, bestSegment, active_synapses,
                    positiveReinforcement=True,
                    permanenceIncrement=self.permanenceIncrement,
                    permanenceDecrement=self.permanenceDecrement
                )
                self.segmentActiveForCell[learningCell] = bestSegment
                self.contextMemory[context_key][learningCell] = bestSegment
                continue

            # 3️⃣ No reusable segment → grow new one
            logger.debug(f"[GROW] Creating new segment on cell {learningCell} for context {sorted(prevWinnerCells)}")
            newSegment = self.connections.createSegment(learningCell, sequence=True)
            growSynapsesToSegment(
                self.connections,
                newSegment,
                sorted(prevWinnerCells),
                initialPermanence=self.initialPermanence,
                permanenceBoost=0.0
            )
            self.segmentActiveForCell[learningCell] = newSegment
            self.contextMemory[context_key][learningCell] = newSegment

    def _predict_cells(self):
        self.predictiveCells.clear()

        for cell in range(self.numCells):
            for segment in self.connections.segmentsForCell(cell):
                # 🔒 Only use sequence segments to drive prediction
                if not self.connections.segmentIsSequence(segment):
                    continue

                synapses = self.connections.synapsesForSegment(segment)
                active_count = sum(
                    1 for s in synapses
                    if self.connections.dataForSynapse(s).presynapticCell in self.prevActiveCells and
                    self.connections.dataForSynapse(s).permanence >= self.connectedPermanence
                )
                if active_count >= self.activationThreshold:
                    self.predictiveCells.add(cell)
                    self.predictedSegmentForCell[cell] = segment
                    logger.debug(
                        f"[PREDICT] cell {cell} ← segment {segment}, "
                        f"overlap={active_count}, from {[self.connections.dataForSynapse(s).presynapticCell for s in synapses if self.connections.dataForSynapse(s).presynapticCell in self.prevActiveCells]}"
                    )
                    break

    def adaptSegment(self, conn, segment, activeSynapses, positiveReinforcement,
                    permanenceIncrement=0.05, permanenceDecrement=0.05):
        logger.debug(f"[ADAPT] Adapting segment {segment} with {len(activeSynapses)} active synapses")
        all_synapses = conn.synapsesForSegment(segment)
        active_set = set(activeSynapses)

        for synapse in active_set:
            if synapse not in all_synapses:
                raise KeyError(f"Synapse {synapse} is not on segment {segment}")

        for synapse in all_synapses:
            data = conn.dataForSynapse(synapse)
            permanence = data.permanence

            if synapse in active_set:
                delta = permanenceIncrement if positiveReinforcement else -permanenceDecrement
            else:
                delta = -permanenceDecrement if positiveReinforcement else 0.0

            conn.updateSynapsePermanence(synapse, delta)

    def _adapt_segment(
        self,
        connections,
        segment,
        activePresynapticCells,
        newSynapseCount,
        increment,
        decrement
    ):
        """
        Adjust permanence of synapses on a segment based on active presynaptic cells.
        """
        synapses = connections.synapsesForSegment(segment)
        for syn in synapses:
            data = connections.dataForSynapse(syn)
            if data.presynapticCell in activePresynapticCells:
                delta = increment
            else:
                delta = -decrement
            connections.updateSynapsePermanence(syn, delta)

        # Optionally grow new synapses
        if newSynapseCount > 0:
            existing = {connections.dataForSynapse(s).presynapticCell for s in synapses}
            potential = list(activePresynapticCells - existing)
            new_cells = potential[:newSynapseCount]
            for cell in new_cells:
                connections.createSynapse(segment, cell, self.initialPermanence)
            logger.debug(f"[TemporalMemory._adapt_segment] segment: {segment} new_cells: {new_cells}")

    def _set_if_unassigned(self, d: dict, key, value):
        if key not in d or d[key] is None:
            d[key] = value

    def reset(self):
        self.activeCells.clear()
        self.winnerCells.clear()
        self.predictiveCells.clear()
        self.prevActiveCells.clear()
        self.prevWinnerCells.clear()
        self.prevPredictiveCells.clear()
        self.activeSegments.clear()
        self.segmentActiveForCell.clear()
        self.winnerCellForColumn.clear()
        self.lastUsedCell = -1
        # Do not clear self.contextMemory (long-term context persistence)


# Helper functions
def getBestMatchingCell(connections, columnCells, activePresynapticCells, minThreshold):
    if not isinstance(columnCells, list):
        raise TypeError("columnCells must be a list")
    if not all(isinstance(c, int) for c in columnCells):
        raise TypeError("All elements of columnCells must be integers")

    best_cell = None
    best_overlap = -1

    for cell in columnCells:
        for segment in connections.segmentsForCell(cell):
            overlap = sum(
                1 for s in connections.synapsesForSegment(segment)
                if connections.dataForSynapse(s).presynapticCell in activePresynapticCells
            )
            if overlap >= minThreshold and overlap > best_overlap:
                best_overlap = overlap
                best_cell = cell

    return best_cell if best_cell is not None else getLeastUsedCell(connections, columnCells)

def getBestMatchingSegment(connections, cell: int, activePresynapticCells: Set[int],
                           minThreshold: int, connectedPermanence: float, return_overlap=False):
    """
    Faithful port of Numenta's TemporalMemory::getBestMatchingSegment_ logic.
    Finds the segment on the given cell with the highest number of connected synapses
    that are connected to currently active presynaptic cells.
    """
    if not isinstance(activePresynapticCells, set):
        raise TypeError("activePresynapticCells must be a set")

    if not isinstance(minThreshold, int) or minThreshold < 0:
        raise ValueError("minThreshold must be a non-negative integer")

    bestSegment = None
    bestOverlap = -1

    for segment in connections.segmentsForCell(cell):
        overlap = 0
        for synapse in connections.synapsesForSegment(segment):
            data = connections.dataForSynapse(synapse)
            if data.permanence >= connectedPermanence and data.presynapticCell in activePresynapticCells:
                overlap += 1

        if overlap > bestOverlap:
            bestSegment = segment
            bestOverlap = overlap

    logger.debug(f"[MATCH] Cell {cell}, bestSegment={bestSegment}, bestOverlap={bestOverlap}")
    if bestSegment is None or bestOverlap < minThreshold:
        return (None, bestOverlap) if return_overlap else None
    else:
        return (bestSegment, bestOverlap) if return_overlap else bestSegment

def getLeastUsedCell(connections, columnCells, lastUsedCell=None):
    """
    Return the cell in columnCells with the fewest segments.
    If there's a tie, prefer the one after `lastUsedCell` in sorted order.
    """
    if not all(isinstance(c, int) for c in columnCells):
        raise ValueError("All cells must be integers")
    if not all(0 <= c < connections.numCells() for c in columnCells):
        raise ValueError("All cells must be within valid range")

    # Map each cell to its segment count
    segment_counts = {cell: len(connections.segmentsForCell(cell)) for cell in columnCells}
    min_segments = min(segment_counts.values())
    candidates = [cell for cell, count in segment_counts.items() if count == min_segments]

    if lastUsedCell in candidates:
        sorted_candidates = sorted(candidates)
        idx = sorted_candidates.index(lastUsedCell)
        return sorted_candidates[(idx + 1) % len(sorted_candidates)]
    else:
        return min(candidates)

def growSynapsesToSegment(connections, segment, presynapticCells, initialPermanence, permanenceBoost=0.0):
    if not isinstance(initialPermanence, (float, int)) or not (0.0 <= initialPermanence <= 1.0):
        raise ValueError("initialPermanence must be a float between 0 and 1")

    existing = {connections.dataForSynapse(s).presynapticCell
                for s in connections.synapsesForSegment(segment)}

    new_synapses = []
    for cell in presynapticCells:
        if cell in existing:
            continue
        perm = min(initialPermanence + permanenceBoost, 1.0)  # FIXED: no hardcoded value
        synapse = connections.createSynapse(segment, cell, perm)
        new_synapses.append(synapse)

    logger.debug(f"[TemporalMemory.growSynapsesToSegment] segment: {segment} new_synapses: {new_synapses}")

    return new_synapses


__all__ = [
    "getBestMatchingCell",
    "getBestMatchingSegment",
    "getLeastUsedCell",
    "growSynapsesToSegment"
]

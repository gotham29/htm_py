import numpy as np
# import logging
from typing import Set, List, Tuple, Union
from collections import defaultdict
from htm_py.connections import Connections

# logger = logging.getLogger("htm_py.tm")
# logger.setLevel(logging.DEBUG)

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
        print(f"\n[TM] >>> compute(activeColumns={activeColumns}, learn={learn})")
        self.learn = learn
        self.activeColumns = activeColumns
        self.activeCells, self.winnerCells = self._activate_columns(activeColumns)
        # self._activate_columns(activeColumns)
        print(f"[TM] ActiveCells after _activate_columns: {sorted(self.activeCells)}")
        print(f"[TM] WinnerCells after _activate_columns: {sorted(self.winnerCells)}")

        if learn:
            self._learn_segments(self.activeColumns, self.prevWinnerCells)

        # 🔥 Fix: move this BEFORE prediction
        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)

        print(f"[DEBUG] prevActiveCells before prediction: {sorted(self.prevActiveCells)}")
        self._predict_cells()
        print(f"[TM] PredictiveCells after _predict_cells: {sorted(self.predictiveCells)}")

        anomaly_score = self.calculate_anomaly_score()
        prediction_count = self.calculate_prediction_count()
        print(f"[TM] Anomaly Score: {anomaly_score:.3f}, Prediction Count: {prediction_count:.3f}")

        # Finalize timestep state
        self.prevPredictiveCells = set(self.predictiveCells)
        print("[DEBUG] active_synapses function found? ->", hasattr(self.connections, "active_synapses"))

        return anomaly_score, prediction_count

    def calculate_anomaly_score(self):
        if not self.winnerCells:
            print("[ANOMALY] No winner cells → anomaly score = 0.0")
            return 0.0

        correctly_predicted = self.winnerCells & self.prevPredictiveCells
        score = 1.0 - len(correctly_predicted) / len(self.winnerCells)
        print(f"[ANOMALY] WinnerCells: {self.winnerCells}, PrevPredicted: {self.prevPredictiveCells}")
        print(f"[ANOMALY] CorrectlyPredicted: {correctly_predicted} → anomaly score = {score:.3f}")
        return score

    def calculate_prediction_count(self):
        if not self.activeColumns:
            return 0.0  # Avoid divide-by-zero when no active columns
        return len(self.prevPredictiveCells) / len(self.activeColumns)

    def _cells_for_column(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def _activate_columns(self, activeColumns: List[int]) -> Tuple[Set[int], Set[int]]:
        if not activeColumns:
            return set(), set()

        winnerCells = set()
        activeCells = set()

        for col in activeColumns:
            predictedCells = [
                c for c in self._cells_for_column(col)
                if c in self.prevPredictiveCells
            ]
            if predictedCells:
                winnerCell = predictedCells[0]
                bestSeg, overlap = getBestMatchingSegment(
                    self.connections, winnerCell, self.prevActiveCells,
                    self.activationThreshold, self.connectedPermanence, return_overlap=True
                )
                if bestSeg is not None:
                    print(f"[ACTIVATE] Best segment for cell {winnerCell}: {bestSeg} from sources {[self.connections.dataForSynapse(s).presynapticCell for s in self.connections.synapsesForSegment(bestSeg)]} with overlap {overlap}")
                activeCells.add(winnerCell)  #bestCell
                winnerCells.add(winnerCell)  #bestCell
                self.segmentActiveForCell[winnerCell] = bestSeg  #[bestCell]
                self.winnerCellForColumn[col] = winnerCell  #bestCell
            else:
                cells = self._cells_for_column(col)
                activeCells.update(cells)
                winnerCell = getLeastUsedCell(self.connections, cells, self.lastUsedCell)
                winnerCells.add(winnerCell)
                self.winnerCellForColumn[col] = winnerCell

                self.lastUsedCell = winnerCell

        return activeCells, winnerCells

    def _predict_cells(self):
        self.predictiveCells.clear()
        # self.segmentActiveForCell.clear()

        for cell in range(self.numCells):
            segments = self.connections.segmentsForCell(cell)
            for segment in segments:
                synapses = self.connections.synapsesForSegment(segment)
                active_count = sum(
                    1 for s in synapses
                    if self.connections.dataForSynapse(s).presynapticCell in self.prevActiveCells and
                    self.connections.dataForSynapse(s).permanence >= self.connectedPermanence
                )
                if active_count >= self.activationThreshold:
                    syns = self.connections.synapsesForSegment(segment)
                    srcs = [self.connections.dataForSynapse(s).presynapticCell for s in syns]
                    print(f"[PREDICT] cell {cell} predicted by segment {segment}, active count = {active_count}, sources = {sorted(srcs)}")
                    self.predictiveCells.add(cell)
                    # self.segmentActiveForCell[cell] = segment
                    self.predictedSegmentForCell[cell] = segment
                    break  # stop after first matching segment

    def _learn_segments(self, activeColumns: List[int], prevWinnerCells: Set[int]) -> None:
        print(f"[LEARN] prevWinnerCells: {sorted(prevWinnerCells)}")
        for col in activeColumns:
            learningCell = self.winnerCellForColumn.get(col)
            if learningCell is None:
                continue

            if not prevWinnerCells:
                print(f"[LEARN] Skipping learning for column {col} (no prevWinnerCells)")
                continue

            matchingSegment = None
            overlap = -1

            if learningCell in self.prevPredictiveCells and learningCell in self.segmentActiveForCell:
                matchingSegment = self.segmentActiveForCell[learningCell]
                overlap = self.minThreshold  # Allow reuse
                print(f"[LEARN] Reusing existing matching segment {matchingSegment} on cell {learningCell}")
            else:
                matchingSegment, overlap = getBestMatchingSegment(
                    self.connections, learningCell, prevWinnerCells,
                    self.minThreshold, self.connectedPermanence, return_overlap=True)
                print(f"[LEARN] Best matching segment: {matchingSegment} (overlap={overlap})")

            # 🚨 Force new segment if context differs
            if matchingSegment is not None and overlap >= self.minThreshold:
                existing_sources = {
                    self.connections.dataForSynapse(s).presynapticCell
                    for s in self.connections.synapsesForSegment(matchingSegment)
                }
                if not existing_sources.issuperset(prevWinnerCells):
                    print(f"[LEARN] Context mismatch detected — growing new segment instead of reusing")
                    matchingSegment = None

            if matchingSegment is not None and overlap >= self.minThreshold:
                active_synapses = [
                    s for s in self.connections.synapsesForSegment(matchingSegment)
                    if self.connections.dataForSynapse(s).presynapticCell in prevWinnerCells
                ]
                print(f"[LEARN] Adapting segment {matchingSegment} with {len(active_synapses)} active synapses")
                self.adaptSegment(
                    self.connections, matchingSegment, active_synapses,
                    positiveReinforcement=True,
                    permanenceIncrement=self.permanenceIncrement,
                    permanenceDecrement=self.permanenceDecrement
                )
                self.segmentActiveForCell[learningCell] = matchingSegment
            else:
                print(f"[LEARN] Creating new segment on cell {learningCell} (column {col})")
                newSegment = self.connections.createSegment(learningCell, sequence=True)
                print(f"[LEARN] Growing synapses from presynaptic cells: {sorted(prevWinnerCells)}")
                syns = growSynapsesToSegment(
                    self.connections,
                    newSegment,
                    list(prevWinnerCells),
                    initialPermanence=self.initialPermanence,
                    permanenceBoost=0.1
                )
                print(f"[LEARN] Grown {len(syns)} synapses on segment {newSegment}")
                self.segmentActiveForCell[learningCell] = newSegment

    def adaptSegment(self, conn, segment, activeSynapses, positiveReinforcement,
                    permanenceIncrement=0.05, permanenceDecrement=0.05):
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

# def getBestMatchingSegment(connections, cell: int, activeSynapses: Set[int], minThreshold: int, connectedPermanence: float, return_overlap=False):
#     if not isinstance(activeSynapses, set):
#         raise TypeError(f"Expected activeSynapses to be a set, got {type(activeSynapses).__name__}")
#     if not isinstance(minThreshold, int) or minThreshold < 0:
#         raise ValueError(f"minThreshold must be a non-negative integer, got {minThreshold}")

#     bestSegment = None
#     bestOverlap = -1

#     for seg in connections.segmentsForCell(cell):
#         synapses = connections.synapsesForSegment(seg)
#         overlap = sum(1 for s in synapses
#                       if connections.dataForSynapse(s).presynapticCell in activeSynapses and
#                          connections.dataForSynapse(s).permanence >= connectedPermanence)
#         if overlap > bestOverlap:
#             bestSegment = seg
#             bestOverlap = overlap

#     if return_overlap:
#         return bestSegment, bestOverlap
#     else:
#         return bestSegment

def getBestMatchingSegment(connections, cell: int, activeSynapses: Set[int], minThreshold: int, connectedPermanence: float, return_overlap=False):
    if not isinstance(activeSynapses, set):
        raise TypeError(f"Expected activeSynapses to be a set, got {type(activeSynapses).__name__}")
    if not isinstance(minThreshold, int) or minThreshold < 0:
        raise ValueError(f"minThreshold must be a non-negative integer, got {minThreshold}")

    bestSegment = None
    bestOverlap = -1

    for seg in connections.segmentsForCell(cell):
        synapses = connections.synapsesForSegment(seg)
        overlap = sum(1 for s in synapses
                      if connections.dataForSynapse(s).presynapticCell in activeSynapses and
                         connections.dataForSynapse(s).permanence >= connectedPermanence)
        if overlap > bestOverlap:
            bestSegment = seg
            bestOverlap = overlap

    if return_overlap:
        return bestSegment, bestOverlap
    else:
        return bestSegment if bestOverlap >= minThreshold else None

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

    return new_synapses


__all__ = [
    "getBestMatchingCell",
    "getBestMatchingSegment",
    "getLeastUsedCell",
    "growSynapsesToSegment"
]

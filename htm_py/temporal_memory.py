import numpy as np
# import logging
from typing import Union
from typing import Set, List
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
        self._activate_columns(activeColumns)
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
        prediction_count = len(self.prevPredictiveCells) / len(self.activeColumns)
        return prediction_count

    def _cells_for_column(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def _activate_columns(self, activeColumns):
        self.activeCells.clear()
        self.winnerCells.clear()
        self.winnerCellForColumn.clear()

        for column in activeColumns:
            column_cells = self._cells_for_column(column)
            predictive = set(column_cells).intersection(self.prevPredictiveCells)

            if predictive:
                # Predicted input — use predictive cell
                for cell in predictive:
                    self.activeCells.add(cell)
                    self.winnerCells.add(cell)
                self.winnerCellForColumn[column] = next(iter(predictive))
            else:
                # Burst: all cells become active
                for cell in column_cells:
                    self.activeCells.add(cell)

                # Try to reuse a cell with a matching segment
                best_cell = None
                best_segment = None
                best_score = -1

                for cell in column_cells:
                    segment, score = getBestMatchingSegment(
                        self.connections,
                        cell,
                        self.prevWinnerCells,
                        self.minThreshold,
                        return_overlap=True
                    )
                    if segment is not None and score > best_score:
                        best_cell = cell
                        best_segment = segment
                        best_score = score

                if best_segment is not None and best_score >= self.minThreshold:
                    self.winnerCells.add(best_cell)
                    self.winnerCellForColumn[column] = best_cell
                else:
                    # cell = getLeastUsedCell(self.connections, column_cells)
                    cell = getLeastUsedCell(self.connections, column_cells, self.lastUsedCell)
                    self.lastUsedCell = cell
                    self.winnerCells.add(cell)
                    self.winnerCellForColumn[column] = cell

    def _predict_cells(self):
        """Evaluate segments and mark predictive cells."""
        self.predictiveCells.clear()

        for cell in range(self.numCells):
            for seg in self.connections.segmentsForCell(cell):
                active = 0
                for syn in self.connections.synapsesForSegment(seg):
                    syn_data = self.connections.dataForSynapse(syn)
                    if syn_data.permanence >= self.connectedPermanence and syn_data.presynapticCell in self.prevActiveCells:
                        active += 1

                if active >= self.activationThreshold:
                    self.predictiveCells.add(cell)

                    print(f"[TM Predict] Cell {cell} → predictive via segment {seg} with {active} active synapses")
                else:
                    print(f"[TM Predict] Cell {cell} NOT predictive (only {active} active synapses on segment {seg})")

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
            if learningCell in self.prevPredictiveCells and learningCell in self.segmentActiveForCell:
                matchingSegment = self.segmentActiveForCell[learningCell]
                print(f"[LEARN] Reusing existing matching segment {matchingSegment} on cell {learningCell}")
            else:
                matchingSegment, overlap = getBestMatchingSegment(
                    self.connections, learningCell, prevWinnerCells,
                    self.minThreshold, return_overlap=True)
                print(f"[LEARN] Best matching segment: {matchingSegment} (overlap={overlap})")

            if matchingSegment is not None:
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
                # delta = -permanenceDecrement if positiveReinforcement else 0.0
                if positiveReinforcement:
                    delta = -permanenceDecrement
                else:
                    delta = 0.0
            new_perm = min(max(permanence + delta, 0.0), 1.0)
            conn.updateSynapsePermanence(synapse, new_perm)

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
                new_perm = min(1.0, data.permanence + increment)
            else:
                new_perm = max(0.0, data.permanence - decrement)
            connections.updateSynapsePermanence(syn, new_perm)

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

def getBestMatchingSegment(connections, cell, activePresynapticCells, minThreshold, return_overlap=False):
    if not isinstance(activePresynapticCells, set):
        raise TypeError("activePresynapticCells must be a set")
    if not isinstance(minThreshold, int) or minThreshold < 0:
        raise ValueError("minThreshold must be a non-negative integer")
    
    best_segment = None
    best_overlap = -1

    for segment in connections.segmentsForCell(cell):
        overlap = sum(
            1 for s in connections.synapsesForSegment(segment)
            if connections.dataForSynapse(s).presynapticCell in activePresynapticCells
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment

    if return_overlap:
        return (best_segment, best_overlap)

    # Enforce minThreshold before returning
    if best_segment is not None and best_overlap >= minThreshold:
        return best_segment
    return None

def getLeastUsedCell(connections, columnCells, lastUsedCell=None):
    """
    Returns the cell with the fewest segments in the given column.
    If there's a tie, rotate from the previously used winner cell.
    """
    if not all(isinstance(c, int) for c in columnCells):
        raise ValueError("All cells must be integers")
    if not all(0 <= c < connections.numCells() for c in columnCells):
        raise ValueError("All cells must be within valid range")

    min_segments = float('inf')
    candidates = []

    for cell in columnCells:
        seg_count = len(connections.segmentsForCell(cell))
        if seg_count < min_segments:
            min_segments = seg_count
            candidates = [cell]
        elif seg_count == min_segments:
            candidates.append(cell)

    # Rotate selection from lastUsedCell if provided
    if lastUsedCell is not None and lastUsedCell in columnCells:
        sorted_candidates = sorted(candidates)
        idx = sorted_candidates.index(lastUsedCell) if lastUsedCell in sorted_candidates else -1
        rotated = sorted_candidates[(idx + 1) % len(sorted_candidates)]
        return rotated

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

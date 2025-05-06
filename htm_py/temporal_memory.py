import numpy as np
# import logging
from typing import Union
from typing import Set, List
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
        self.learn = learn
        self.activeColumns = activeColumns
        self._activate_columns()
        if learn:
            self._learn_segments(self.activeColumns, self.prevWinnerCells)
        self._predict_cells()

        # logger.debug(f"[COMPUTE] Winner cells: {sorted(self.winnerCells)}")
        # logger.debug(f"[COMPUTE] Predictive cells: {sorted(self.predictiveCells)}")

        anomaly_score = self.calculate_anomaly_score()
        prediction_count = self.calculate_prediction_count()

        # SAVE STATE for next step
        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)
        self.prevPredictiveCells = set(self.predictiveCells)

        return anomaly_score, prediction_count

    def calculate_anomaly_score(self):
        # Build set of columns that were predicted last timestep
        predicted_columns_last_step = {
            cell // self.cellsPerColumn for cell in self.prevPredictiveCells
        }

        # Count how many currently active columns were predicted
        num_predicted_columns = sum(
            1 for col in self.activeColumns if col in predicted_columns_last_step
        )

        # Total number of active columns
        num_active_columns = len(self.activeColumns)

        # Compute anomaly score
        anomaly_score = 1.0 - (num_predicted_columns / num_active_columns)

        return anomaly_score

    def calculate_prediction_count(self):
        prediction_count = len(self.prevPredictiveCells) / len(self.activeColumns)
        return prediction_count

    def _cells_for_column(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def _activate_columns(self):
        self.activeCells = set()
        self.winnerCells = set()
        self.winnerCellForColumn = {}

        for col in self.activeColumns:
            column_cells = self._cells_for_column(col)
            predicted_cells = [c for c in column_cells if c in self.prevPredictiveCells]

            if predicted_cells:
                # If any cell in this column was predicted, pick one as winner
                cell = predicted_cells[0]
                self.activeCells.add(cell)
                self.winnerCells.add(cell)
                self.winnerCellForColumn[col] = cell
            else:
                # No predicted cells: column bursts
                self.activeCells.update(column_cells)

                if self.learn:
                    # Select a cell to grow a segment from (winner cell)
                    cell = getLeastUsedCell(self.connections, column_cells)
                    self.winnerCells.add(cell)
                    self.winnerCellForColumn[col] = cell

                    # 🔥 PATCH: Ensure segment growth will happen
                    # Force segment creation by marking no matching segment
                    if cell not in self.prevPredictiveCells:
                        self.segmentActiveForCell[cell] = None

    def _predict_cells(self):
        self.predictiveCells = set()
        self.activeSegments = set()
        self.segmentActiveForCell = {}

        for segment in self.connections.segments():
            # logger.debug(f"[PREDICT] Checking segment {segment}")
            active_syns = self.connections.active_synapses(segment, self.activeCells, self.connectedPermanence)
            # logger.debug(f"[PREDICT] Segment {segment}: {len(active_syns)} active synapses")
            if len(active_syns) >= self.activationThreshold:
                cell = self.connections.cell_for_segment(segment)
                self.predictiveCells.add(cell)
                self.activeSegments.add(segment)
                self.segmentActiveForCell[cell] = segment

    def _learn_segments(self, activeColumns: List[int], prevWinnerCells: Set[int]) -> None:
        for col in activeColumns:
            learningCell = self.winnerCellForColumn.get(col)
            if learningCell is None:
                continue  # Skip if no winner cell for this column
            
            # logger.debug(f"[LEARN] Column {col} → LearningCell {learningCell}")
            matchingSegment = None
            if learningCell in self.prevPredictiveCells and learningCell in self.segmentActiveForCell:
                matchingSegment = self.segmentActiveForCell[learningCell]
                overlap = self.activationThreshold  # Assume it's active
            else:
                matchingSegment, overlap = getBestMatchingSegment(
                    self.connections, learningCell, prevWinnerCells, self.minThreshold, return_overlap=True)
            
            if matchingSegment is not None:
                # Adapt existing matching segment
                active_synapses = [
                    s for s in self.connections.synapsesForSegment(matchingSegment)
                    if self.connections.dataForSynapse(s).presynapticCell in prevWinnerCells
                ]
                self.adaptSegment(self.connections, matchingSegment, active_synapses,
                                positiveReinforcement=True,
                                permanenceIncrement=self.permanenceIncrement,
                                permanenceDecrement=self.permanenceDecrement)
                self.segmentActiveForCell[learningCell] = matchingSegment
            else:
                # Create new segment if no match found
                newSegment = self.connections.createSegment(learningCell)
                growSynapsesToSegment(
                    self.connections,
                    newSegment,
                    list(prevWinnerCells),
                    initialPermanence=self.connectedPermanence,
                    permanenceBoost=0.1
                )
                self.segmentActiveForCell[learningCell] = newSegment

    def reset(self):
        self.activeCells.clear()
        self.predictiveCells.clear()
        self.winnerCells.clear()
        self.prevActiveCells.clear()
        self.prevWinnerCells.clear()

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

            new_perm = min(max(permanence + delta, 0.0), 1.0)
            conn.updateSynapsePermanence(synapse, new_perm)

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

def getLeastUsedCell(connections, columnCells):
    if any(not isinstance(c, int) or c < 0 or c >= connections.numCells() for c in columnCells):
        raise ValueError("columnCells must contain valid cell indices")

    # Sort by (num_segments, cell_id) so ties go to lowest-indexed cell
    return min(columnCells, key=lambda c: (len(connections.segmentsForCell(c)), c))

def growSynapsesToSegment(connections, segment, presynapticCells, initialPermanence, permanenceBoost=0.0):
    if not isinstance(initialPermanence, (float, int)) or not (0.0 <= initialPermanence <= 1.0):
        raise ValueError("initialPermanence must be a float between 0 and 1")

    existing = {connections.dataForSynapse(s).presynapticCell
                for s in connections.synapsesForSegment(segment)}

    new_synapses = []
    for cell in presynapticCells:
        if cell in existing:
            continue
        perm = min(initialPermanence + permanenceBoost, 1.0)
        # logger.debug(f"[GROW] Creating synapse to presynapticCell {cell} with permanence={perm}")
        synapse = connections.createSynapse(segment, cell, perm)
        new_synapses.append(synapse)

    return new_synapses


__all__ = [
    "getBestMatchingCell",
    "getBestMatchingSegment",
    "getLeastUsedCell",
    "growSynapsesToSegment"
]

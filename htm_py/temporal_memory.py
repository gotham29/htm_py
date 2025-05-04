# import logging
# logging.basicConfig(
#     filename="tm_debug.log",
#     level=logging.DEBUG,
#     filemode='w',
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     force=True  # <- REQUIRED under pytest or if logger was already initialized
# )

# import numpy as np
# from htm_py.connections import Connections


# class TemporalMemory:
#     def __init__(self, columnDimensions, cellsPerColumn,
#                  activationThreshold, initialPermanence, connectedPermanence,
#                  minThreshold, maxNewSynapseCount,
#                  permanenceIncrement, permanenceDecrement,
#                  predictedSegmentDecrement):

#         # --------------------------
#         # CORE MODEL CONFIGURATION
#         # --------------------------
#         self.columnDimensions = columnDimensions
#         self.cellsPerColumn = cellsPerColumn
#         self.numColumns = np.prod(columnDimensions)

#         # Synapse permanence & thresholds
#         self.initialPermanence = initialPermanence
#         self.connectedPermanence = connectedPermanence
#         self.activationThreshold = activationThreshold
#         self.minThreshold = minThreshold
#         self.maxNewSynapseCount = maxNewSynapseCount
#         self.permanenceIncrement = permanenceIncrement
#         self.permanenceDecrement = permanenceDecrement
#         self.predictedSegmentDecrement = predictedSegmentDecrement

#         # --------------------------
#         # CELL STATE
#         # --------------------------
#         self.activeCells = set()
#         self.prevActiveCells = set()
#         self.winnerCells = set()
#         self.prevWinnerCells = set()
#         self.predictiveCells = set()
#         self.prevPredictiveCells = set()
#         self.burstColumns = set()

#         # --------------------------
#         # TEMPORAL TRACKING
#         # --------------------------
#         self.iteration = 0
#         self.learn = True
#         self.activeColumns = []
#         self.predictedColumns = set()

#         # --------------------------
#         # CONNECTIVITY STRUCTURE
#         # --------------------------
#         self.connections = Connections(self.numColumns * self.cellsPerColumn)

#         # --------------------------
#         # OUTPUTS
#         # --------------------------
#         self.anomaly_score = None
#         self.prediction_count = None

#     def compute(self, activeColumns, learn):
#         self.iteration += 1
#         self.learn = learn
#         self.activeColumns = activeColumns
#         self.burstColumns = set()
#         self.activeCells = set()
#         self.winnerCells = set()

#         # Update predictedColumns before activation
#         self.predictedColumns = {
#             self.connections.column_for_cell(c) for c in self.prevPredictiveCells
#         }

#         self._activate_columns()

#         if learn:
#             self._learn_segments()

#         self._predict_cells()
#         self._calculate_anomaly_score()
#         self._calculate_prediction_count()
#         self._update_state()

#         for handler in logging.getLogger().handlers:
#             handler.flush()

#         return self.anomaly_score, self.prediction_count

#     def _activate_columns(self):
#         for col in self.activeColumns:
#             best_cell, best_seg = self._get_best_matching_segment(col)
#             logging.debug(f"[Activate] Column {col}: best_cell={best_cell}, best_seg={best_seg}")
#             if best_seg and self._num_active_connected_synapses(best_seg, self.prevActiveCells) >= self.activationThreshold:
#                 self.activeCells.add(best_cell)
#                 winner = self._pick_winner_cell_for_segment(best_seg)
#                 self.winnerCells.add(winner)
#                 logging.debug(f"[Activate] Predictive column {col}, winner={winner}")
#             else:
#                 self.burstColumns.add(col)
#                 logging.debug(f"[Activate] Bursting column {col}")
#                 self._burst_column(col)

#     def _burst_column(self, column):
#         for cell in self._cells_for_column(column):
#             self.activeCells.add(cell)

#         best_cell, best_seg = self._get_best_matching_segment(column)
#         if best_seg:
#             winner = self._pick_winner_cell_for_segment(best_seg)
#             self.winnerCells.add(winner)
#             logging.debug(f"[Burst] Column {column}: winner from best segment: {winner}")
#         else:
#             fallback = self._get_least_used_cell(column)
#             self.winnerCells.add(fallback)
#             logging.debug(f"[Burst] Column {column}: no matching segment, fallback winner={fallback}")

#     def _cells_for_column(self, column):
#         start = column * self.cellsPerColumn
#         return list(range(start, start + self.cellsPerColumn))

#     def _get_best_matching_segment(self, column):
#         """
#         Returns (cell, segment) with the best matching segment on a cell in the column.
#         A segment is considered matching if it has >= minThreshold active connected synapses.
#         If no such segment exists, returns (None, None).
#         """
#         best_cell = None
#         best_segment = None
#         max_overlap = self.minThreshold

#         for cell in self._cells_for_column(column):
#             segments = self.connections.segments_for_cell(cell)
#             for seg in segments:
#                 count = self._num_active_connected_synapses(seg, self.prevActiveCells)
#                 logging.debug(f"[BestMatch] Column={column} Cell={cell} Segment={seg} ActiveConnected={count}")
#                 if count >= max_overlap:
#                     max_overlap = count
#                     best_cell = cell
#                     best_segment = seg

#         if best_cell is None:
#             logging.debug(f"[BestMatch] Column={column} has no segment meeting minThreshold={self.minThreshold}")
#         else:
#             logging.debug(f"[BestMatch] Column={column} selected Cell={best_cell} Segment={best_segment} with overlap={max_overlap}")

#         return best_cell, best_segment

#     def _num_active_connected_synapses(self, segment, prev_cells):
#         active = self.connections.active_synapses(
#             segment, prev_cells, connected_only=True
#         )
#         return len(active)

#     def _pick_winner_cell_for_segment(self, segment):
#         return self.connections.cell_for_segment(segment)

#     def _get_least_used_cell(self, column):
#         min_segments = float('inf')
#         best_cell = None
#         for cell in self._cells_for_column(column):
#             seg_count = len(self.connections.segments_for_cell(cell))
#             if seg_count < min_segments:
#                 min_segments = seg_count
#                 best_cell = cell
#         return best_cell

#     def _learn_segments(self):
#         logging.debug(f"[Learn] prevActiveCells={self.prevActiveCells}")
#         for column in self.activeColumns:
#             if column in self.predictedColumns:
#                 cell, segment = self._get_best_matching_segment(column)
#                 if cell is None or segment is None:
#                     logging.debug(f"[Learn] Predicted column {column} but no best match found.")
#                     continue
#                 self.winnerCells.add(cell)
#                 if self.learn:
#                     active_syns = self._get_segment_active_synapses(segment, self.prevActiveCells)
#                     self.connections.adapt_segment(segment, active_syns, self.initialPermanence)
#                     logging.debug(f"[Learn] Adapted segment {segment} on cell {cell} with synapses {active_syns}")
#             else:
#                 cell, _ = self._get_best_matching_segment(column)
#                 if cell is None:
#                     logging.debug(f"[Learn] Bursting column {column} but no best match found.")
#                     continue
#                 self.winnerCells.add(cell)
#                 if self.learn:
#                     new_segment = self.connections.create_segment(cell)
#                     syn_sources = list(self.prevActiveCells)
#                     self.connections.create_synapses(new_segment, syn_sources, self.initialPermanence)
#                     logging.debug(f"[Learn] Created new segment {new_segment} on cell {cell} with synapses {syn_sources}")

#     def _get_segment_active_synapses(self, segment, prev_active_cells):
#         syns = self.connections.synapses_for_segment(segment)
#         return [s for s in syns if self.connections.synapse_to_cell(s) in prev_active_cells]

#     def _predict_cells(self):
#         self.predictiveCells = set()

#         for cell in range(self.numColumns * self.cellsPerColumn):
#             segments = self.connections.segments_for_cell(cell)
#             for seg in segments:
#                 if self.connections.segment_active(
#                     seg,
#                     self.winnerCells,
#                     self.activationThreshold,
#                     connected_only=False
#                 ):
#                     self.predictiveCells.add(cell)
#                     break  # no need to check other segments for this cell
#         logging.debug(f"Predicted cells this step: {self.predictiveCells}")

#     def _calculate_anomaly_score(self):
#         if not self.prevPredictiveCells:
#             self.anomaly_score = 1.0
#         else:
#             overlap = self.activeCells & self.prevPredictiveCells
#             score = 1.0 - len(overlap) / len(self.activeCells) if self.activeCells else 0.0
#             self.anomaly_score = score
#             logging.debug(f"[Anomaly] Overlap={overlap}, Score={score}")

#     def _calculate_prediction_count(self):
#         self.prediction_count = len(self.predictiveCells) / len(self.activeColumns)

#     def _update_state(self):
#         self.prevActiveCells = set(self.activeCells)
#         self.prevWinnerCells = set(self.winnerCells)
#         self.prevPredictiveCells = set(self.predictiveCells)



# temporal_memory.py

from collections import defaultdict
import numpy as np
from .connections import Connections

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
        maxSynapsesPerSegment,
        maxSegmentsPerCell,
        permanenceIncrement,
        permanenceDecrement,
        predictedSegmentDecrement,
        seed=None
    ):
        # Spatial layout
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn

        # Learning thresholds
        self.activationThreshold = activationThreshold
        self.minThreshold = minThreshold
        self.maxNewSynapseCount = maxNewSynapseCount
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell

        # Permanence tuning
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.predictedSegmentDecrement = predictedSegmentDecrement

        # Connections: handles segments, synapses, and permanence
        self.connections = Connections(
            columnDimensions=columnDimensions,
            cellsPerColumn=cellsPerColumn,
            initialPermanence=initialPermanence,
            connectedPermanence=connectedPermanence
        )

        # State variables
        self.activeColumns = set()
        self.activeCells = set()
        self.winnerCells = set()
        self.predictedColumns = set()
        self.predictiveCells = set()
        self.activeSegments = set()
        self.learningSegments = set()

        # Previous timestep
        self.prevActiveCells = set()
        self.prevWinnerCells = set()
        self.prevPredictiveCells = set()

        if seed is not None:
            np.random.seed(seed)

    def compute(self, activeColumns, learn=True):
        """
        Main Temporal Memory compute call — matches Numenta's TM.cpp structure:
        Phase 1: Activate cells (bursting, matching segments)
        Phase 2: Learn (segment adaptation and creation)
        Phase 3: Predict (set predictive state for next time step)
        Final: Anomaly score, prediction count, state update
        """
        self.activeColumns = set(activeColumns)
        self.learn = learn

        self._resetState()  # clears sets
        self._activateCells(activeColumns)
        self._activateDendrites()
        if learn:
            self._learnSegments()
        self._predictCells()
        if learn:
            self._punishPredictedSegments()
        self._calculateAnomalyScore()
        self._calculatePredictionCount()
        self._updateState()

        return self.anomalyScore, self.predictionCount

    def _resetState(self):
        self.activeColumns = set()
        self.activeCells = set()
        self.winnerCells = set()
        self.predictedColumns = set()
        self.predictiveCells = set()
        self.activeSegments = set()
        self.learningSegments = set()

    def _activateCells(self, activeColumns):
        """
        Phase 1: Activates cells based on predictions or bursts.
        """
        self.activeCells = set()
        self.winnerCells = set()
        self.activeSegments = set()
        self.learningSegments = set()

        for column in activeColumns:
            predictiveCells = [
                cell for cell in self.connections.cellsForColumn(column, self.cellsPerColumn)
                if self.connections.segmentActive(
                    self.connections.activeSegmentsForCell(cell, self.prevActiveCells),
                    self.activationThreshold
                )
            ]

            if predictiveCells:
                # At least one cell was predicted correctly → activate those cells
                for cell in predictiveCells:
                    self.activeCells.add(cell)
                    self.winnerCells.add(cell)

                # Mark their predictive segments as active
                for cell in predictiveCells:
                    segments = self.connections.activeSegmentsForCell(cell, self.prevActiveCells)
                    for seg in segments:
                        self.activeSegments.add(seg)
                        self.learningSegments.add(seg)
                self.predictedColumns.add(column)
            else:
                # BURST column: activate all cells
                for cell in self.connections.cellsForColumn(column, self.cellsPerColumn):
                    self.activeCells.add(cell)

                # Find best matching segment (BMS) on any cell in column
                bestCell, bestSeg = self._getBestMatchingSegment(column, self.prevActiveCells)

                if bestSeg is not None:
                    # Activate BMS cell and mark for learning
                    self.winnerCells.add(bestCell)
                    self.activeSegments.add(bestSeg)
                    self.learningSegments.add(bestSeg)
                else:
                    # No good match → choose least used cell and create new segment
                    bestCell = self._getLeastUsedCell(column)
                    if bestCell is not None:
                        self.winnerCells.add(bestCell)
                    else:
                        # All cells reached max segments, skip learning
                        pass

        self.activeColumns = activeColumns

    def _activateDendrites(self):
        """
        Determines which segments are active or matching.
        This must occur after cells are activated.
        """
        self.activeSegments = set()
        self.matchingSegments = set()

        for segment in self.connections.segments:
            activeSynapses = self.connections.activeSynapses(segment, self.prevActiveCells)
            if len(activeSynapses) >= self.activationThreshold:
                self.activeSegments.add(segment)
            if len(activeSynapses) >= self.minThreshold:
                self.matchingSegments.add(segment)

    def _learnSegments(self):
        """
        Reinforces learning segments and creates new ones when necessary.
        Matches Numenta's adaptSegments() in TM.cpp
        """
        self.segmentsToUpdate = set()

        for segment in self.learningSegments:
            cell = self.connections.cellForSegment(segment)

            if self.connections.segmentExists(segment):
                # Adapt existing segment
                self.connections.adaptSegment(
                    segment=segment,
                    activeSynapsesSource=self.prevActiveCells,
                    permanenceIncrement=self.permanenceIncrement,
                    permanenceDecrement=self.permanenceDecrement,
                    connectedPermanence=self.connectedPermanence,
                    maxSynapsesPerSegment=self.maxSynapsesPerSegment
                )
            else:
                # Only create segment if cell is under limit
                if len(self.connections.segmentsForCell(cell)) < self.maxSegmentsPerCell:
                    new_segment = self.connections.createSegment(cell)
                    self.connections.createSynapses(
                        segment=new_segment,
                        presynapticCells=self.prevActiveCells,
                        initialPermanence=self.initialPermanence,
                        maxNewSynapses=self.maxNewSynapseCount
                    )
                    segment = new_segment  # Ensure we track the new segment

            self.segmentsToUpdate.add(segment)

    def _getBestMatchingSegment(self, column, activeCells):
        bestCell = None
        bestSegment = None
        bestNumActive = -1

        for cell in self.connections.cellsForColumn(column, self.cellsPerColumn):
            segments = self.connections.segmentsForCell(cell)
            for seg in segments:
                activeSynapses = self.connections.activeSynapses(seg, activeCells)
                if len(activeSynapses) >= self.minThreshold and len(activeSynapses) > bestNumActive:
                    bestNumActive = len(activeSynapses)
                    bestCell = cell
                    bestSegment = seg

        return bestCell, bestSegment

    def _getLeastUsedCell(self, column):
        """
        Selects the cell in the column with the fewest segments.
        If there are ties, selects randomly among them.
        """
        cells = self.connections.cellsForColumn(column, self.cellsPerColumn)
        cell_segment_counts = [(cell, len(se
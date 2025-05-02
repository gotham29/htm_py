from htm_py.connections import Connections
import numpy as np


class TemporalMemory:
    def __init__(self,
                 columnDimensions,
                 cellsPerColumn,
                 activationThreshold,
                 initialPermanence,
                 connectedPermanence,
                 minThreshold,
                 maxNewSynapseCount,
                 permanenceIncrement,
                 permanenceDecrement,
                 predictedSegmentDecrement=0.0,
                 seed=42,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=32,
                 checkInputs=True):

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

        self.numColumns = np.prod(columnDimensions)
        self.numCells = self.numColumns * cellsPerColumn
        self.rng = np.random.default_rng(seed)

        self.connections = Connections(self.numCells)

        self.iteration = 0
        self.activeCells = []
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []
        self.lastUsedIterationForSegment = {}

    def compute(self, activeColumns, learn):
        prev_active_cells = set(self.activeCells)
        prev_winner_cells = set(self.winnerCells)

        self.activeCells = []
        self.winnerCells = []

        self._activate_cells(activeColumns, prev_active_cells, prev_winner_cells, learn)
        self._activate_dendrites(learn)

        if learn:
            for segment in self.activeSegments:
                self.lastUsedIterationForSegment[segment] = self.iteration
            self.iteration += 1

    def _activate_cells(self, activeColumns, prevActiveCells, prevWinnerCells, learn):
        columns = sorted(set(activeColumns))
        for col in columns:
            predicted_cells = []
            matching_segments = []
            active_segments = []

            for cell in self.cellsForColumn(col):
                segs = self.connections.segments_for_cell(cell)
                for seg in segs:
                    synapses = self.connections.synapses_for_segment(seg)
                    active_connected = [
                        s for s in synapses
                        if s.permanence >= self.connectedPermanence and
                        s.presynapticCell in prevActiveCells
                    ]
                    active_potential = [
                        s for s in synapses
                        if s.presynapticCell in prevActiveCells
                    ]

                    if len(active_connected) >= self.activationThreshold:
                        active_segments.append(seg)
                    if len(active_potential) >= self.minThreshold:
                        matching_segments.append((seg, len(active_potential)))

            if active_segments:
                self._activate_predicted_column(active_segments, prevActiveCells, prevWinnerCells, learn)
            else:
                self._burst_column(col, matching_segments, prevActiveCells, prevWinnerCells, learn)

    def _activate_predicted_column(self, activeSegments, prevActiveCells, prevWinnerCells, learn):
        used_cells = set()
        for seg in activeSegments:
            cell = self.connections.cell_for_segment(seg)
            if cell not in used_cells:
                self.activeCells.append(cell)
                self.winnerCells.append(cell)
                used_cells.add(cell)

                if learn:
                    self._adapt_segment(seg, prevActiveCells)
                    n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(seg, prevActiveCells)
                    if n_desired > 0:
                        self._grow_synapses(seg, prevWinnerCells, n_desired)

    def _burst_column(self, col, matchingSegments, prevActiveCells, prevWinnerCells, learn):
        cells = self.cellsForColumn(col)
        self.activeCells.extend(cells)

        if matchingSegments:
            best_segment = max(matchingSegments, key=lambda x: x[1])[0]
            winner_cell = self.connections.cell_for_segment(best_segment)
        else:
            # Choose cell with fewest segments
            num_segments = [len(self.connections.segments_for_cell(c)) for c in cells]
            min_segments = min(num_segments)
            candidates = [c for c, n in zip(cells, num_segments) if n == min_segments]
            winner_cell = self.rng.choice(candidates)

        self.winnerCells.append(winner_cell)

        if learn and prevWinnerCells:
            if matchingSegments:
                self._adapt_segment(best_segment, prevActiveCells)
                n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(best_segment, prevActiveCells)
                if n_desired > 0:
                    self._grow_synapses(best_segment, prevWinnerCells, n_desired)
            else:
                new_seg = self.connections.create_segment(winner_cell)
                self.lastUsedIterationForSegment[new_seg] = self.iteration
                self._grow_synapses(new_seg, prevWinnerCells, self.maxNewSynapseCount)

    def _adapt_segment(self, segment, prevActiveCells):
        synapses = self.connections.synapses_for_segment(segment)
        for s in synapses[:]:  # copy for safe removal
            cell = self.connections.presynaptic_cell(s)
            perm = self.connections.permanence(s)
            if cell in prevActiveCells:
                perm += self.permanenceIncrement
            else:
                perm -= self.permanenceDecrement
            perm = min(1.0, max(0.0, perm))
            if perm < 1e-6:
                self.connections.destroy_synapse(s)
            else:
                self.connections.update_permanence(s, perm)

    def _grow_synapses(self, segment, prevWinnerCells, n_desired):
        existing = {self.connections.presynaptic_cell(s) for s in self.connections.synapses_for_segment(segment)}
        candidates = sorted(set(prevWinnerCells) - existing)
        self.rng.shuffle(candidates)
        for cell in candidates[:n_desired]:
            self.connections.create_synapse(segment, cell, self.initialPermanence)

    def _activate_dendrites(self, learn):
        self.activeSegments = []
        self.matchingSegments = []
        for cell in range(self.numCells):
            for seg in self.connections.segments_for_cell(cell):
                n_connected = self.connections.num_active_connected_synapses(seg, self.activeCells, self.connectedPermanence)
                n_potential = self.connections.num_active_potential_synapses(seg, self.activeCells)

                if n_connected >= self.activationThreshold:
                    self.activeSegments.append(seg)
                if n_potential >= self.minThreshold:
                    self.matchingSegments.append(seg)

    def reset(self):
        self.activeCells = []
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []

    def cellsForColumn(self, column):
        start = column * self.cellsPerColumn
        return [start + i for i in range(self.cellsPerColumn)]

    def get_active_cells(self):
        return self.activeCells

    def get_winner_cells(self):
        return self.winnerCells

    def get_predictive_cells(self):
        return sorted({self.connections.cell_for_segment(seg) for seg in self.activeSegments})





# import random
# import logging
# from .connections import Connections

# logger = logging.getLogger(__name__)

# class TemporalMemory:
#     def __init__(self, columnDimensions,
#                  cellsPerColumn, activationThreshold,
#                  initialPermanence, connectedPermanence,
#                  permanenceIncrement, permanenceDecrement,
#                  maxNewSynapseCount, minThreshold, verbose=False, seed=42):
        
#         self.connections = Connections()
#         self.columnDimensions = columnDimensions
#         self.cellsPerColumn = cellsPerColumn
#         self.activationThreshold = activationThreshold
#         self.initialPermanence = initialPermanence
#         self.connectedPermanence = connectedPermanence
#         self.permanenceIncrement = permanenceIncrement
#         self.permanenceDecrement = permanenceDecrement
#         self.maxNewSynapseCount = maxNewSynapseCount
#         self.minThreshold = minThreshold
#         self.verbose = verbose
#         self.seed = seed
#         random.seed(seed)

#         self.prevActiveCells = set()
#         self.prevWinnerCells = set()
#         self.prevPredictiveCells = set()
#         self.activeCells = set()
#         self.winnerCells = set()
#         self.predictiveCells = set()

#     def compute(self, activeColumns, learn=True, verbose=False):
#         logger.debug("=== TM.compute() start ===")
#         self._reset_state()

#         segments_per_cell = {
#             cell: self.connections.segments_for_cell(cell)
#             for cell in self.connections.allCells()
#         }

#         cells_per_column = {
#             col: self.cellsForColumn(col)
#             for col in activeColumns
#         }

#         self._phase1_activate_columns(activeColumns, cells_per_column, segments_per_cell, learn)
#         anomaly_score = len(self.unpredictedColumns) / len(activeColumns)

#         if learn:
#             self._phase2_learn_segments()

#         self._phase3_predict_next(segments_per_cell)
#         prediction_count = len(self.predictiveCells) / len(activeColumns)

#         self._update_state()

#         return {
#             "anomaly_score": anomaly_score,
#             "prediction_count": prediction_count
#         }

#     def _phase1_activate_columns(self, activeColumns, cells_per_column, segments_per_cell, learn):
#         self.unpredictedColumns = set()
#         for col in activeColumns:
#             col_cells = cells_per_column[col]
#             prevPredictedCol = [c for c in col_cells if c in self.prevPredictiveCells]
#             colPredicted = len(prevPredictedCol) > 0

#             if colPredicted:
#                 self.activeCells.update(prevPredictedCol)
#                 self.winnerCells.update(prevPredictedCol)
#             else:

#                 self.burst_column(col, col_cells, segments_per_cell, learn)

#                 # self.unpredictedColumns.add(col)
#                 # self.activeCells.update(col_cells)

#                 # best_segment, best_overlap = self._get_best_matching_segment(col_cells, self.prevActiveCells)
#                 # if best_segment:
#                 #     winner_cell = self.connections.cell_for_segment(best_segment)
#                 #     self.winnerCells.add(winner_cell)
#                 #     if learn:
#                 #         self.connections.adapt_segment(best_segment, self.prevWinnerCells, True)
#                 # else:
#                 #     least_used = self._get_least_used_cells(col_cells, segments_per_cell)
#                 #     winner_cell = random.choice(least_used)
#                 #     self.winnerCells.add(winner_cell)
#                 #     if learn and len(self.prevWinnerCells) > 0:
#                 #         self.connections.create_segment(winner_cell, self.prevWinnerCells, self.initialPermanence)

#     def burst_column(self, col, col_cells, segments_per_cell, learn):
#         self.unpredictedColumns.add(col)
#         self.activeCells.update(col_cells)

#         best_segment, best_overlap = self._get_best_matching_segment(col_cells, self.prevActiveCells)
#         if best_segment:
#             winner_cell = self.connections.cell_for_segment(best_segment)
#             self.winnerCells.add(winner_cell)
#             if learn:
#                 self.connections.adapt_segment(best_segment, self.prevWinnerCells, True)
#         else:
#             least_used = self._get_least_used_cells(col_cells, segments_per_cell)
#             winner_cell = random.choice(least_used)
#             self.winnerCells.add(winner_cell)
#             if learn:
#                 if len(self.prevWinnerCells) > 0:
#                     segment = self.connections.create_segment(winner_cell, self.prevWinnerCells, self.initialPermanence)
#                     logger.debug(f"[Learn] Created new segment on burst cell {winner_cell} with synapses to {sorted(self.prevWinnerCells)}")
#                 else:
#                     logger.debug(f"[Learn] Not enough previous winner cells to create a new segment on cell {winner_cell}")
#         return winner_cell

#     def _phase2_learn_segments(self):
#         for seg in self.connections.get_active_segments():
#             cell = self.connections.cell_for_segment(seg)
#             positive = cell in self.activeCells
#             self.connections.adapt_segment(seg, self.prevWinnerCells, positive)

#         self.connections.clear_active_segments()

#     def _phase3_predict_next(self, segments_per_cell):
#         self.predictiveCells.clear()
#         for cell in self.connections.allCells():
#             segments = segments_per_cell.get(cell, [])
#             for segment in segments:
#                 overlap = self.connections.segment_overlap(segment, self.prevActiveCells, connected_only=True, permanence_connected=self.connectedPermanence)
#                 if overlap >= self.activationThreshold:
#                     self.predictiveCells.add(cell)
#                     if self.verbose:
#                         self.logger.debug(f"[Predict] Cell {cell} is predictive via segment {segment.id}")
#                     break

#     def _get_best_matching_segment(self, col_cells, prevActiveCells):
#         """
#         Return the best matching segment on this cell that meets the minThreshold criterion.
#         Returns (segment, overlap) tuple or (None, 0).
#         """
#         best_segment = None
#         best_overlap = 0
#         for cell in col_cells:
#             for segment in self.connections.segments_for_cell(cell):
#                 overlap = self.connections.segment_overlap(segment, prevActiveCells)
#                 if overlap > best_overlap and overlap >= self.minThreshold:
#                     best_segment = segment
#                     best_overlap = overlap

#         return best_segment, best_overlap

#     def _get_least_used_cells(self, col_cells, segments_per_cell):
#         cell_segment_count = {c: len(segments_per_cell.get(c, [])) for c in col_cells}
#         min_count = min(cell_segment_count.values())
#         least_used = [c for c in col_cells if cell_segment_count[c] == min_count]
#         return least_used

#     def get_predictive_cells(self):
#         return self.predictiveCells

#     def get_winner_cells(self):
#         return self.winnerCells

#     def _reset_state(self):
#         self.activeCells.clear()
#         self.winnerCells.clear()
#         self.predictiveCells.clear()

#     def _update_state(self):
#         self.prevActiveCells = self.activeCells.copy()
#         self.prevWinnerCells = self.winnerCells.copy()
#         self.prevPredictiveCells = self.predictiveCells.copy()

#     def cellsForColumn(self, col):
#         start = col * self.cellsPerColumn
#         return list(range(start, start + self.cellsPerColumn))

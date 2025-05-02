import random
import logging
from .connections import Connections

logger = logging.getLogger(__name__)

class TemporalMemory:
    def __init__(self, columnDimensions,
                 cellsPerColumn, activationThreshold,
                 initialPermanence, connectedPermanence,
                 permanenceIncrement, permanenceDecrement,
                 maxNewSynapseCount, minThreshold, verbose=False, seed=42):
        
        self.connections = Connections()
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.maxNewSynapseCount = maxNewSynapseCount
        self.minThreshold = minThreshold
        self.verbose = verbose
        self.seed = seed
        random.seed(seed)

        self.prevActiveCells = set()
        self.prevWinnerCells = set()
        self.prevPredictiveCells = set()
        self.activeCells = set()
        self.winnerCells = set()
        self.predictiveCells = set()

    def compute(self, activeColumns, learn=True, verbose=False):
        logger.debug("=== TM.compute() start ===")
        self._reset_state()

        segments_per_cell = {
            cell: self.connections.segments_for_cell(cell)
            for cell in self.connections.allCells()
        }

        cells_per_column = {
            col: self.cellsForColumn(col)
            for col in activeColumns
        }

        self._phase1_activate_columns(activeColumns, cells_per_column, segments_per_cell, learn)
        anomaly_score = len(self.unpredictedColumns) / len(activeColumns)

        if learn:
            self._phase2_learn_segments()

        self._phase3_predict_next(segments_per_cell)
        prediction_count = len(self.predictiveCells) / len(activeColumns)

        self._update_state()

        return {
            "anomaly_score": anomaly_score,
            "prediction_count": prediction_count
        }

    def _phase1_activate_columns(self, activeColumns, cells_per_column, segments_per_cell, learn):
        self.unpredictedColumns = set()
        for col in activeColumns:
            col_cells = cells_per_column[col]
            prevPredictedCol = [c for c in col_cells if c in self.prevPredictiveCells]
            colPredicted = len(prevPredictedCol) > 0

            if colPredicted:
                self.activeCells.update(prevPredictedCol)
                self.winnerCells.update(prevPredictedCol)
            else:

                self.burst_column(col, col_cells, segments_per_cell, learn)

                # self.unpredictedColumns.add(col)
                # self.activeCells.update(col_cells)

                # best_segment, best_overlap = self._get_best_matching_segment(col_cells, self.prevActiveCells)
                # if best_segment:
                #     winner_cell = self.connections.cell_for_segment(best_segment)
                #     self.winnerCells.add(winner_cell)
                #     if learn:
                #         self.connections.adapt_segment(best_segment, self.prevWinnerCells, True)
                # else:
                #     least_used = self._get_least_used_cells(col_cells, segments_per_cell)
                #     winner_cell = random.choice(least_used)
                #     self.winnerCells.add(winner_cell)
                #     if learn and len(self.prevWinnerCells) > 0:
                #         self.connections.create_segment(winner_cell, self.prevWinnerCells, self.initialPermanence)

    def burst_column(self, col, col_cells, segments_per_cell, learn):
        self.unpredictedColumns.add(col)
        self.activeCells.update(col_cells)

        best_segment, best_overlap = self._get_best_matching_segment(col_cells, self.prevActiveCells)
        if best_segment:
            winner_cell = self.connections.cell_for_segment(best_segment)
            self.winnerCells.add(winner_cell)
            if learn:
                self.connections.adapt_segment(best_segment, self.prevWinnerCells, True)
        else:
            least_used = self._get_least_used_cells(col_cells, segments_per_cell)
            winner_cell = random.choice(least_used)
            self.winnerCells.add(winner_cell)
            if learn:
                if len(self.prevWinnerCells) > 0:
                    segment = self.connections.create_segment(winner_cell, self.prevWinnerCells, self.initialPermanence)
                    logger.debug(f"[Learn] Created new segment on burst cell {winner_cell} with synapses to {sorted(self.prevWinnerCells)}")
                else:
                    logger.debug(f"[Learn] Not enough previous winner cells to create a new segment on cell {winner_cell}")
        return winner_cell

    def _phase2_learn_segments(self):
        for seg in self.connections.get_active_segments():
            cell = self.connections.cell_for_segment(seg)
            positive = cell in self.activeCells
            self.connections.adapt_segment(seg, self.prevWinnerCells, positive)

        self.connections.clear_active_segments()

    def _phase3_predict_next(self, segments_per_cell):
        self.predictiveCells.clear()
        for cell in self.connections.allCells():
            segments = segments_per_cell.get(cell, [])
            for segment in segments:
                overlap = self.connections.segment_overlap(segment, self.prevActiveCells, connected_only=True, permanence_connected=self.connectedPermanence)
                if overlap >= self.activationThreshold:
                    self.predictiveCells.add(cell)
                    if self.verbose:
                        self.logger.debug(f"[Predict] Cell {cell} is predictive via segment {segment.id}")
                    break

    def _get_best_matching_segment(self, col_cells, prevActiveCells):
        """
        Return the best matching segment on this cell that meets the minThreshold criterion.
        Returns (segment, overlap) tuple or (None, 0).
        """
        best_segment = None
        best_overlap = 0
        for cell in col_cells:
            for segment in self.connections.segments_for_cell(cell):
                overlap = self.connections.segment_overlap(segment, prevActiveCells)
                if overlap > best_overlap and overlap >= self.minThreshold:
                    best_segment = segment
                    best_overlap = overlap

        return best_segment, best_overlap

    def _get_least_used_cells(self, col_cells, segments_per_cell):
        cell_segment_count = {c: len(segments_per_cell.get(c, [])) for c in col_cells}
        min_count = min(cell_segment_count.values())
        least_used = [c for c in col_cells if cell_segment_count[c] == min_count]
        return least_used

    def _reset_state(self):
        self.activeCells.clear()
        self.winnerCells.clear()
        self.predictiveCells.clear()

    def _update_state(self):
        self.prevActiveCells = self.activeCells.copy()
        self.prevWinnerCells = self.winnerCells.copy()
        self.prevPredictiveCells = self.predictiveCells.copy()

    def cellsForColumn(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

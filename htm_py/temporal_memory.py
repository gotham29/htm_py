import logging
import logging
import os

LOG_FILE = "htm_debug.log"

# Ensure only one handler is added
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

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
        connections=None,
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
        if connections is None:
            self.connections = Connections(
                columnDimensions=columnDimensions,
                cellsPerColumn=cellsPerColumn,
                initialPermanence=initialPermanence,
                connectedPermanence=connectedPermanence,
                maxSegmentsPerCell=maxSegmentsPerCell
            )
        else:
            self.connections = connections

        # State variables
        self.iteration_num = 0
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
        self.learn = learn
        self._resetState()  # Clear previous state first
        self.activeColumns = set(activeColumns)  # Then assign active columns

        self._activateCells()
        self._activateDendrites()
        if learn:
            self._learnSegments()

        self._predictCells()
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

    def _activateCells(self):
        """
        Phase 1: Activates cells based on predictions or bursts.
        """
        self.activeCells = set()
        self.winnerCells = set()
        self.activeSegments = set()
        self.learningSegments = set()

        for column in self.activeColumns:
            predictiveCells = [
                cell for cell in self.connections.cellsForColumn(column, self.cellsPerColumn)
                if self.connections.segmentActive(
                    self.connections.activeSegmentsForCell(cell, self.prevActiveCells),
                    self.prevActiveCells,  # This was missing!
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
                    logging.debug(f"✅ Winner cell selected for new segment: {bestCell}")
                else:
                    logging.debug(f"❌ No winner cell found for column {column}")
                    # No good match → choose least used cell and create new segment
                    bestCell = self._getLeastUsedCell(column)
                    if bestCell is not None:
                        self.winnerCells.add(bestCell)
                    else:
                        # All cells reached max segments, skip learning
                        pass

        logger.debug(f"[ACTIVATE] ActiveColumns: {self.activeColumns}")
        logger.debug(f"[ACTIVATE] WinnerCells: {self.winnerCells}")
        logger.debug(f"[ACTIVATE] LearningSegments: {[s.id for s in self.learningSegments]}")

        # self.activeColumns = set(activeColumns)

    def _activateDendrites(self):
        """
        Determines which segments are active or matching.
        This must occur after cells are activated.
        """
        self.activeSegments = set()
        self.matchingSegments = set()

        for segment in self.connections._segments:
            activeSynapses = self.connections.activeSynapses(segment, self.prevActiveCells)
            if len(activeSynapses) >= self.activationThreshold:
                self.activeSegments.add(segment)
            if len(activeSynapses) >= self.minThreshold:
                self.matchingSegments.add(segment)

    def _learnSegments(self):
        """
        Phase 2: Learning. For each active column, choose a cell (best match or least used),
        and either adapt an existing segment or create a new one to learn from prevActiveCells.
        """
        prev_active = self.prevActiveCells
        prev_winnder = self.prevWinnerCells
        logger.debug(f"Phase 2: Learning on active columns: {self.activeColumns}")

        for column in self.activeColumns:
            best_cell, best_segment, num_active_synapses = self.connections.getBestMatchingSegment(
                column, prev_active, self.activationThreshold)

            if best_segment is not None and num_active_synapses >= self.minThreshold:
                self.connections.adapt_segment(
                    segment=best_segment,
                    prevWinnerCells=prev_winnder,
                    positive_reinforcement=True,
                    perm_inc=self.permanenceIncrement,
                    perm_dec=self.permanenceDecrement)
                logger.debug(f"Adapted segment on cell {best_cell} in column {column} with presynaptic cells {prev_active}")
                self.winnerCells.add(best_cell)
            else:
                learning_cell = best_cell if best_cell is not None else self._getLeastUsedCell(column)
                segment = self.connections.createSegment(learning_cell, self.iteration_num)
                self.connections.create_synapses(
                    segment,
                    presynaptic_cells=self.prevWinnerCells or self.prevActiveCells,
                    initial_permanence=self.initialPermanence,
                    max_new_synapses=self.maxNewSynapseCount
                )
                logger.debug(f"Created new segment on cell {learning_cell} in column {column} with synapses to {self.prevWinnerCells}")
                self.winnerCells.add(learning_cell)

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
        cell_segment_counts = [(cell, len(self.connections.segmentsForCell(cell))) for cell in cells]

        min_count = min(count for cell, count in cell_segment_counts if count < self.maxSegmentsPerCell)
        tied_cells = [cell for cell, count in cell_segment_counts if count == min_count]

        return np.random.choice(tied_cells) if tied_cells else None

    def _predictCells(self):
        """
        Phase 3: Predictive state - determine predictive cells for next timestep.
        """
        self.predictiveCells.clear()

        for segment in self.connections._segments:
            connected_sources = self.connections.connectedSynapses(segment, self.connectedPermanence)
            active_sources = connected_sources & self.activeCells
            logger.debug(f"Segment {segment.id} - connected sources: {connected_sources}, active sources: {active_sources}")

            if len(active_sources) >= self.activationThreshold:
                cell = self.connections.cellForSegment(segment)
                self.predictiveCells.add(cell)
                logger.debug(f"Cell {cell} added to predictiveCells")

    def _punishPredictedSegments(self):
        """
        Reduce permanence of segments that predicted incorrectly.
        A segment is punished if it was predictive but did not become active.
        """
        for cell in self.prevPredictiveCells:
            segments = self.connections.segmentsForCell(cell)
            for segment in segments:
                # If segment was predicted but not active now
                if segment not in self.activeSegments:
                    self.connections.decreaseSynapsePermanences(
                        segment,
                        self.predictedSegmentDecrement
                    )

    def _calculateAnomalyScore(self):
        """
        Calculates the anomaly score as the fraction of active columns that burst.
        A column bursts when no cell in it was predicted.
        """
        burstingColumns = self.activeColumns - self.predictedColumns
        self.anomalyScore = len(burstingColumns) / float(len(self.activeColumns))
        # logger.debug(f"[ANOMALY] ActiveColumns: {self.activeColumns}")
        logger.debug(f"[ANOMALY] PredictedColumns: {self.predictedColumns}")
        logger.debug(f"[ANOMALY] BurstingColumns: {burstingColumns}")
        logger.debug(f"[ANOMALY] Anomaly Score: {self.anomalyScore}")

    def _calculatePredictionCount(self):
        """
        Computes the number of unique columns that contain predictive cells.
        """
        self.predictionCount = len(self.predictiveCells) / len(self.activeColumns)


    def _updateState(self):
        """
        Updates the internal state by saving current cells and segments
        as the 'previous' state for the next timestep.
        """
        self.prevActiveCells = self.activeCells
        self.prevWinnerCells = self.winnerCells
        self.prevPredictiveCells = self.predictiveCells


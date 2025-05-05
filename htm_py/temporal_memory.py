import logging
from typing import List, Set, Tuple, Optional
from itertools import groupby
from collections import defaultdict

from htm_py.connections import Connections
from htm_py.connections import Segment


logger = logging.getLogger(__name__)


from typing import List, Tuple, Set, Optional
from htm_py.connections import Connections, Segment


class TemporalMemory:
    def __init__(
        self,
        columnDimensions: Tuple[int],
        cellsPerColumn: int,
        activationThreshold: int,
        initialPermanence: float,
        connectedPermanence: float,
        minThreshold: int,
        permanenceIncrement: float,
        permanenceDecrement: float,
        predictedSegmentDecrement: float,
        maxSegmentsPerCell: int,
        maxSynapsesPerSegment: int,
        maxNewSynapseCount: int,
    ):
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.predictedSegmentDecrement = predictedSegmentDecrement
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxNewSynapseCount = maxNewSynapseCount

        self.connections = Connections(
            numberOfColumns=self.numberOfColumns(),
            cellsPerColumn=cellsPerColumn,
            initialPermanence=initialPermanence,
            connectedPermanence=connectedPermanence,
            permanenceIncrement=permanenceIncrement,
            permanenceDecrement=permanenceDecrement,
            maxSegmentsPerCell=maxSegmentsPerCell,
            maxSynapsesPerSegment=maxSynapsesPerSegment,
        )

        self.activeCells: List[int] = []
        self.winnerCells: List[int] = []
        self.prevActiveCells: Set[int] = set()
        self.iteration = 0

    def numberOfColumns(self) -> int:
        product = 1
        for dim in self.columnDimensions:
            product *= dim
        return product

    def compute(self, activeColumns: List[int], learn: bool = True, iteration: Optional[int] = None):
        self.activeCells.clear()
        self.winnerCells.clear()
        self.prevActiveCellsDense = self._to_dense(self.prevActiveCells)

        for column in activeColumns:
            matching_segments = self._matching_segments(column)
            self._burst_column(column, matching_segments, learn)

        self.adaptSegmentsForPredictedColumns(activeColumns, learn)
        self.punishPredictedColumnsThatDidNotBecomeActive()

        self.prevActiveCells = set(self.activeCells)
        if iteration is not None:
            self.iteration = iteration

    def _burst_column(self, column: int, matching_segments: List[Segment], learn: bool):
        """
        Burst a column by activating all its cells. If learning is enabled,
        adapt an existing segment or create a new one only if minThreshold is met.

        Mirrors Numenta’s TemporalMemory::burstColumn().
        """
        # Always activate all cells in the column (Phase 1)
        for cell in self._cells_for_column(column):
            self.activeCells.append(cell)

        if not learn:
            return

        if matching_segments:
            segment = self._best_matching_segment(matching_segments)
            cell = self.connections.cellForSegment(segment)
            self.connections.adaptSegment(
                segment,
                self.prevActiveCellsDense,
                self.permanenceIncrement,
                self.permanenceDecrement
            )
            self.winnerCells.append(cell)
        else:
            connected_count = sum(
                1 for c in self.prevActiveCells
                if c is not None and c < len(self.prevActiveCellsDense)
                and self.prevActiveCellsDense[c]
            )
            if connected_count < self.minThreshold:
                logger.debug(f"[DEBUG SKIP] Skipping segment creation: only {connected_count} active synapses < minThreshold={self.minThreshold}")
                return

            best_cell = self._get_least_used_cell(column)
            segment = self.connections.createSegment(best_cell, self.iteration)
            if segment is None:
                logger.debug(f"[DEBUG SKIP] Segment creation disabled (maxSegmentsPerCell=0) for cell {best_cell}")
                return

            self._grow_synapses_to_segment(segment, self.prevActiveCells)
            self.winnerCells.append(best_cell)

    def adaptSegmentsForPredictedColumns(
        self,
        activeColumns: List[int],
        learn: bool
    ):
        """
        Phase 2 subroutine:
        For each cell that was predicted (i.e., has an active segment from previous step)
        and whose column is currently active, reinforce the active segment.

        Mirrors TemporalMemory::adaptSegmentsForPredictedColumns().
        """
        if not learn:
            return

        active_columns_set = set(activeColumns)

        for cell in self.prevActiveCells:
            segments = self.connections.segmentsForCell(cell)
            for seg in segments:
                # Only consider segments that predicted this cell (i.e., were active last timestep)
                # and whose segment is connected enough (>= activationThreshold)
                active_syn_count = sum(
                    1
                    for syn in self.connections.synapsesForSegment(seg)
                    if syn.presynaptic_cell in self.prevActiveCells
                )
                if active_syn_count >= self.activationThreshold:
                    col = cell // self.cellsPerColumn
                    if col in active_columns_set:
                        self.connections.adaptSegment(
                            seg,
                            self.prevActiveCellsDense,
                            self.permanenceIncrement,
                            self.permanenceDecrement
                        )
                        self.winnerCells.append(cell)

    def punishPredictedColumnsThatDidNotBecomeActive(self):
        """
        For all previously predicted cells that were *not* activated by input,
        punish the segments that led to those false predictions by decreasing
        permanence of their synapses.

        Mirrors Numenta’s TemporalMemory::punishPredictedColumn().
        """
        total_cells = self.numberOfColumns() * self.cellsPerColumn

        for cell in range(total_cells):
            if cell in self.prevActiveCells:
                continue  # Correct prediction → no punishment

            segments = self.connections.segmentsForCell(cell)
            for segment in segments:
                # Only punish segments that would have predicted this cell
                active_syn_count = sum(
                    1 for syn in self.connections.synapsesForSegment(segment)
                    if syn.presynaptic_cell in self.prevActiveCells
                )
                if active_syn_count >= self.activationThreshold:
                    logger.debug(f"[DEBUG PUNISH] Cell {cell} predicted but not active — punishing segment.")
                    self.connections.adaptSegment(
                        segment,
                        self.prevActiveCellsDense,
                        -self.predictedSegmentDecrement,  # decrease active
                        self.predictedSegmentDecrement    # increase inactive
                    )

    def _matching_segments(self, column: int) -> List[Segment]:
        result = []
        for cell in self._cells_for_column(column):
            for seg in self.connections.segmentsForCell(cell):
                active = sum(
                    1
                    for syn in self.connections.synapsesForSegment(seg)
                    if syn.presynaptic_cell in self.prevActiveCells
                )
                if active >= self.minThreshold:
                    result.append(seg)
        return result

    def _best_matching_segment(self, segments: List[Segment]) -> Segment:
        return max(
            segments,
            key=lambda seg: sum(
                1
                for syn in self.connections.synapsesForSegment(seg)
                if 0 <= syn.presynaptic_cell < len(self.prevActiveCellsDense)
                    and self.prevActiveCellsDense[syn.presynaptic_cell]
            ),
        )

    def _best_matching_cell(self, column: int, prev_active_dense: List[bool]) -> Optional[int]:
        best_cell = None
        max_active = self.minThreshold
        for cell in self._cells_for_column(column):
            for seg in self.connections.segmentsForCell(cell):
                active = sum(
                    1
                    for syn in self.connections.synapsesForSegment(seg)
                    if 0 <= syn.presynaptic_cell < len(prev_active_dense)
                    and prev_active_dense[syn.presynaptic_cell]
                )
                if active >= max_active:
                    best_cell = cell
                    max_active = active
        return best_cell

    def _grow_synapses_to_segment(self, segment: Segment, presynaptic_cells: Set[int]):
        existing_presyn = {
            syn.presynaptic_cell for syn in self.connections.synapsesForSegment(segment)
        }
        candidates = sorted(presynaptic_cells - existing_presyn)
        n_add = min(len(candidates), self.maxNewSynapseCount)
        for presyn in candidates[:n_add]:
            self.connections.createSynapse(segment, presyn, self.initialPermanence)

    def _get_least_used_cell(self, column: int) -> int:
        return min(
            self._cells_for_column(column),
            key=lambda c: len(self.connections.segmentsForCell(c))
        )

    def _cells_for_column(self, column: int) -> List[int]:
        start = column * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def _to_dense(self, active_set: Set[int]) -> List[bool]:
        total_cells = self.numberOfColumns() * self.cellsPerColumn
        dense = [False] * total_cells
        for i in active_set:
            if 0 <= i < total_cells:
                dense[i] = True
        return dense

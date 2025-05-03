import math
import random
from htm_py.connections import Connections


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
                 predictedSegmentDecrement,
                 seed=42,
                 encoderWidth=40,
                 maxSegmentsPerCell=None,
                 maxSynapsesPerSegment=None):

        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.numColumns = columnDimensions[0]
        self.numCells = self.numColumns * self.cellsPerColumn

        self.connections = Connections(self.numCells)

        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.encoderWidth = encoderWidth
        self.maxNewSynapseCount = maxNewSynapseCount
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.predictedSegmentDecrement = predictedSegmentDecrement

        self.rng = random.Random(seed)

        self.activeCells = []
        self.winnerCells = []
        self.activeSegments = []
        self.matchingSegments = []
        self.iteration = 0
        self.lastUsedIterationForSegment = {}
        self.predictiveCells = set()
        self.prevActiveCells = set()
        self.prevWinnerCells = set()
        self.prevPredictiveCells = set()


    def compute(self, activeColumns, learn):
        self.iteration += 1
        self.learn = learn
        self.active_columns = activeColumns

        # Capture state BEFORE compute
        prev_predictive_cells = set(self.prevPredictiveCells)

        self._burst_columns(activeColumns)
        if learn:
            self._learn_segments(activeColumns)
        self._predict_cells()
        self._calculate_anomaly_score(prev_predictive_cells)
        self._calculate_prediction_count(prev_predictive_cells)

        # ✅ Final step: preserve state for next timestep
        self._update_state()

    def _activate_dendrites(self):
        self.activeSegments = []
        self.matchingSegments = []
        predictive_cells = set()

        for cell in range(self.numCells):
            for seg in self.connections.segments_for_cell(cell):
                n_connected = self.connections.num_active_connected_synapses(seg, set(self.activeCells), self.connectedPermanence)
                n_potential = self.connections.num_active_potential_synapses(seg, set(self.activeCells))

                if n_connected >= self.activationThreshold:
                    self.activeSegments.append(seg)
                    predictive_cells.add(cell)
                elif n_potential >= self.minThreshold:
                    self.matchingSegments.append(seg)

                print(f"[DEND] Cell {cell} has {n_connected} connected synapses → predictive: {n_connected >= self.activationThreshold}")
        
        return predictive_cells

    def _learn_segments(self, active_columns):
        """
        Phase 2: Learning — grow segments on matching or new cells for learning columns.
        This version fixes segment creation to target the correct winner cell per column.
        """
        for col in active_columns:
            learning_cells = self._get_winner_cells_for_column(col)
            prev_active = list(self.prevActiveCells)
            best_cell, best_seg = self._get_best_matching_cell_and_segment(col, prev_active)

            if best_seg is not None:
                self._adapt_segment(best_seg, prev_active, self.permanenceIncrement, self.permanenceDecrement)
                print(f"[LEARN] Adapted existing segment on cell {best_cell} for col {col}")
            else:
                for cell in learning_cells:
                    seg = self.connections.create_segment(cell)
                    self._connect_new_synapses(seg, prev_active, self.initialPermanence)
                    print(f"[LEARN] Created new segment on winner cell {cell} for col {col} with synapses to {prev_active}")

    def _adapt_segment(self, segment, prevActiveCells, perm_inc, perm_dec):
        """
        Reinforce or punish synapses on the given segment depending on whether
        their presynaptic cells were active in the previous timestep.
        """
        synapses = self.connections.synapses_for_segment(segment)
        seen_presyn_cells = {self.connections.presynaptic_cell(s) for s in synapses}

        for s in synapses[:]:  # Copy for safe iteration
            cell = self.connections.presynaptic_cell(s)
            perm = self.connections.permanence(s)
            perm += perm_inc if cell in prevActiveCells else -perm_dec
            perm = max(0.0, min(1.0, perm))
            if perm < 1e-6:
                self.connections.destroy_synapse(s)
            else:
                self.connections.update_permanence(s, perm)

        # Add new synapses to active cells that were missing before
        new_syn_cells = set(prevActiveCells) - seen_presyn_cells
        if new_syn_cells:
            self.connections.add_synapses(segment, list(new_syn_cells),
                                        self.initialPermanence,
                                        self.maxNewSynapseCount)

    def _activate_cells(self, activeColumns, prevActiveCells, prevWinnerCells, learn):
        for col in sorted(set(activeColumns)):
            col_cells = self.cells_for_column(col)
            matching_segments = []
            active_segments = []

            for cell in col_cells:
                for seg in self.connections.segments_for_cell(cell):
                    n_connected = self.connections.num_active_connected_synapses(seg, prevActiveCells, self.connectedPermanence)
                    n_total = self.connections.num_active_potential_synapses(seg, prevActiveCells)
                    if n_connected >= self.activationThreshold:
                        active_segments.append(seg)
                    if n_total >= self.minThreshold:
                        matching_segments.append((seg, n_total))

            if active_segments:
                self._activate_predicted_column(active_segments, prevActiveCells, prevWinnerCells, learn)
            else:
                # self.burst_column(col, matching_segments, prevActiveCells, prevWinnerCells, learn)
                self.burst_column(col)

    def _activate_predicted_column(self, segments, prevActiveCells, prevWinnerCells, learn):
        # Choose best matching segment based on connected synapses
        best_segment = max(
            segments,
            key=lambda seg: self.connections.num_active_connected_synapses(seg, prevActiveCells, self.connectedPermanence)
        )
        cell = self.connections.cell_for_segment(best_segment)
        self.activeCells.append(cell)
        self.winnerCells.append(cell)

        if learn:
            self._adapt_segment(best_segment, prevActiveCells, self.permanenceIncrement, self.permanenceDecrement)
            n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(best_segment, prevActiveCells)
            if n_desired > 0 and prevWinnerCells:
                self._grow_synapses(best_segment, prevWinnerCells, n_desired)

            # Cleanup empty segments
            if not self.connections.synapses_for_segment(best_segment):
                self.connections.destroy_segment(best_segment)

    def _burst_columns(self, active_columns):
        """Apply burst_column logic to all active columns."""
        for col in active_columns:
            self.burst_column(col)

    def burst_column(self, column: int):
        """Burst a column: if predicted, only predicted cells become active & winner;
        otherwise, all become active, and one is chosen as winner."""
        cells = self.cells_for_column(column)
        predicted_cells = [c for c in cells if c in self.predictiveCells]

        if predicted_cells:
            # Numenta: if predicted, those cells become active and winner
            for cell in predicted_cells:
                self.activeCells.append(cell)
                self.winnerCells.append(cell)
            print(f"[BURST] col={column} predicted → winners={predicted_cells}")
        else:
            # Numenta: if unpredicted, all become active, pick least-used winner
            for cell in cells:
                self.activeCells.append(cell)
            winner = self.get_least_used_cell(cells)
            self.winnerCells.append(winner)
            print(f"[BURST] col={column} unpredicted → all active, winner_cell={winner}")

    def _grow_synapses(self, segment, prevWinnerCells, n_desired):
        if not prevWinnerCells:
            return

        existing = {self.connections.presynaptic_cell(s) for s in self.connections.synapses_for_segment(segment)}
        candidates = list(set(prevWinnerCells) - existing)
        self.rng.shuffle(candidates)

        existing_count = self.connections.num_synapses(segment)
        max_allowed = self.maxSynapsesPerSegment or float('inf')
        room = max_allowed - existing_count
        n_create = min(n_desired, room)

        if n_create <= 0:
            print(f"[GROW] Skipped: n_create={n_create}, candidates={candidates}, existing_count={existing_count}")
            return

        print(f"[GROW] Attempting to grow on seg={segment} with prevWinnerCells={prevWinnerCells}")
        print(f"[GROW] Existing presyn cells: {existing}")
        print(f"[GROW] Candidates: {candidates[:n_create]}")

        if n_create < len(candidates) and self.maxSynapsesPerSegment:
            to_remove = sorted(
                self.connections.synapses_for_segment(segment),
                key=lambda s: s.permanence
            )[:len(candidates) - n_create]
            for s in to_remove:
                self.connections.destroy_synapse(s)

        for cell in candidates[:n_create]:
            self.connections.create_synapse(segment, cell, self.initialPermanence)        
        print(f"[GROW] Segment {segment} grew synapses to: {candidates[:n_create]}")

    def get_least_used_cell(self, cells):
        # If input is a column index (int), get its cells
        if isinstance(cells, int):
            cells = self.cells_for_column(cells)
        elif not isinstance(cells, list):
            raise TypeError(f"Expected int or list[int] for cells, got {type(cells)}")

        # Now select the cell with the fewest number of segments
        min_segments = float("inf")
        least_used = None
        for cell in cells:
            num_segments = len(self.connections.segments_for_cell(cell))
            if num_segments < min_segments:
                min_segments = num_segments
                least_used = cell
        return least_used

    def cells_for_column(self, col):
        if isinstance(col, list):
            raise TypeError(f"cells_for_column() expected int but got list: {col}")
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def get_winner_cells(self):
        return list(self.winnerCells)

    def get_active_cells(self):
        return self.activeCells

    def get_predictive_cells(self):
        return sorted(self.predictiveCells)

    def _calculate_anomaly_score(self, prev_predictive_cells):
        if not self.activeCells:
            return 0.0
        if not prev_predictive_cells:
            return 1.0

        unpredicted = [c for c in self.activeCells if c not in prev_predictive_cells]
        return len(unpredicted) / len(self.activeCells)

    def _calculate_prediction_count(self, prev_predictive_cells):
        """
        Returns the number of predictive cells (for analysis/debugging).
        """
        return len(prev_predictive_cells) / self.encoderWidth

    def _get_winner_cells_for_column(self, col):
        """
        Return all winner cells in a column from the previous timestep.
        If none exist, this should never happen due to burst logic assigning one.
        """
        cells = self.cells_for_column(col)
        winner_cells = [c for c in cells if c in self.winnerCells]  #self.prevWinnerCells

        # 🔍 Sanity check
        if not winner_cells:
            print(f"[ERROR] No winner cells found for column {col} in iteration {self.iteration}")
            raise RuntimeError(f"No winner cells found for column {col}")

        return winner_cells

    def _predict_cells(self):
        self.predictiveCells = self._activate_dendrites()

    def _get_best_matching_cell_and_segment(self, col, prev_active_cells):
        best_cell = None
        best_segment = None
        best_num = -1

        for cell in self.cells_for_column(col):
            for seg in self.connections.segments_for_cell(cell):
                n_potential = self.connections.num_active_potential_synapses(seg, prev_active_cells)
                if n_potential > best_num:
                    best_cell = cell
                    best_segment = seg
                    best_num = n_potential

        return best_cell, best_segment

    def _connect_new_synapses(self, segment, source_cells, initial_perm):
        """
        Create synapses from `source_cells` to the given `segment` with `initial_perm`.
        Only create synapses to cells not already connected.
        """
        existing = {self.connections.presynaptic_cell(s) for s in self.connections.synapses_for_segment(segment)}
        new_cells = list(set(source_cells) - existing)
        self.rng.shuffle(new_cells)

        max_allowed = self.maxSynapsesPerSegment or float('inf')
        if max_allowed == float('inf'):
            to_add = new_cells
        else:
            room = int(max_allowed - self.connections.num_synapses(segment))
            to_add = new_cells[:room]
        for cell in to_add:
            self.connections.create_synapse(segment, cell, initial_perm)

        print(f"[CONNECT] Segment {segment} → synapses added: {to_add}")

    def winner_cells_for_column(self, col):
        """Return the list of winner cells in the given column for the current timestep."""
        return [c for c in self.winnerCells if c // self.cellsPerColumn == col]

    @property
    def active_cells(self):
        return self.activeCells

    @property
    def winner_cells(self):
        return self.winnerCells

    def number_of_cells(self):
        return self.number_of_columns() * self.cellsPerColumn

    def winner_cells_for_column(self, col):
        return [c for c in self.winnerCells if self.column_for_cell(c) == col]

    def column_for_cell(self, cell):
        return cell % self.columnDimensions[0]
    
    def number_of_columns(self):
        return math.prod(self.columnDimensions)

    def _update_state(self):
        self.prevActiveCells = set(self.activeCells)
        self.prevWinnerCells = set(self.winnerCells)
        self.prevPredictiveCells = set(self.predictiveCells)

    def reset(self):
        self.activeCells = set()
        self.winnerCells = set()
        self.predictiveCells = set()
        self.prevActiveCells = set()
        self.prevWinnerCells = set()



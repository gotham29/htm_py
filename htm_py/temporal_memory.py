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
                 maxSegmentsPerCell=None,
                 maxSynapsesPerSegment=None):

        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.numColumns = columnDimensions[0]
        self.numCells = self.numColumns * self.cellsPerColumn

        self.connections = Connections(
            num_cells=self.numCells,
            maxSegmentsPerCell=maxSegmentsPerCell,
            maxSynapsesPerSegment=maxSynapsesPerSegment
        )

        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.maxNewSynapseCount = maxNewSynapseCount
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
        for col in sorted(set(activeColumns)):
            col_cells = self.cellsForColumn(col)
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
                self._burst_column(col, matching_segments, prevActiveCells, prevWinnerCells, learn)

    def _activate_predicted_column(self, segments, prevActiveCells, prevWinnerCells, learn):
        for seg in segments:
            cell = self.connections.cell_for_segment(seg)
            self.activeCells.append(cell)
            self.winnerCells.append(cell)

            if learn:
                self._adapt_segment(seg, prevActiveCells)
                n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(seg, prevActiveCells)
                if n_desired > 0 and prevWinnerCells:
                    self._grow_synapses(seg, prevWinnerCells, n_desired)

                # Cleanup empty segments
                if not self.connections.synapses_for_segment(seg):
                    self.connections.destroy_segment(seg)

    def _burst_column(self, col, matchingSegments, prevActiveCells, prevWinnerCells, learn):
        cells = self.cellsForColumn(col)
        self.activeCells.extend(cells)

        if matchingSegments:
            best_segment = max(matchingSegments, key=lambda x: x[1])[0]
            winner_cell = self.connections.cell_for_segment(best_segment)
            print(f"[burstColumn] MATCHING: Using best segment on cell {winner_cell}")
        else:
            # Select cell with fewest segments, tie-break randomly
            num_segments = [len(self.connections.segments_for_cell(c)) for c in cells]
            min_segments = min(num_segments)
            candidates = [c for c, n in zip(cells, num_segments) if n == min_segments]
            winner_cell = self.rng.choice(candidates)
            print(f"[burstColumn] BURST: Chose new winner cell {winner_cell} (min segments = {min_segments})")

        self.winnerCells.append(winner_cell)

        if not learn:
            print("[burstColumn] Skipping learning")
            return

        if matchingSegments and prevWinnerCells:
            # Adapt best segment and grow if possible
            print(f"[burstColumn] Adapting best segment on cell {winner_cell}")
            self._adapt_segment(best_segment, prevActiveCells)
            n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(best_segment, prevActiveCells)
            if n_desired > 0:
                self._grow_synapses(best_segment, prevWinnerCells, n_desired)
            self.lastUsedIterationForSegment[best_segment] = self.iteration

        elif prevWinnerCells:
            # Create a new segment only if prevWinnerCells is non-empty
            n_desired = self.maxNewSynapseCount
            if n_desired > 0:
                new_seg = self.connections.create_segment(
                    winner_cell,
                    last_used_map=self.lastUsedIterationForSegment,
                    current_iter=self.iteration
                )
                print(f"[burstColumn] Creating new segment on cell {winner_cell} with {n_desired} synapses")
                self._grow_synapses(new_seg, prevWinnerCells, n_desired)
                self.lastUsedIterationForSegment[new_seg] = self.iteration

    # def _burst_column(self, col, matchingSegments, prevActiveCells, prevWinnerCells, learn):
    #     cells = self.cellsForColumn(col)
    #     self.activeCells.extend(cells)

    #     if matchingSegments:
    #         best_segment = max(matchingSegments, key=lambda x: x[1])[0]
    #         winner_cell = self.connections.cell_for_segment(best_segment)
    #     else:
    #         # Choose cell with fewest segments (tie-break by RNG)
    #         num_segments = [len(self.connections.segments_for_cell(c)) for c in cells]
    #         min_segments = min(num_segments)
    #         candidates = [c for c in cells if len(self.connections.segments_for_cell(c)) == min_segments]
    #         winner_cell = self.rng.choice(candidates)

    #     self.winnerCells.append(winner_cell)

    #     if not learn:
    #         return

    #     if matchingSegments and prevWinnerCells:
    #         # Adapt and grow on best segment
    #         self._adapt_segment(best_segment, prevActiveCells)
    #         n_desired = self.maxNewSynapseCount - self.connections.num_active_potential_synapses(best_segment, prevActiveCells)
    #         if n_desired > 0:
    #             self._grow_synapses(best_segment, prevWinnerCells, n_desired)

    #     elif prevWinnerCells:
    #         # Only create new segment if prevWinnerCells is not empty
    #         new_seg = self.connections.create_segment(winner_cell)
    #         self.lastUsedIterationForSegment[new_seg] = self.iteration
    #         self._grow_synapses(new_seg, prevWinnerCells, self.maxNewSynapseCount)

    def _adapt_segment(self, segment, prevActiveCells):
        synapses = self.connections.synapses_for_segment(segment)
        for s in synapses[:]:  # copy for safe iteration
            cell = self.connections.presynaptic_cell(s)
            perm = self.connections.permanence(s)
            perm += self.permanenceIncrement if cell in prevActiveCells else -self.permanenceDecrement
            perm = max(0.0, min(1.0, perm))
            if perm < 1e-6:
                self.connections.destroy_synapse(s)
            else:
                self.connections.update_permanence(s, perm)

        if not self.connections.synapses_for_segment(segment):
            self.connections.destroy_segment(segment)

    def _grow_synapses(self, segment, prevWinnerCells, n_desired):
        if not prevWinnerCells:
            return

        existing = {self.connections.presynaptic_cell(s) for s in self.connections.synapses_for_segment(segment)}
        candidates = list(set(prevWinnerCells) - existing)
        self.rng.shuffle(candidates)

        existing_count = self.connections.num_synapses(segment)
        max_allowed = self.connections.maxSynapsesPerSegment or float('inf')
        room = max_allowed - existing_count
        n_create = min(n_desired, room)

        if n_create < len(candidates) and self.connections.maxSynapsesPerSegment:
            to_remove = sorted(
                self.connections.synapses_for_segment(segment),
                key=lambda s: s.permanence
            )[:len(candidates) - n_create]
            for s in to_remove:
                self.connections.destroy_synapse(s)

        for cell in candidates[:n_create]:
            self.connections.create_synapse(segment, cell, self.initialPermanence)
        print(f"[growSynapses] segment={segment}, nDesired={n_desired}, available={len(candidates)}")

    def _activate_dendrites(self, learn):
        self.activeSegments = []
        self.matchingSegments = []
        for cell in range(self.numCells):
            for seg in self.connections.segments_for_cell(cell):
                if self.connections.num_active_connected_synapses(seg, set(self.activeCells), self.connectedPermanence) >= self.activationThreshold:
                    self.activeSegments.append(seg)
                elif self.connections.num_active_potential_synapses(seg, set(self.activeCells)) >= self.minThreshold:
                    self.matchingSegments.append(seg)

    def cellsForColumn(self, col):
        start = col * self.cellsPerColumn
        return list(range(start, start + self.cellsPerColumn))

    def get_winner_cells(self):
        return self.winnerCells

    def get_active_cells(self):
        return self.activeCells

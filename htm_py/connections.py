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
import itertools
import numpy as np

from collections import defaultdict, namedtuple

SynapseData = namedtuple("SynapseData", ["presynapticCell", "permanence"])


class Segment:
    def __init__(self, segment_id, cell, iteration_num):
        self.id = segment_id
        self.cell = cell
        self.iteration_num = iteration_num
        self.synapses = []

    def __eq__(self, other):
        return isinstance(other, Segment) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<Segment id={self.id} cell={self.cell} synapses={len(self.synapses)}>"


class Synapse:
    def __init__(self, presynaptic_cell, permanence):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence


class CellData:
    def __init__(self, segments):
        self.segments = segments


class Connections:
    def __init__(self, columnDimensions, cellsPerColumn, initialPermanence, connectedPermanence, maxSegmentsPerCell):
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.numCells = np.prod(columnDimensions) * cellsPerColumn
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence

        self._segments = set()  # should contain full Segment objects
        self._segment_to_cell = {}
        self._cell_to_segments = defaultdict(list)
        self._segment_to_synapses = defaultdict(set)
        self._segment_last_used = {}
        self._next_segment_id = 0

    def createSegment(self, cell, iteration_num):
        """
        Creates a new Segment object for the given cell and registers it.
        """
        if cell not in self._cell_to_segments:
            self._cell_to_segments[cell] = []

        segments = self._cell_to_segments[cell]
        if len(segments) >= self.maxSegmentsPerCell:
            # Evict least recently used segment
            oldest_segment = min(segments, key=lambda s: self._segment_last_used.get(s.id, -1))
            self.destroySegment(oldest_segment)

        new_segment_id = self._next_segment_id
        self._next_segment_id += 1

        segment = Segment(new_segment_id, cell, iteration_num)
        print(f"[DEBUG CREATE] Registering segment {segment} for cell {cell}")

        self._cell_to_segments[cell].append(segment)
        self._segment_to_cell[segment] = cell
        self._segment_to_synapses[segment] = segment.synapses
        self._segment_last_used[segment] = iteration_num
        self._segments.add(segment)

        return segment

    def create_synapse(self, segment, presynaptic_cell, permanence):
        syn = Synapse(presynaptic_cell, permanence)
        segment.synapses.append(syn)
        return syn

    def create_synapses(self, segment, presynaptic_cells, initial_permanence, max_new_synapses):
        count = 0
        for cell in presynaptic_cells:
            if count >= max_new_synapses:
                break
            self.create_synapse(segment, cell, initial_permanence)
            count += 1
            print(f"[DEBUG] Created synapse on segment {segment.id} for presynaptic cell {cell}")

    def segmentExists(self, segment):
        return segment in self._segment_to_cell

    def segmentsForCell(self, cell):
        print(f"[DEBUG FETCH] segmentsForCell({cell}) → {self._cell_to_segments.get(cell)}")
        return self._cell_to_segments.get(cell, [])

    def cellForSegment(self, segment):
        return self._segment_to_cell.get(segment)

    def cellsForColumn(self, column, cellsPerColumn):
        start = column * cellsPerColumn
        return list(range(start, start + cellsPerColumn))

    def activeSynapses(self, segment, activeCells):
        return {
            syn for syn in segment.synapses
            if syn.presynaptic_cell in activeCells
        }

    def connectedSynapses(self, segment, connectedPermanence):
        return {
            syn.presynaptic_cell for syn in segment.synapses
            if syn.permanence >= connectedPermanence
        }

    def adapt_segment(self, segment, prevWinnerCells, positive_reinforcement, perm_inc, perm_dec):
        """
        Adapt synapses on `segment` based on whether the reinforcement is positive or negative.
        In positive reinforcement: increase permanence for synapses with presynaptic cells in prevWinnerCells,
        and decrease for others.
        In negative reinforcement: only decrease for synapses with presynaptic cells in prevWinnerCells.
        Remove synapse if permanence drops to 0.
        """
        for syn in list(segment.synapses):  # Copy list to safely mutate
            cell = syn.presynaptic_cell
            if positive_reinforcement:
                if cell in prevWinnerCells:
                    syn.permanence = min(1.0, syn.permanence + perm_inc)
                else:
                    syn.permanence = max(0.0, syn.permanence - perm_dec)
            else:
                if cell in prevWinnerCells:
                    syn.permanence = max(0.0, syn.permanence - perm_dec)

        # Optional: prune synapses with zero permanence
        segment.synapses = [s for s in segment.synapses if s.permanence > 0.0]

    def activeSegmentsForCell(self, cell, prevActiveCells):
        return [
            seg for seg in self.segmentsForCell(cell)
            if any(syn.presynaptic_cell in prevActiveCells and syn.permanence >= self.connectedPermanence
                   for syn in seg.synapses)
        ]

    def decreaseSynapsePermanences(self, segment, amount):
        for synapse in self._segments[segment]:
            self.permanences[synapse] = max(0.0, self.permanences[synapse] - amount)

    def segmentActive(self, segments, activePresynapticCells, threshold):
        """
        Returns True if any of the given segments has enough active synapses to be considered active.
        This matches the logic used in TemporalMemory.cpp:segmentActive()

        Parameters:
            segments: list or set of segment indices
            activePresynapticCells: set of active cells from the previous timestep
            threshold: minimum number of active synapses for a segment to be considered active

        Returns:
            bool: True if any segment is active
        """
        for segment in segments:
            active_synapses = self.activeSynapses(segment, activePresynapticCells)
            if len(active_synapses) >= threshold:
                return True
        return False

    def destroySegment(self, segment):
        """
        Completely removes a segment and all its synapses.
        """
        cell = self.cellForSegment(segment)
        if cell in self._cell_to_segments:
            self._cell_to_segments[cell] = [s for s in self._cell_to_segments[cell] if s.id != segment.id]
        self._segment_to_cell.pop(segment.id, None)
        self._segment_to_synapses.pop(segment.id, None)
        self._segment_last_used.pop(segment.id, None)
        self.segments.discard(segment)
        logging.debug(f"❌ Destroyed segment {segment.id} from cell {cell}")

    def getBestMatchingSegment(self, column, activeCells, minThreshold):
        """
        For the given column, return the cell and its segment with the most
        active *connected* synapses (those with permanence ≥ connectedPermanence),
        provided the segment has at least `minThreshold` active synapses.

        This matches the real HTM logic in Numenta's TemporalMemory.cpp.
        """
        bestCell = None
        bestSegment = None
        bestNumActive = -1

        for cell in self.cellsForColumn(column, self.cellsPerColumn):
            for segment in self.segmentsForCell(cell):
                activeSynapses = self.activeSynapses(segment, activeCells)
                if len(activeSynapses) >= minThreshold and len(activeSynapses) > bestNumActive:
                    bestNumActive = len(activeSynapses)
                    bestCell = cell
                    bestSegment = segment

        return bestCell, bestSegment, bestNumActive

    def synapsesForSegment(self, segment):
        """
        Return a list of all synapse IDs on the given segment.
        """
        return self._segment_to_synapses.get(segment, [])
    
    def dataForSynapse(self, synapse_id):
        """
        Return the SynapseData object for the given synapse ID.
        """
        return self._synapse_data[synapse_id]

    @property
    def segments(self):
        return self._segments
        # return [seg for cell_segs in self._cell_to_segments.values() for seg in cell_segs]

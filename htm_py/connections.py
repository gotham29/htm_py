# import logging
# import os

# # LOG_FILE = "htm_debug.log"

# # # Ensure only one handler is added
# # if not logging.getLogger().handlers:
# #     logging.basicConfig(
# #         level=logging.DEBUG,
# #         format='%(asctime)s [%(levelname)s] %(message)s',
# #         handlers=[
# #             logging.FileHandler(LOG_FILE, mode='w'),
# #             logging.StreamHandler()
# #         ]
# #     )
# # logger = logging.getLogger(__name__)

# from collections import defaultdict
# import itertools
# import numpy as np
# from typing import List, Set, Optional

# # from collections import defaultdict, namedtuple

# # SynapseData = namedtuple("SynapseData", ["presynapticCell", "permanence"])


# class Segment:
#     def __init__(self, segment_id, cell, iteration_num):
#         self.id = segment_id
#         self.cell = cell
#         self.iteration_num = iteration_num
#         self.synapses = []

#     def __eq__(self, other):
#         return isinstance(other, Segment) and self.id == other.id

#     def __hash__(self):
#         return hash(self.id)

#     def __repr__(self):
#         return f"<Segment id={self.id} cell={self.cell} synapses={len(self.synapses)}>"


# class Synapse:
#     def __init__(self, presynaptic_cell, permanence):
#         self.presynaptic_cell = presynaptic_cell
#         self.permanence = permanence


# class CellData:
#     def __init__(self, segments):
#         self.segments = segments


# class Connections:
#     def __init__(self, columnDimensions, cellsPerColumn, initialPermanence, connectedPermanence, maxSegmentsPerCell):
#         self.columnDimensions = columnDimensions
#         self.cellsPerColumn = cellsPerColumn
#         self.numCells = np.prod(columnDimensions) * cellsPerColumn
#         self.maxSegmentsPerCell = maxSegmentsPerCell
#         self.initialPermanence = initialPermanence
#         self.connectedPermanence = connectedPermanence

#         self._segments = set()  # should contain full Segment objects
#         self._segment_to_cell = {}
#         self._cell_to_segments = defaultdict(list)
#         self._segment_to_synapses = defaultdict(set)
#         self._segment_last_used = {}
#         self._next_segment_id = 0

#     def createSegment(self, cell: int, iteration_num: int) -> Segment:
#         """
#         Creates a new Segment object for the given cell and registers it.
#         If maxSegmentsPerCell is reached, evicts the least recently used segment.

#         Mirrors Numenta's behavior in TemporalMemory::createSegment().
#         """
#         if cell not in self._cell_to_segments:
#             self._cell_to_segments[cell] = []

#         segments = self._cell_to_segments[cell]

#         if self.maxSegmentsPerCell > 0 and len(segments) >= self.maxSegmentsPerCell:
#             # Evict least recently used segment (lowest last used iteration)
#             oldest_segment = min(
#                 segments,
#                 key=lambda s: self._segment_last_used.get(s.id, float('-inf'))
#             )
#             print(f"[DEBUG EVICT] Removing LRU segment {oldest_segment} from cell {cell}")
#             self.destroySegment(oldest_segment)

#         new_segment_id = self._next_segment_id
#         self._next_segment_id += 1

#         segment = Segment(new_segment_id, cell, iteration_num)
#         print(f"[DEBUG CREATE] Registering segment {segment} for cell {cell}")

#         self._cell_to_segments[cell].append(segment)
#         self._segment_to_cell[segment] = cell
#         self._segment_to_synapses[segment] = segment.synapses
#         self._segment_last_used[segment.id] = iteration_num
#         self._segments.add(segment)

#         return segment

#     def createSynapse(self, segment, presynaptic_cell, permanence):
#         syn = Synapse(presynaptic_cell, permanence)
#         segment.synapses.append(syn)
#         return syn

#     def createSynapses(self, segment, presynaptic_cells, initial_permanence, max_new_synapses):
#         count = 0
#         for cell in presynaptic_cells:
#             if count >= max_new_synapses:
#                 break
#             self.createSynapse(segment, cell, initial_permanence)
#             count += 1
#             print(f"[DEBUG] Created synapse on segment {segment.id} for presynaptic cell {cell}")

#     def segmentExists(self, segment):
#         return segment in self._segment_to_cell

#     def segmentsForCell(self, cell: int) -> List[Segment]:
#         """
#         Returns all segments associated with a given cell.
#         """
#         segments = self._cell_to_segments.get(cell, [])
#         print(f"[DEBUG FETCH] segmentsForCell({cell}) → {segments}")
#         return segments

#     def cellForSegment(self, segment):
#         return self._segment_to_cell.get(segment)

#     def cellsForColumn(self, column, cellsPerColumn):
#         start = column * cellsPerColumn
#         return list(range(start, start + cellsPerColumn))

#     def activeSynapses(self, segment, activeCells):
#         return {
#             syn for syn in segment.synapses
#             if syn.presynaptic_cell in activeCells
#         }

#     def connectedSynapses(self, segment, connectedPermanence):
#         return {
#             syn.presynaptic_cell for syn in segment.synapses
#             if syn.permanence >= connectedPermanence
#         }

#     def adaptSegment(
#         self,
#         segment: Segment,
#         active_presynaptic: Set[int],
#         permanence_increment: float,
#         permanence_decrement: float,
#         iteration_num: Optional[int] = None
#     ) -> None:
#         """
#         Reinforces active synapses and punishes inactive ones.
#         Updates LRU timestamp.
#         """
#         existing_synapses = {syn.presynaptic_cell: syn for syn in segment.synapses}

#         for presynaptic_cell, synapse in existing_synapses.items():
#             if presynaptic_cell in active_presynaptic:
#                 synapse.permanence = min(1.0, synapse.permanence + permanence_increment)
#             else:
#                 synapse.permanence = max(0.0, synapse.permanence - permanence_decrement)

#         # Update usage for LRU eviction policy
#         if iteration_num is not None:
#             self._segment_last_used[segment.id] = iteration_num

#     def activeSegmentsForCell(self, cell, prevActiveCells):
#         return [
#             seg for seg in self.segmentsForCell(cell)
#             if any(syn.presynaptic_cell in prevActiveCells and syn.permanence >= self.connectedPermanence
#                    for syn in seg.synapses)
#         ]

#     def decreaseSynapsePermanences(self, segment, amount):
#         for synapse in self._segments[segment]:
#             self.permanences[synapse] = max(0.0, self.permanences[synapse] - amount)

#     def segmentActive(self, segments, activePresynapticCells, threshold):
#         """
#         Returns True if any of the given segments has enough active synapses to be considered active.
#         If a single Segment is passed, evaluates that one.
        
#         Parameters:
#             segments (Segment or iterable of Segment): segment(s) to evaluate
#             activePresynapticCells (set[int]): previously active cells
#             threshold (int): number of active synapses required for activation
        
#         Returns:
#             bool: True if any segment is active
#         """
#         if isinstance(segments, Segment):
#             segments = [segments]  # wrap single segment in list

#         for segment in segments:
#             active_count = sum(
#                 1 for syn in segment.synapses
#                 if syn.presynaptic_cell in activePresynapticCells and syn.permanence >= self.connectedPermanence
#             )
#             if active_count >= threshold:
#                 return True

#         return False

#     def destroySegment(self, segment: Segment) -> None:
#         """Destroys a segment and removes all its synapses."""
#         print(f"[DEBUG DESTROY] Destroying segment {segment}")
#         cell = segment.cell
#         if segment in self._segments:
#             self._segments.remove(segment)
#         if cell in self._cell_to_segments:
#             self._cell_to_segments[cell] = [s for s in self._cell_to_segments[cell] if s != segment]
#         if segment in self._segment_to_cell:
#             del self._segment_to_cell[segment]
#         if segment in self._segment_to_synapses:
#             del self._segment_to_synapses[segment]
#         if segment.id in self._segment_last_used:
#             del self._segment_last_used[segment.id]

#     def getBestMatchingSegment(self, column, activeCells, minThreshold):
#         """
#         For the given column, return the cell and its segment with the most
#         active *connected* synapses (those with permanence ≥ connectedPermanence),
#         provided the segment has at least `minThreshold` active synapses.

#         This matches the real HTM logic in Numenta's TemporalMemory.cpp.
#         """
#         bestCell = None
#         bestSegment = None
#         bestNumActive = -1

#         for cell in self.cellsForColumn(column, self.cellsPerColumn):
#             for segment in self.segmentsForCell(cell):
#                 activeSynapses = self.activeSynapses(segment, activeCells)
#                 if len(activeSynapses) >= minThreshold and len(activeSynapses) > bestNumActive:
#                     bestNumActive = len(activeSynapses)
#                     bestCell = cell
#                     bestSegment = segment

#         return bestCell, bestSegment, bestNumActive

#     def synapsesForSegment(self, segment):
#         """
#         Return a list of all synapse IDs on the given segment.
#         """
#         return self._segment_to_synapses.get(segment, [])
    
#     def dataForSynapse(self, synapse_id):
#         """
#         Return the SynapseData object for the given synapse ID.
#         """
#         return self._synapse_data[synapse_id]

#     def numSegments(self, cell: int) -> int:
#         return len(self.segmentsForCell(cell))

#     @property
#     def segments(self):
#         return self._segments
#         # return [seg for cell_segs in self._cell_to_segments.values() for seg in cell_segs]


from typing import Dict, Set, List, Optional


class Synapse:
    def __init__(self, presynaptic_cell: int, permanence: float):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence

    def __eq__(self, other):
        if not isinstance(other, Synapse):
            return False
        return (
            self.presynaptic_cell == other.presynaptic_cell and
            abs(self.permanence - other.permanence) < 1e-6
        )

    def __hash__(self):
        # Round permanence to avoid precision drift
        return hash((self.presynaptic_cell, round(self.permanence, 6)))

    def __repr__(self):
        return f"<Synapse pre={self.presynaptic_cell} perm={self.permanence:.2f}>"


class Segment:
    def __init__(self, id: int, cell: int, iteration_num: int):
        self.id = id
        self.cell = cell
        self.synapses = set()
        self.iteration_num = iteration_num

    def __eq__(self, other):
        if not isinstance(other, Segment):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<Segment id={self.id} cell={self.cell} synapses={len(self.synapses)}>"


class Connections:
    def __init__(
        self,
        numberOfColumns: int,
        cellsPerColumn: int,
        initialPermanence: float,
        connectedPermanence: float,
        permanenceIncrement: float,
        permanenceDecrement: float,
        maxSegmentsPerCell: int,
        maxSynapsesPerSegment: int,
    ):
        self.cellsPerColumn = cellsPerColumn
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment

        self._cell_to_segments: Dict[int, List[Segment]] = {}
        self._segment_to_cell: Dict[Segment, int] = {}
        self._segment_to_synapses: Dict[Segment, Set[Synapse]] = {}
        self._segment_last_used: Dict[int, int] = {}
        self._segments: Set[Segment] = set()
        self._next_segment_id = 0

    def segmentsForCell(self, cell: int) -> List[Segment]:
        segments = self._cell_to_segments.get(cell, [])
        print(f"[DEBUG FETCH] segmentsForCell({cell}) → {segments}")
        return segments

    def cellForSegment(self, segment: Segment) -> int:
        return self._segment_to_cell[segment]

    def synapsesForSegment(self, segment: Segment) -> Set[Synapse]:
        return self._segment_to_synapses.get(segment, set())

    def createSegment(self, cell: int, iteration_num: int) -> Optional[Segment]:
        """
        Creates a new Segment object for the given cell and registers it.
        If maxSegmentsPerCell is reached, evicts the least recently used segment.
        If maxSegmentsPerCell == 0, segment creation is disabled.
        """
        if self.maxSegmentsPerCell == 0:
            print(f"[DEBUG SKIP] Segment creation disabled (maxSegmentsPerCell=0) for cell {cell}")
            return None

        if cell not in self._cell_to_segments:
            self._cell_to_segments[cell] = []

        segments = self._cell_to_segments[cell]

        if self.maxSegmentsPerCell > 0 and len(segments) >= self.maxSegmentsPerCell:
            # Evict least recently used segment (lowest last used iteration)
            oldest_segment = min(
                segments,
                key=lambda s: self._segment_last_used.get(s.id, float('-inf'))
            )
            print(f"[DEBUG EVICT] Removing LRU segment {oldest_segment} from cell {cell}")
            self.destroySegment(oldest_segment)

        new_segment_id = self._next_segment_id
        self._next_segment_id += 1

        segment = Segment(new_segment_id, cell, iteration_num)
        print(f"[DEBUG CREATE] Registering segment {segment} for cell {cell}")

        self._cell_to_segments[cell].append(segment)
        self._segment_to_cell[segment] = cell
        self._segment_to_synapses[segment] = segment.synapses
        self._segment_last_used[segment.id] = iteration_num
        self._segments.add(segment)

        return segment

    def destroySegment(self, segment: Segment):
        cell = self._segment_to_cell.pop(segment)
        if segment in self._cell_to_segments[cell]:
            self._cell_to_segments[cell].remove(segment)

        self._segment_to_synapses.pop(segment, None)
        self._segment_last_used.pop(segment.id, None)
        self._segments.discard(segment)
        print(f"[DEBUG DESTROY] Segment {segment} removed from cell {cell}")

    def createSynapse(self, segment: Segment, presynaptic_cell: int, permanence: float):
        syn = Synapse(presynaptic_cell, permanence)
        self._segment_to_synapses[segment].add(syn)
        segment.synapses.add(syn)

    def adaptSegment(self, segment: Segment, prev_active_dense: List[bool],
                    permanenceInc: float, permanenceDec: float):
        synapses = self._segment_to_synapses[segment]
        for syn in synapses:
            if 0 <= syn.presynaptic_cell < len(prev_active_dense):
                if prev_active_dense[syn.presynaptic_cell]:
                    syn.permanence = min(1.0, syn.permanence + permanenceInc)
                else:
                    syn.permanence = max(0.0, syn.permanence - permanenceDec)

    def getBestMatchingSegment(
        self,
        cell: int,
        prev_active_dense: List[bool],
        min_threshold: int,
    ) -> Optional[Segment]:
        best_segment = None
        max_active = min_threshold
        for segment in self.segmentsForCell(cell):
            active = sum(
                1
                for syn in self.synapsesForSegment(segment)
                if 0 <= syn.presynaptic_cell < len(prev_active_dense)
                and prev_active_dense[syn.presynaptic_cell]
            )
            if active >= max_active:
                best_segment = segment
                max_active = active
        return best_segment

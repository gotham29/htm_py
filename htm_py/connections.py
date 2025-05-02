import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# @dataclass
# class Synapse:
#     presynapticCell: int
#     permanence: float

class Synapse:
    def __init__(self, presynaptic_cell, permanence):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence

# class Segment:
#     _id_counter = 0

#     def __init__(self, cell):
#         self.cell = cell
#         self.synapses = []
#         self.id = Segment._id_counter
#         Segment._id_counter += 1

class Segment:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class Connections:
    # def __init__(self, defaultPerm=0.21):
    #     self._segments = defaultdict(list)  # cell -> set of segments
    #     self._active_segments = set()
    #     self._segment_id_counter = 0
    #     self.defaultPerm = defaultPerm

    def __init__(self, num_cells, defaultPerm=0.21):
        self.num_cells = num_cells
        self._segment_to_synapses = defaultdict(list)
        self._segment_owner = {}  # segment → cell
        self._cell_to_segments = defaultdict(list)
        self._next_segment_id = 0
        self._next_synapse_id = 0
        self._segments = defaultdict(list)  # cell -> set of segments
        self._active_segments = set()
        self._segment_id_counter = 0
        self.defaultPerm = defaultPerm


    def create_segment(self, cell, presynapticCells=None, initialPermanence=0.21, maxNewSynapseCount=None):
        presynapticCells = presynapticCells or []
        # segment = Segment(cell)
        # segment.id = self._segment_id_counter
        # self._segment_id_counter += 1
        # self._segments[cell].append(segment)
        segment = Segment(self._next_segment_id)
        self._next_segment_id += 1
        self._segment_owner[segment] = cell
        self._cell_to_segments[cell].append(segment)
        self._segment_to_synapses[segment] = []

        if presynapticCells:
            self.grow_synapses(
                segment,
                presynapticCells,
                initialPermanence,
                max_new_synapses=maxNewSynapseCount
            )

        return segment

    def grow_synapses(self, segment, presynaptic_cells, initial_permanence, max_new_synapses=None):
        """
        Add new synapses from the given presynaptic cells to this segment.
        Only adds synapses to presynaptic cells not already connected.
        Limits the number to `max_new_synapses` if specified.
        """
        existing_presyn = {syn.presynapticCell for syn in segment.synapses}
        candidates = [cell for cell in sorted(presynaptic_cells) if cell not in existing_presyn]

        if max_new_synapses is not None:
            candidates = candidates[:max_new_synapses]

        for cell in candidates:
            self.create_synapse(segment, cell, initial_permanence)

    # def create_synapse(self, segment, presynapticCell, permanence):
    #     synapse = Synapse(presynapticCell, permanence)
    #     segment.synapses.append(synapse)
    #     return synapse

    def create_synapse(self, segment, presynaptic_cell, permanence):
        synapse = Synapse(presynaptic_cell, permanence)
        self._segment_to_synapses[segment].append(synapse)
        return synapse

    def segments_for_cell(self, cell):
        return self._segments[cell]

    def segment_overlap(self, segment, activity_set, connected_only=False, permanence_connected=0.0):
        overlap = 0
        for syn in segment.synapses:
            if connected_only and syn.permanence < permanence_connected:
                continue
            if syn.presynapticCell in activity_set:
                overlap += 1
        return overlap

    def cell_for_segment(self, segment):
        return segment.cell

    def get_active_segments(self):
        return list(self._active_segments)

    def mark_segment_active(self, segment):
        self._active_segments.add(segment)

    def clear_active_segments(self):
        self._active_segments.clear()

    def allCells(self):
        return list(self._segments.keys())

    def adapt_segment(self, segment, prevWinnerCells, positive_reinforcement, perm_inc=0.1, perm_dec=0.1):
        for synapse in segment.synapses:
            if synapse.presynapticCell in prevWinnerCells:
                if positive_reinforcement:
                    synapse.permanence = min(1.0, synapse.permanence + perm_inc)
                else:
                    synapse.permanence = max(0.0, synapse.permanence - perm_dec)
            else:
                if positive_reinforcement:
                    synapse.permanence = max(0.0, synapse.permanence - perm_dec)

    def clear_active_segments(self):
        # Stub — not needed for basic tests
        pass

    def get_active_segments(self):
        # Stub — not needed for basic tests
        return []

    def segments(self):
        return [segment for segment_list in self._segments.values() for segment in segment_list]

    def synapses_for_segment(self, segment):
        """
        Return a list of synapse objects for the given segment.
        """
        return self._segment_to_synapses.get(segment, [])

    # def num_active_connected_synapses(self, segment, active_cells, connected_perm):
    #     """
    #     Count number of synapses on `segment` whose presynaptic cell is in `active_cells`
    #     and whose permanence is >= connected_perm.
    #     """
    #     count = 0
    #     for syn in self.synapses_for_segment(segment):
    #         if syn.presynapticCell in active_cells and syn.permanence >= connected_perm:
    #             count += 1
    #     return count
    
    def num_active_connected_synapses(self, segment, active_cells, connected_perm):
        """
        Count number of synapses on `segment` whose presynaptic cell is in `active_cells`
        and whose permanence is >= connected_perm.
        """
        return sum(
            1 for s in self.synapses_for_segment(segment)
            if s.presynaptic_cell in active_cells and s.permanence >= connected_perm
        )

    def clear(self):
        """Reset all internal state of the Connections (for test setup)."""
        self._segments.clear()
        self._active_segments.clear()
        self._nextSegmentId = 0
        self._nextSynapseId = 0
        self._segment_id_counter = 0

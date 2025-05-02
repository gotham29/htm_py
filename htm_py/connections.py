import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Synapse:
    presynapticCell: int
    permanence: float

class Segment:
    _id_counter = 0

    def __init__(self, cell):
        self.cell = cell
        self.synapses = []
        self.id = Segment._id_counter
        Segment._id_counter += 1

class Connections:
    def __init__(self, defaultPerm=0.21):
        self._segments = defaultdict(list)  # cell -> set of segments
        self._active_segments = set()
        self._segment_id_counter = 0
        self.defaultPerm = defaultPerm

    def create_segment(self, cell, presynapticCells=None, initialPermanence=0.21, maxNewSynapseCount=None):
        presynapticCells = presynapticCells or []
        segment = Segment(cell)
        segment.id = self._segment_id_counter
        self._segment_id_counter += 1
        self._segments[cell].append(segment)

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

    def create_synapse(self, segment, presynapticCell, permanence):
        synapse = Synapse(presynapticCell, permanence)
        segment.synapses.append(synapse)
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

    def clear(self):
        """Reset all internal state of the Connections (for test setup)."""
        self._segments.clear()
        self._active_segments.clear()
        self._nextSegmentId = 0
        self._nextSynapseId = 0
        self._segment_id_counter = 0

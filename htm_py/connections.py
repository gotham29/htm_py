# from collections import defaultdict


# class Segment:
#     def __init__(self, id):
#         self.id = id

#     def __hash__(self):
#         return hash(self.id)

#     def __eq__(self, other):
#         return self.id == other.id


# class Synapse:
#     def __init__(self, presynaptic_cell, permanence):
#         self.presynaptic_cell = presynaptic_cell
#         self.permanence = permanence


# class Connections:
#     def __init__(self, num_cells, maxSegmentsPerCell=None, maxSynapsesPerSegment=None):
#         self.num_cells = num_cells
#         self.maxSegmentsPerCell = maxSegmentsPerCell
#         self.maxSynapsesPerSegment = maxSynapsesPerSegment
#         self._next_segment_id = 0

#         self._segment_to_synapses = defaultdict(list)
#         self._segment_owner = {}  # segment -> cell
#         self._cell_to_segments = defaultdict(list)  # cell -> [segments]

#     def create_segment(self, cell):
#         segments = self._cell_to_segments[cell]
#         if self.maxSegmentsPerCell is not None and len(segments) >= self.maxSegmentsPerCell:
#             oldest = segments.pop(0)  # ✅ enforce FIFO for segment eviction
#             self.destroy_segment(oldest)

#         segment = Segment(self._next_segment_id)
#         self._next_segment_id += 1
#         self._segment_owner[segment] = cell
#         self._cell_to_segments[cell].append(segment)
#         self._segment_to_synapses[segment] = []
#         return segment

#     def destroy_segment(self, segment):
#         cell = self._segment_owner.pop(segment, None)
#         if cell is not None:
#             if segment in self._cell_to_segments[cell]:
#                 self._cell_to_segments[cell].remove(segment)
#         self._segment_to_synapses.pop(segment, None)

#     def segments_for_cell(self, cell):
#         return list(self._cell_to_segments.get(cell, []))  # ✅ Return copy to avoid external mutation

#     def cell_for_segment(self, segment):
#         return self._segment_owner[segment]

#     def create_synapse(self, segment, presynaptic_cell, permanence):
#         synapses = self._segment_to_synapses[segment]
#         if self.maxSynapsesPerSegment is not None and len(synapses) >= self.maxSynapsesPerSegment:
#             # ✅ prune weakest to make room
#             to_remove = sorted(synapses, key=lambda s: s.permanence)[:len(synapses) - self.maxSynapsesPerSegment + 1]
#             for s in to_remove:
#                 self.destroy_synapse(s)

#         synapse = Synapse(presynaptic_cell, permanence)
#         self._segment_to_synapses[segment].append(synapse)
#         return synapse

#     def destroy_synapse(self, synapse_to_remove):
#         for seg, syns in self._segment_to_synapses.items():
#             if synapse_to_remove in syns:
#                 syns.remove(synapse_to_remove)
#                 # ✅ If segment is now empty, destroy it
#                 if not syns:
#                     self.destroy_segment(seg)
#                 return

#     def synapses_for_segment(self, segment):
#         return list(self._segment_to_synapses.get(segment, []))

#     def num_synapses(self, segment):
#         return len(self._segment_to_synapses.get(segment, []))

#     def data_for_synapse(self, synapse):
#         return synapse

#     def update_permanence(self, synapse, new_perm):
#         synapse.permanence = new_perm

#     def permanence(self, synapse):
#         return synapse.permanence

#     def presynaptic_cell(self, synapse):
#         return synapse.presynaptic_cell

#     def num_active_connected_synapses(self, segment, active_cells, connected_perm):
#         return sum(
#             1 for s in self._segment_to_synapses.get(segment, [])
#             if s.presynaptic_cell in active_cells and s.permanence >= connected_perm
#         )

#     def num_active_potential_synapses(self, segment, active_cells):
#         return sum(
#             1 for s in self._segment_to_synapses.get(segment, [])
#             if s.presynaptic_cell in active_cells
#         )



from collections import defaultdict

class Segment:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class Synapse:
    def __init__(self, presynaptic_cell, permanence):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence

class Connections:
    def __init__(self, num_cells, maxSegmentsPerCell=None, maxSynapsesPerSegment=None):
        self.num_cells = num_cells
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment

        self._next_segment_id = 0
        self._segment_to_synapses = defaultdict(list)
        self._segment_owner = {}
        self._cell_to_segments = defaultdict(list)

    def create_segment(self, cell, last_used_map=None, current_iter=None):
        segments = self._cell_to_segments[cell]
        
        if self.maxSegmentsPerCell is not None and len(segments) >= self.maxSegmentsPerCell:
            # Find LRU segment
            if last_used_map is not None and current_iter is not None:
                oldest_seg = min(
                    segments,
                    key=lambda s: last_used_map.get(s, -1)
                )
                print(f"[destroySegment] Destroying LRU segment {oldest_seg.id} on cell {cell}")
                self.destroy_segment(oldest_seg)
            else:
                # Fall back to FIFO if no iteration info
                print(f"[destroySegment] Destroying first segment (FIFO) on cell {cell}")
                self.destroy_segment(segments[0])

        segment = Segment(self._next_segment_id)
        self._next_segment_id += 1
        self._segment_owner[segment] = cell
        self._cell_to_segments[cell].append(segment)
        self._segment_to_synapses[segment] = []
        
        print(f"[createSegment] Created segment {segment.id} for cell {cell}")
        return segment

    # def create_segment(self, cell):
    #     segments = self._cell_to_segments[cell]
    #     if self.maxSegmentsPerCell is not None and len(segments) >= self.maxSegmentsPerCell:
    #         oldest = segments[0]
    #         self.destroy_segment(oldest)

    #     segment = Segment(self._next_segment_id)
    #     self._next_segment_id += 1
    #     self._segment_owner[segment] = cell
    #     self._cell_to_segments[cell].append(segment)
    #     self._segment_to_synapses[segment] = []

    #     return segment

    def destroy_segment(self, segment):
        cell = self._segment_owner.pop(segment, None)
        if cell is not None:
            segs = self._cell_to_segments[cell]
            if segment in segs:
                segs.remove(segment)
            if not segs:
                del self._cell_to_segments[cell]
        if segment in self._segment_to_synapses:
            del self._segment_to_synapses[segment]

    def segments_for_cell(self, cell):
        return self._cell_to_segments.get(cell, [])

    def cell_for_segment(self, segment):
        return self._segment_owner[segment]

    def create_synapse(self, segment, presynaptic_cell, permanence):
        synapses = self._segment_to_synapses[segment]
        if self.maxSynapsesPerSegment is not None and len(synapses) >= self.maxSynapsesPerSegment:
            # Remove the lowest permanence synapse to make room
            weakest = min(synapses, key=lambda s: s.permanence, default=None)
            if weakest:
                synapses.remove(weakest)

        synapse = Synapse(presynaptic_cell, permanence)
        synapses.append(synapse)
        return synapse

    def destroy_synapse(self, synapse):
        for syns in self._segment_to_synapses.values():
            if synapse in syns:
                syns.remove(synapse)
                break

    def synapses_for_segment(self, segment):
        return self._segment_to_synapses.get(segment, [])

    def num_synapses(self, segment):
        return len(self._segment_to_synapses.get(segment, []))

    def data_for_synapse(self, synapse):
        return synapse

    def update_permanence(self, synapse, new_perm):
        synapse.permanence = new_perm

    def permanence(self, synapse):
        return synapse.permanence

    def presynaptic_cell(self, synapse):
        return synapse.presynaptic_cell

    def num_active_connected_synapses(self, segment, active_cells, connected_perm):
        return sum(
            1 for s in self.synapses_for_segment(segment)
            if s.presynaptic_cell in active_cells and s.permanence >= connected_perm
        )

    def num_active_potential_synapses(self, segment, active_cells):
        return sum(
            1 for s in self.synapses_for_segment(segment)
            if s.presynaptic_cell in active_cells
        )

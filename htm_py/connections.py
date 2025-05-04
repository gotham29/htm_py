# # # from collections import defaultdict

# # # class Synapse:
# # #     def __init__(self, presynaptic_cell, permanence):
# # #         self.presynaptic_cell = presynaptic_cell
# # #         self.permanence = permanence

# # # class CellData:
# # #     def __init__(self, segments):
# # #         self.segments = segments

# # # class Segment:
# # #     _id_counter = 0

# # #     def __init__(self):
# # #         self.id = Segment._id_counter
# # #         Segment._id_counter += 1
# # #         self.synapses = []

# # #     def __eq__(self, other):
# # #         return isinstance(other, Segment) and self.id == other.id

# # #     def __hash__(self):
# # #         return hash(self.id)

# # # class Connections:
# # #     def __init__(self, numCells):
# # #         self.numCells = numCells
# # #         self._segments = []
# # #         self._segment_to_cell = {}
# # #         self.cell_to_segments = defaultdict(list)

# # #     def create_segment(self, cell):
# # #         seg = Segment()
# # #         self._segments.append(seg)
# # #         self._segment_to_cell[seg] = cell
# # #         self.cell_to_segments[cell].append(seg)
# # #         # print(f"[CREATE] Segment {seg} attached to cell {cell}")
# # #         return seg

# # #     def create_synapse(self, segment, presynaptic_cell, permanence):
# # #         synapse = Synapse(presynaptic_cell, permanence)
# # #         segment.synapses.append(synapse)
# # #         return synapse

# # #     def segments_for_cell(self, cell):
# # #         return list(self.cell_to_segments.get(cell, []))

# # #     def segment_for_cell(self, cell):
# # #         segments = self.segments_for_cell(cell)
# # #         return segments[0] if segments else None

# # #     def synapses_for_segment(self, segment):
# # #         return list(segment.synapses)

# # #     def num_synapses(self, segment):
# # #         return len(self.synapses_for_segment(segment))

# # #     def get_cell_for_segment(self, segment):
# # #         return self._segment_to_cell.get(segment, None)

# # #     def delete_segment(self, segment):
# # #         cell = self._segment_to_cell.get(segment)
# # #         if cell is not None:
# # #             self.cell_to_segments[cell].remove(segment)
# # #         if segment in self._segments:
# # #             self._segments.remove(segment)
# # #         if segment in self._segment_to_cell:
# # #             del self._segment_to_cell[segment]

# # #     def num_active_connected_synapses(self, segment, active_cells, connected_perm):
# # #         return sum(
# # #             1 for s in self.synapses_for_segment(segment)
# # #             if s.presynaptic_cell in active_cells and s.permanence >= connected_perm
# # #         )

# # #     def num_active_potential_synapses(self, segment, active_cells):
# # #         return sum(
# # #             1 for s in self.synapses_for_segment(segment)
# # #             if s.presynaptic_cell in active_cells
# # #         )

# # #     def add_synapses(self, segment, presynaptic_cells, permanence, max_new_synapses):
# # #         """
# # #         Add synapses to the given segment from the provided presynaptic cells,
# # #         up to the given max_new_synapses limit.
# # #         """
# # #         count = 0
# # #         for cell in presynaptic_cells:
# # #             if count >= max_new_synapses:
# # #                 break
# # #             self.create_synapse(segment, cell, permanence)
# # #             count += 1

# # #     def data_for_cell(self, cell):
# # #         return type("CellData", (), {"segments": self.cell_to_segments[cell]})()
    
# # #     def presynaptic_cell(self, synapse):
# # #         return synapse.presynaptic_cell

# # #     def permanence(self, synapse):
# # #         return synapse.permanence

# # #     def update_permanence(self, synapse, new_permanence):
# # #         synapse.permanence = new_permanence

# # #     # def cell_for_segment(self, segment):
# # #     #     return self.segment_to_cell[segment]

# # #     # def segment_for_cell(self, cell):
# # #     #     return self.cell_to_segments[cell]



# # import numpy as np
# # from collections import defaultdict

# # class Connections:
# #     def __init__(self, num_cells):
# #         self.num_cells = num_cells
# #         self._segments = defaultdict(set)        # cell → set(segment)
# #         self._segment_synapses = defaultdict(set)  # segment → set(synapse)
# #         self._synapse_to_cell = {}              # synapse → presynaptic cell
# #         self._permanences = {}                  # synapse → permanence
# #         self._next_segment = 0
# #         self._next_synapse = 0

# #     def segments_for_cell(self, cell):
# #         return list(self._segments[cell])

# #     def synapses_for_segment(self, segment):
# #         return list(self._segment_synapses[segment])

# #     def synapse_to_cell(self, synapse):
# #         return self._synapse_to_cell[synapse]

# #     def segment_for_synapse(self, synapse):
# #         for seg, syns in self._segment_synapses.items():
# #             if synapse in syns:
# #                 return seg
# #         return None

# #     def permanences_for_segment(self, segment):
# #         return {syn: self._permanences[syn] for syn in self._segment_synapses[segment]}

# #     def segment_exists(self, segment):
# #         return segment in self._segment_synapses

# #     def create_synapse(self, segment, presynaptic_cell, permanence):
# #         return self.create_synapses(segment, [presynaptic_cell], permanence)[0]

# #     def create_segment(self, cell):
# #         segment = self._next_segment
# #         self._next_segment += 1
# #         self._segments[cell].add(segment)
# #         self._segment_synapses[segment] = set()
# #         return segment

# #     def create_synapses(self, segment, source_cells, initial_permanence):
# #         for cell in source_cells:
# #             synapse = self._next_synapse
# #             self._next_synapse += 1
# #             self._synapse_to_cell[synapse] = cell
# #             self._permanences[synapse] = initial_permanence
# #             self._segment_synapses[segment].add(synapse)

# #     def active_synapses(self, segment, prev_active_cells, connected_only=True):
# #         synapses = []
# #         for syn in self._segment_synapses[segment]:
# #             src = self._synapse_to_cell[syn]
# #             if src in prev_active_cells:
# #                 if connected_only:
# #                     if self._permanences[syn] >= 0.5:
# #                         synapses.append(syn)
# #                 else:
# #                     synapses.append(syn)
# #         return synapses

# #     def segment_active(self, segment, prev_active_cells, activation_threshold, connected_only=True):
# #         active = self.active_synapses(segment, prev_active_cells, connected_only)
# #         return len(active) >= activation_threshold

# #     def adapt_segment(self, segment, active_synapses, initial_permanence):
# #         for syn in self._segment_synapses[segment]:
# #             if syn in active_synapses:
# #                 self._permanences[syn] = min(1.0, self._permanences[syn] + 0.05)
# #             else:
# #                 self._permanences[syn] = max(0.0, self._permanences[syn] - 0.05)

# #     def increment_permanences(self, segment, synapses):
# #         for syn in synapses:
# #             if syn in self._segment_synapses[segment]:
# #                 self._permanences[syn] = min(1.0, self._permanences[syn] + 0.05)

# #     def decrement_permanences(self, segment, synapses):
# #         for syn in synapses:
# #             if syn in self._segment_synapses[segment]:
# #                 self._permanences[syn] = max(0.0, self._permanences[syn] - 0.05)

# #     def delete_segment(self, segment):
# #         if segment in self._segment_synapses:
# #             for syn in self._segment_synapses[segment]:
# #                 del self._permanences[syn]
# #                 del self._synapse_to_cell[syn]
# #             del self._segment_synapses[segment]
# #             # Also remove from any cell
# #             for cell in self._segments:
# #                 self._segments[cell].discard(segment)

# #     @property
# #     def data(self):
# #         return self._segments  # or self._synapses depending on structure


# from collections import defaultdict
# import logging

# logger = logging.getLogger("htm_tm")
# logger.setLevel(logging.DEBUG)

# # Optional: only add handler if none exist (avoids duplicate logs)
# if not logger.handlers:
#     fh = logging.FileHandler("tm_debug.log", mode='a')  # append mode to keep logs
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)


# class Connections:
#     def __init__(self, num_cells):
#         self._next_segment_id = 0
#         self._next_synapse_id = 0
#         self.num_cells = num_cells

#         self._segments = defaultdict(list)  # cell -> list of segment ids
#         self._segment_to_cell = {}          # segment id -> cell id
#         self._synapses = defaultdict(list)  # segment id -> list of synapse ids
#         self._synapse_to_data = {}          # synapse id -> (source_cell, permanence)

#     def create_segment(self, cell):
#         seg = self._next_segment_id
#         self._next_segment_id += 1
#         self._segments[cell].append(seg)
#         self._segment_to_cell[seg] = cell
#         return seg

#     def create_synapses(self, segment, presynaptic_cells, permanence):
#         ids = []
#         for cell in presynaptic_cells:
#             syn = self._next_synapse_id
#             self._next_synapse_id += 1
#             self._synapses[segment].append(syn)
#             self._synapse_to_data[syn] = (cell, permanence)
#             ids.append(syn)

#             cell_logged, perm = self._synapse_to_data[syn]
#             logger.debug(f"[Connections] Creating synapse: segment={segment}, cell={cell_logged}, perm={perm}")

#         return ids

#     def create_synapse(self, segment, presynaptic_cell, permanence):
#         return self.create_synapses(segment, [presynaptic_cell], permanence)[0]

#     def synapses_for_segment(self, segment):
#         return self._synapses[segment]

#     def synapse_to_cell(self, synapse):
#         return self._synapse_to_data[synapse][0]

#     def permanence(self, synapse):
#         return self._synapse_to_data[synapse][1]

#     def segments_for_cell(self, cell):
#         segments = list(self._segments.get(cell, []))
#         logging.debug(f"[Connections] Segments for cell {cell}: {segments}")
#         return segments

#     def segment_active(self, segment, active_cells, threshold, connected_only=True, connected_perm=0.5):
#         count = 0
#         for syn in self._synapses[segment]:
#             src, perm = self._synapse_to_data[syn]
#             if src in active_cells and (not connected_only or perm >= connected_perm):
#                 count += 1
#         return count >= threshold

#     def active_synapses(self, segment, active_cells, connected_only=True, connected_perm=0.5):
#         result = []
#         for syn in self._synapses[segment]:
#             src, perm = self._synapse_to_data[syn]
#             if src in active_cells and (not connected_only or perm >= connected_perm):
#                 result.append(syn)
#         return result

#     def adapt_segment(self, segment, active_synapses, initial_permanence,
#                       inc=0.05, dec=0.05, connected_perm=0.5):
#         for syn in self._synapses[segment]:
#             src, perm = self._synapse_to_data[syn]
#             if syn in active_synapses:
#                 new_perm = min(1.0, perm + inc)
#             else:
#                 new_perm = max(0.0, perm - dec)
#             self._synapse_to_data[syn] = (src, new_perm)

#     @property
#     def data(self):
#         return {
#             "segments": dict(self._segments),
#             "synapses": dict(self._synapses),
#             "synapse_data": dict(self._synapse_to_data)
#         }


from collections import defaultdict
import itertools
import numpy as np

class Segment:
    _ids = itertools.count(0)

    def __init__(self):
        self.id = next(Segment._ids)
        self.synapses = []

    def __eq__(self, other):
        return isinstance(other, Segment) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Synapse:
    def __init__(self, presynaptic_cell, permanence):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence


class CellData:
    def __init__(self, segments):
        self.segments = segments


class Connections:
    def __init__(self, columnDimensions, cellsPerColumn, initialPermanence, connectedPermanence):
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.numCells = np.prod(columnDimensions) * cellsPerColumn

        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence

        self._segments = []
        self._segment_to_cell = {}
        self.cell_to_segments = defaultdict(list)

    def createSegment(self, cell):
        """
        Creates a new segment on the given cell.
        If the cell already has maxSegmentsPerCell segments, evicts the least recently used one.
        """
        if cell not in self.cellToSegments:
            self.cellToSegments[cell] = []

        segments = self.cellToSegments[cell]
        if len(segments) >= self.maxSegmentsPerCell:
            # Evict least recently used segment
            oldest_segment = min(segments, key=lambda s: self.segmentLastUsed.get(s, -1))
            self.destroySegment(oldest_segment)

        new_segment = self.nextSegmentId
        self.nextSegmentId += 1

        self.segmentToCell[new_segment] = cell
        self.segmentToSynapses[new_segment] = set()
        self.cellToSegments[cell].append(new_segment)
        self.segmentLastUsed[new_segment] = self.iterationNum

        return new_segment

    def create_synapse(self, segment, presynaptic_cell, permanence):
        syn = Synapse(presynaptic_cell, permanence)
        segment.synapses.append(syn)
        return syn

    def create_synapses(self, segment, presynaptic_cells, initialPermanence, max_new_synapses):
        count = 0
        for cell in presynaptic_cells:
            if count >= max_new_synapses:
                break
            self.create_synapse(segment, cell, initialPermanence)
            count += 1

    def segmentExists(self, segment):
        return segment in self._segment_to_cell

    def segmentsForCell(self, cell):
        return self.cell_to_segments.get(cell, [])

    def cellForSegment(self, segment):
        return self._segment_to_cell.get(segment, None)

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

    def adaptSegment(self, segment, prevActiveCells, perm_inc, perm_dec, connected_perm, maxSynapsesPerSegment):
        # Reinforce existing synapses
        for syn in segment.synapses:
            if syn.presynaptic_cell in prevActiveCells:
                syn.permanence = min(1.0, syn.permanence + perm_inc)
            else:
                syn.permanence = max(0.0, syn.permanence - perm_dec)

        # Optionally grow new synapses to active cells not already present
        existing = {syn.presynaptic_cell for syn in segment.synapses}
        grow_cells = [c for c in prevActiveCells if c not in existing]
        n_grow = max(0, maxSynapsesPerSegment - len(segment.synapses))
        for cell in grow_cells[:n_grow]:
            self.create_synapse(segment, cell, self.initialPermanence)

    def activeSegmentsForCell(self, cell, prevActiveCells):
        return [
            seg for seg in self.segmentsForCell(cell)
            if any(syn.presynaptic_cell in prevActiveCells and syn.permanence >= self.connectedPermanence
                   for syn in seg.synapses)
        ]

    def decreaseSynapsePermanences(self, segment, amount):
        for synapse in self.segments[segment]:
            self.permanences[synapse] = max(0.0, self.permanences[synapse] - amount)

    @property
    def segments(self):
        return self._segments

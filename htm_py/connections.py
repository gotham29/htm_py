from collections import defaultdict

class Synapse:
    def __init__(self, presynaptic_cell, permanence):
        self.presynaptic_cell = presynaptic_cell
        self.permanence = permanence

class CellData:
    def __init__(self, segments):
        self.segments = segments

class Segment:
    _id_counter = 0

    def __init__(self):
        self.id = Segment._id_counter
        Segment._id_counter += 1
        self.synapses = []

    def __eq__(self, other):
        return isinstance(other, Segment) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Connections:
    def __init__(self, numCells):
        self.numCells = numCells
        self._segments = []
        self._segment_to_cell = {}
        self.cell_to_segments = defaultdict(list)

    def create_segment(self, cell):
        seg = Segment()
        self._segments.append(seg)
        self._segment_to_cell[seg] = cell
        self.cell_to_segments[cell].append(seg)
        print(f"[CREATE] Segment {seg} attached to cell {cell}")
        return seg

    def create_synapse(self, segment, presynaptic_cell, permanence):
        synapse = Synapse(presynaptic_cell, permanence)
        segment.synapses.append(synapse)
        return synapse

    def segments_for_cell(self, cell):
        return list(self.cell_to_segments.get(cell, []))

    def segment_for_cell(self, cell):
        segments = self.segments_for_cell(cell)
        return segments[0] if segments else None

    def synapses_for_segment(self, segment):
        return list(segment.synapses)

    def get_cell_for_segment(self, segment):
        return self._segment_to_cell.get(segment, None)

    def delete_segment(self, segment):
        cell = self._segment_to_cell.get(segment)
        if cell is not None:
            self.cell_to_segments[cell].remove(segment)
        if segment in self._segments:
            self._segments.remove(segment)
        if segment in self._segment_to_cell:
            del self._segment_to_cell[segment]

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

    def add_synapses(self, segment, presynaptic_cells, permanence, max_new_synapses):
        """
        Add synapses to the given segment from the provided presynaptic cells,
        up to the given max_new_synapses limit.
        """
        count = 0
        for cell in presynaptic_cells:
            if count >= max_new_synapses:
                break
            self.create_synapse(segment, cell, permanence)
            count += 1

    def data_for_cell(self, cell):
        return type("CellData", (), {"segments": self.cell_to_segments[cell]})()
    
    def presynaptic_cell(self, synapse):
        return synapse.presynaptic_cell

    def permanence(self, synapse):
        return synapse.permanence

    def update_permanence(self, synapse, new_permanence):
        synapse.permanence = new_permanence

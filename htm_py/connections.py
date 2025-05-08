from collections import defaultdict


class Connections:
    def __init__(self, num_cells):
        self.num_cells = num_cells
        self._cell_to_segments = defaultdict(list)        # Cell -> [Segment]
        self._segment_to_synapses = defaultdict(list)     # Segment -> [Synapse]
        self._synapse_data = dict()                       # Synapse ID -> {'presynapticCell': int, 'permanence': float}

        self._next_segment_id = 0
        self._next_synapse_id = 0
        self._segment_to_cell = dict()                    # Segment -> Cell

    def createSegment(self, cell):
        seg = self._next_segment_id
        self._next_segment_id += 1
        self._cell_to_segments[cell].append(seg)
        self._segment_to_cell[seg] = cell
        return seg

    def destroySegment(self, segment):
        cell = self._segment_to_cell.pop(segment)
        self._cell_to_segments[cell].remove(segment)

        for syn in self._segment_to_synapses.pop(segment, []):
            self._synapse_data.pop(syn, None)

    def segmentsForCell(self, cell):
        return list(self._cell_to_segments.get(cell, []))

    def cellForSegment(self, segment):
        return self._segment_to_cell[segment]

    def createSynapse(self, segment, presynapticCell, permanence):
        syn = self._next_synapse_id
        self._next_synapse_id += 1
        self._segment_to_synapses[segment].append(syn)
        self._synapse_data[syn] = {
            'presynapticCell': presynapticCell,
            'permanence': permanence
        }
        return syn

    def destroySynapse(self, synapse):
        for segment, syns in self._segment_to_synapses.items():
            if synapse in syns:
                syns.remove(synapse)
                break
        self._synapse_data.pop(synapse, None)

    def updateSynapsePermanence(self, synapse, permanence):
        self._synapse_data[synapse]['permanence'] = permanence

    def dataForSynapse(self, synapse):
        return self._synapse_data[synapse]

    def synapsesForSegment(self, segment):
        return list(self._segment_to_synapses.get(segment, []))

    def numSegments(self, cell):
        return len(self._cell_to_segments.get(cell, []))

    def numSynapses(self, segment):
        return len(self._segment_to_synapses.get(segment, []))

    def segmentFlatListLength(self):
        return self._next_segment_id

    def numCells(self):
        return self.num_cells

    def segmentsForColumn(self, column, cells_per_column):
        """
        WDND helper: return all segments for all cells in the column
        """
        start = column * cells_per_column
        end = start + cells_per_column
        segments = []
        for c in range(start, end):
            segments.extend(self._cell_to_segments.get(c, []))
        return segments

    def compareSegments(self, segA, segB):
        """
        WDND-style: define sort order based on segment ID (or cell, then index)
        """
        cellA = self.cellForSegment(segA)
        cellB = self.cellForSegment(segB)
        if cellA != cellB:
            return cellA - cellB
        return segA - segB
    
    def active_synapses(self, segment, active_presynaptic_cells):
        """
        Returns a list of synapse indices for synapses on the given segment
        whose presynaptic cells are active.
        """
        return [
            syn for syn in self.synapsesForSegment(segment)
            if self.dataForSynapse(syn)["presynapticCell"] in active_presynaptic_cells
        ]


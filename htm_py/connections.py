# htm_py/connections.py

from typing import List, Dict, Tuple, NamedTuple
import math

CellIdx = int
Segment = int
Synapse = int

class SynapseData(NamedTuple):
    presynapticCell: CellIdx
    permanence: float

class Connections:
    def __init__(self, num_cells: int):
        """
        Initialize the connections system with the specified number of cells.
        Raises ValueError if num_cells < 1.
        """
        if not isinstance(num_cells, int):
            raise TypeError("num_cells must be an integer.")
        if num_cells < 1:
            raise ValueError("Connections must have at least one cell.")

        self.num_cells = num_cells

        # Each cell maps to a list of segments
        self._segments_for_cell: Dict[CellIdx, List[Segment]] = {i: [] for i in range(num_cells)}

        # Each segment maps to its owning cell
        self._cell_for_segment: Dict[Segment, CellIdx] = {}

        # Each segment maps to a list of synapses
        self._synapses_for_segment: Dict[Segment, List[Synapse]] = {}

        # Each synapse maps to its SynapseData
        self._synapse_data: Dict[Synapse, SynapseData] = {}

        # Internal counters for unique IDs
        self._next_segment_id = 0
        self._next_synapse_id = 0

        # Note: Not thread-safe — external locking required in concurrent environments.


     # === Synapse Functions ===
    def createSynapse(self, segment: Segment, presynapticCell: CellIdx, permanence: float) -> Synapse:
        """
        Create a new synapse on a segment, pointing to a presynaptic cell.
        Raises IndexError if presynapticCell is out of range.
        Raises TypeError or ValueError for invalid arguments.
        """
        if not isinstance(presynapticCell, int):
            raise TypeError("Presynaptic cell index must be an integer.")
        if not (0 <= presynapticCell < self.num_cells):
            raise IndexError(f"Invalid presynaptic cell index {presynapticCell}")
        if not isinstance(permanence, float):
            raise TypeError("Permanence must be a float.")
        if not math.isfinite(permanence):
            raise ValueError("Permanence must be a finite number.")

        synapse = self._next_synapse_id
        if synapse in self._synapse_data:
            raise RuntimeError(f"Synapse ID reuse detected: {synapse}")
        self._next_synapse_id += 1

        self._synapses_for_segment[segment].append(synapse)
        self._synapse_data[synapse] = SynapseData(presynapticCell, permanence)

        return synapse

    def destroySynapse(self, synapse: Synapse) -> None:
        """
        Remove synapse from all tracking structures.
        Raises KeyError if synapse doesn't exist.
        """
        if synapse not in self._synapse_data:
            raise KeyError(f"Cannot destroy nonexistent synapse ID {synapse}")

        # Remove from synapse list on its segment
        found = False
        for seg, syn_list in self._synapses_for_segment.items():
            if synapse in syn_list:
                syn_list.remove(synapse)
                found = True
                break
        if not found:
            raise RuntimeError(f"Synapse {synapse} not found in any segment")

        # Remove from synapse data
        del self._synapse_data[synapse]

    def updateSynapsePermanence(self, synapse: Synapse, new_permanence: float) -> None:
        """
        Update the permanence value of a synapse.
        Raises KeyError if synapse doesn't exist.
        Raises ValueError if permanence is not finite.
        """
        if synapse not in self._synapse_data:
            raise KeyError(f"Synapse ID {synapse} not found")
        if not isinstance(new_permanence, float):
            raise TypeError("Permanence must be a float.")
        if not math.isfinite(new_permanence):
            raise ValueError("Permanence must be a finite number.")

        old = self._synapse_data[synapse]
        self._synapse_data[synapse] = SynapseData(old.presynapticCell, new_permanence)

    def synapsesForSegment(self, segment: Segment) -> List[Synapse]:
        """
        Return a COPY of all synapse IDs on the given segment.
        Raises KeyError if segment doesn't exist.
        """
        if segment not in self._synapses_for_segment:
            raise KeyError(f"Segment ID {segment} not found")
        return list(self._synapses_for_segment[segment])  # return copy

    def dataForSynapse(self, synapse: Synapse) -> SynapseData:
        """
        Return (presynapticCell, permanence) tuple for a synapse.
        Raises KeyError if synapse not found.
        """
        if synapse not in self._synapse_data:
            raise KeyError(f"Synapse ID {synapse} not found")
        return self._synapse_data[synapse]

    # === Segment & Cell Helpers ===
    def createSegment(self, cell: CellIdx) -> Segment:
        """
        Create a new segment on the given cell.
        Raises IndexError if cell is out of range.
        Raises TypeError if cell is not an integer.
        """
        if not isinstance(cell, int):
            raise TypeError("Cell index must be an integer.")
        if not (0 <= cell < self.num_cells):
            raise IndexError(f"Invalid cell index {cell}")

        segment = self._next_segment_id
        if segment in self._cell_for_segment:
            raise RuntimeError(f"Segment ID reuse detected: {segment}")
        self._next_segment_id += 1

        self._segments_for_cell[cell].append(segment)
        self._cell_for_segment[segment] = cell
        self._synapses_for_segment[segment] = []

        return segment

    def destroySegment(self, segment: Segment) -> None:
        """
        Remove a segment and all its synapses.
        Raises KeyError if segment doesn't exist.
        """
        if segment not in self._cell_for_segment:
            raise KeyError(f"Segment {segment} does not exist")

        cell = self._cell_for_segment[segment]

        # Remove all synapses on this segment
        for syn in list(self._synapses_for_segment[segment]):
            self.destroySynapse(syn)  # already validated

        # Clean up segment references
        self._segments_for_cell[cell].remove(segment)
        del self._cell_for_segment[segment]
        del self._synapses_for_segment[segment]

    # === Helpers ===
    def segments(self):
        """Return a list of all existing segment IDs."""
        return list(self._cell_for_segment.keys())

    def segmentsForCell(self, cell: CellIdx) -> List[Segment]:
            """
            Return a COPY of all segments owned by a cell.
            Raises IndexError if cell is out of range.
            """
            if not (0 <= cell < self.num_cells):
                raise IndexError(f"Invalid cell index {cell}")
            return list(self._segments_for_cell[cell])  # return copy
    
    def cell_for_segment(self, segment: Segment) -> CellIdx:
        """Public alias for retrieving the cell owning a segment."""
        if segment not in self._cell_for_segment:
            raise KeyError(f"Segment {segment} does not exist")
        return self._cell_for_segment[segment]

    def numSegments(self, cell: CellIdx) -> int:
        """
        Return the number of segments owned by a cell.
        """
        return len(self.segmentsForCell(cell))  # uses validated method

    def numSynapses(self, segment: Segment) -> int:
            """
            Return the number of synapses on a segment.
            """
            return len(self.synapsesForSegment(segment))  # uses validated method

    def numCells(self) -> int:
        """
        Return total number of initialized cells.
        """
        return self.num_cells

    def segmentFlatListLength(self) -> int:
        """
        Total number of live segments (used for index-aligned arrays).
        """
        return len(self._cell_for_segment)

    def active_synapses(self, segment, active_cells, connected_permanence):
        """
        Return a list of synapses on `segment` whose presynaptic cells are active
        and whose permanence is >= connected_permanence.
        """
        return [
            s for s in self.synapsesForSegment(segment)
            if self.dataForSynapse(s).presynapticCell in active_cells
            and self.dataForSynapse(s).permanence >= connected_permanence
        ]

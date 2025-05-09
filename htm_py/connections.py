import numpy as np

class Connections:
    def __init__(self):
        # Maps each cell to its list of segments
        self.cell_to_segments = {}

        # Maps each segment to a list of synapse data structures
        self.segment_to_synapses = {}

        # Synapse data: {synapse_id: (presynaptic_cell, permanence)}
        self.synapse_data = {}

        # Internal counters for unique segment and synapse IDs
        self._segment_id_counter = 0
        self._synapse_id_counter = 0


    def create_segment(self, cell):
        """
        Create a new segment on a given cell.

        Args:
            cell (int): Cell index.

        Returns:
            int: New segment ID.
        """
        segment_id = self._segment_id_counter
        self._segment_id_counter += 1

        self.cell_to_segments.setdefault(cell, []).append(segment_id)
        self.segment_to_synapses[segment_id] = []

        return segment_id


    def create_synapse(self, segment, presynaptic_cell, initial_permanence):
        """
        Create a new synapse on a segment.

        Args:
            segment (int): Segment ID.
            presynaptic_cell (int): Presynaptic cell index.
            initial_permanence (float): Initial permanence value.

        Returns:
            int: New synapse ID.
        """
        synapse_id = self._synapse_id_counter
        self._synapse_id_counter += 1

        self.synapse_data[synapse_id] = (presynaptic_cell, initial_permanence)
        self.segment_to_synapses[segment].append(synapse_id)

        return synapse_id


    def segments_for_cell(self, cell):
        """
        Returns all segments associated with a given cell.

        Args:
            cell (int): Cell index.

        Returns:
            list of int: Segment IDs.
        """
        return self.cell_to_segments.get(cell, [])


    def synapses_for_segment(self, segment):
        """
        Returns all synapses associated with a given segment.

        Args:
            segment (int): Segment ID.

        Returns:
            list of int: Synapse IDs.
        """
        return self.segment_to_synapses.get(segment, [])


    def num_segments(self, cell):
        """
        Returns the number of segments associated with a cell.

        Args:
            cell (int): Cell index.

        Returns:
            int: Number of segments.
        """
        return len(self.cell_to_segments.get(cell, []))


    def synapse_data_for(self, synapse_id):
        """
        Get the presynaptic cell and permanence for a given synapse.

        Args:
            synapse_id (int): Synapse ID.

        Returns:
            (int, float): (Presynaptic cell index, permanence value)
        """
        return self.synapse_data[synapse_id]


    def num_active_connected_synapses(self, segment, active_cells, connected_permanence):
        """
        Count how many connected synapses on a segment are active.

        Args:
            segment (int): Segment ID.
            active_cells (set of int): Currently active cells.
            connected_permanence (float): Permanence threshold for connection.

        Returns:
            int: Number of active connected synapses.
        """
        count = 0
        for synapse in self.synapses_for_segment(segment):
            presynaptic_cell, permanence = self.synapse_data_for(synapse)
            if permanence >= connected_permanence and presynaptic_cell in active_cells:
                count += 1
        return count


    def num_active_potential_synapses(self, segment, active_cells):
        """
        Count how many potential synapses on a segment are active (regardless of permanence).

        Args:
            segment (int): Segment ID.
            active_cells (set of int): Currently active cells.

        Returns:
            int: Number of active potential synapses.
        """
        count = 0
        for synapse in self.synapses_for_segment(segment):
            presynaptic_cell, _ = self.synapse_data_for(synapse)
            if presynaptic_cell in active_cells:
                count += 1
        return count


    def adapt_segment(self, segment, prev_active_cells, permanence_increment, permanence_decrement):
        """
        Adapt the permanence values of synapses on a segment based on previous active cells.

        Args:
            segment (int): Segment ID.
            prev_active_cells (set of int): Previously active cells at t-1.
            permanence_increment (float): Increment value for active synapses.
            permanence_decrement (float): Decrement value for inactive synapses.
        """
        for synapse in self.synapses_for_segment(segment):
            presynaptic_cell, permanence = self.synapse_data_for(synapse)
            if presynaptic_cell in prev_active_cells:
                permanence += permanence_increment
            else:
                permanence -= permanence_decrement

            # Clamp permanence between [0.0, 1.0]
            permanence = min(max(permanence, 0.0), 1.0)
            self.synapse_data[synapse] = (presynaptic_cell, permanence)


    def grow_synapses(self, segment, prev_winner_cells, initial_permanence, max_new_synapses):
        """
        Grow new synapses on a segment connecting to previous winner cells.

        Args:
            segment (int): Segment ID.
            prev_winner_cells (set of int): Winner cells from t-1 to connect to.
            initial_permanence (float): Permanence value for new synapses.
            max_new_synapses (int): Max number of synapses to grow.
        """
        # Find presynaptic cells that are not already connected by this segment
        existing_presynaptic = {
            self.synapse_data_for(synapse)[0] for synapse in self.synapses_for_segment(segment)
        }
        candidates = list(prev_winner_cells - existing_presynaptic)

        # Limit growth to max_new_synapses
        np.random.shuffle(candidates)
        for presynaptic_cell in candidates[:max_new_synapses]:
            self.create_synapse(segment, presynaptic_cell, initial_permanence)


    def destroy_synapse(self, synapse_id):
        """
        Removes a synapse completely from the model.

        Args:
            synapse_id (int): Synapse ID to destroy.
        """
        if synapse_id not in self.synapse_data:
            return  # Already removed

        presynaptic_cell, _ = self.synapse_data[synapse_id]
        del self.synapse_data[synapse_id]

        # Also remove from its segment
        for segment, synapses in self.segment_to_synapses.items():
            if synapse_id in synapses:
                synapses.remove(synapse_id)
                break  # Synapse found and removed


    def destroy_segment(self, segment_id):
        """
        Removes a segment and all its associated synapses.

        Args:
            segment_id (int): Segment ID to destroy.
        """
        if segment_id not in self.segment_to_synapses:
            return  # Already removed

        # Remove all synapses tied to this segment
        for synapse_id in list(self.segment_to_synapses[segment_id]):
            self.destroy_synapse(synapse_id)

        # Remove the segment from the owning cell
        for cell, segments in self.cell_to_segments.items():
            if segment_id in segments:
                segments.remove(segment_id)
                break

        # Finally, remove the segment entry itself
        del self.segment_to_synapses[segment_id]


    def segments(self):
        """
        Returns all existing segment IDs in the model.

        Returns:
            list of int: Segment IDs.
        """
        return list(self.segment_to_synapses.keys())


    def is_cell_predictive(self, cell, active_segments):
        """
        Determines if a cell is in a predictive state.

        Args:
            cell (int): Cell index.
            active_segments (set or list of int): Currently active segment IDs.

        Returns:
            bool: True if any of the cell's segments are active.
        """
        return any(segment in active_segments for segment in self.segments_for_cell(cell))


    def matching_segments_for_column(self, column, cells_per_column, active_cells, min_threshold):
        """
        Finds matching segments for the given column.

        Args:
            column (int): Column index.
            cells_per_column (int): Cells per column.
            active_cells (set of int): Previously active cells at t-1.
            min_threshold (int): Min active potential synapses for a segment to match.

        Returns:
            list of int: Matching segment IDs.
        """
        matching_segments = []
        start_cell = column * cells_per_column
        end_cell = start_cell + cells_per_column

        for cell in range(start_cell, end_cell):
            for segment in self.segments_for_cell(cell):
                active_potentials = self.num_active_potential_synapses(segment, active_cells)
                if active_potentials >= min_threshold:
                    matching_segments.append(segment)

        return matching_segments


    def cell_for_segment(self, segment_id):
        """
        Returns the cell that owns the given segment.

        Args:
            segment_id (int): Segment ID.

        Returns:
            int: Cell index.
        """
        for cell, segments in self.cell_to_segments.items():
            if segment_id in segments:
                return cell
        raise ValueError(f"Segment ID {segment_id} not found in any cell.")


    def column_for_cell(self, cell, cells_per_column):
        """
        Returns the column index for a given cell index.

        Args:
            cell (int): Cell index.
            cells_per_column (int): Number of cells per column.

        Returns:
            int: Column index.
        """
        return cell // cells_per_column

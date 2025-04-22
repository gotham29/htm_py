import numpy as np

class TemporalMemory:
    def __init__(
        self,
        column_dim,
        cells_per_column=32,
        activation_threshold=12,
        initial_permanence=0.21,
        connected_permanence=0.5,
        permanence_increment=0.1,
        permanence_decrement=0.1,
        max_segments_per_cell=128,
        max_synapses_per_segment=128,
    ):
        self.column_dim = column_dim
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment

        self.prev_active_cells = set()
        self.segments = {}

    def _cell_index(self, col, cell):
        return col * self.cells_per_column + cell

    def compute(self, active_columns):
        active_cells = set()
        predictive_cells = set()
        
        for col in active_columns:
            predicted = False
            for cell in range(self.cells_per_column):
                idx = self._cell_index(col, cell)
                if idx in self.segments:
                    for synapse_set in self.segments[idx]:
                        overlap = len(synapse_set & self.prev_active_cells)
                        if overlap >= self.activation_threshold:
                            predicted = True
                            active_cells.add(idx)
                            predictive_cells.add(idx)
                            break
            if not predicted:
                for cell in range(self.cells_per_column):
                    active_cells.add(self._cell_index(col, cell))

        for idx in active_cells:
            if idx not in self.segments:
                self.segments[idx] = []
            if len(self.segments[idx]) < self.max_segments_per_cell:
                new_synapse_set = set(np.random.choice(
                    list(active_cells),
                    size=min(self.max_synapses_per_segment, len(active_cells)),
                    replace=False))
                self.segments[idx].append(new_synapse_set)

        self.prev_active_cells = active_cells
        anomaly_score = 1.0 - (len(predictive_cells & active_cells) / len(active_cells)) if active_cells else 1.0
        prediction_count = len(predictive_cells)
        return anomaly_score, prediction_count
